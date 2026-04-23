"""
PINN-LISTA-LNN for NILM — REDD specific splits.

Combines LISTA (Learned ISTA / algorithm unrolling) with a Physics
Consistency Loss (PINN condition) that enforces:

    Σ p̂_i ≤ P_agg + ε   (sum of appliance predictions ≤ total mains)

All 4 appliances are predicted simultaneously (multi-head) so the full
energy conservation constraint can be applied.

Architecture:
    y  (batch, WIN, 1)
     → BasicLiquidTimeLayer (fixed tau)  →  h  (batch, hidden)   [LNN encoder]
     → Linear projection h → x^(0)                               [warm start]
     → K × LISTALayer      →  x^(K)  (batch, n_atoms)            [sparse refinement]
     → 4 × Linear heads    →  p̂     (batch, 4)
    Dictionary D (n_atoms, WIN) jointly learned for reconstruction.

Loss:
    Stage 1 (epochs 1–WARMUP_EPOCHS): MSE + λ_recon·Recon + λ_sparse·L1
    Stage 2 (remaining epochs):       + λ_phys·L_phys + per-appliance BCE
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm
import pickle
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source Code'))
from utils import calculate_nilm_metrics, save_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCHS      = 80
PATIENCE    = 20
LR          = 1e-3
BATCH       = 32
WIN         = 100
STRIDE      = 5
N_ATOMS     = 64
K_LAYERS    = 8
HIDDEN_SIZE = 64
DT          = 0.1

LAMBDA_RECON  = 0.1
LAMBDA_SPARSE = 0.01
LAMBDA_PHYS   = 0.01
EPSILON_W     = 50.0
WARMUP_EPOCHS = 20

APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'microwave':    10.0,
    'washer dryer':  0.5,
}

BCE_LAMBDA = {'dish washer': 0.3, 'fridge': 0.5, 'microwave': 2.0, 'washer dryer': 2.0}
BCE_ALPHA  = {'dish washer': 0.5, 'fridge': 0.5, 'microwave': 3.0, 'washer dryer': 2.0}


# ---------------------------------------------------------------------------
# Physics Consistency Loss
# ---------------------------------------------------------------------------

class PhysicsConsistencyLoss(nn.Module):
    """
    Soft one-sided penalty:  ReLU(Σ p̂_i_raw - P_agg_raw - ε)

    All arithmetic in raw Watts via differentiable MinMaxScaler inverse:
        x_raw = x_scaled * data_range_ + data_min_
    """

    def __init__(self, x_scaler, y_scalers, appliances, epsilon_w=EPSILON_W):
        super().__init__()
        self.epsilon = epsilon_w

        self.register_buffer('x_min',   torch.tensor(float(x_scaler.data_min_[0]),   dtype=torch.float32))
        self.register_buffer('x_range', torch.tensor(float(x_scaler.data_range_[0]), dtype=torch.float32))

        y_mins   = [float(y_scalers[i].data_min_[0])   for i in range(len(appliances))]
        y_ranges = [float(y_scalers[i].data_range_[0]) for i in range(len(appliances))]
        self.register_buffer('y_mins',   torch.tensor(y_mins,   dtype=torch.float32))
        self.register_buffer('y_ranges', torch.tensor(y_ranges, dtype=torch.float32))

    def forward(self, x_mid_scaled: torch.Tensor, pred_scaled: torch.Tensor) -> torch.Tensor:
        x_raw     = x_mid_scaled * self.x_range + self.x_min
        p_raw     = pred_scaled  * self.y_ranges + self.y_mins
        p_sum     = p_raw.sum(dim=1)
        violation = F.relu(p_sum - x_raw - self.epsilon)
        return violation.mean()


# ---------------------------------------------------------------------------
# Soft threshold activation
# ---------------------------------------------------------------------------

def soft_threshold(z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    return torch.sign(z) * torch.clamp(torch.abs(z) - lam, min=0.0)


# ---------------------------------------------------------------------------
# LISTA layer
# ---------------------------------------------------------------------------

class LISTALayer(nn.Module):
    def __init__(self, signal_len: int, n_atoms: int):
        super().__init__()
        self.We        = nn.Linear(signal_len, n_atoms, bias=True)
        self.Wr        = nn.Linear(n_atoms,    n_atoms, bias=False)
        self.threshold = nn.Parameter(torch.full((n_atoms,), 0.1))

    def forward(self, y: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        z = self.We(y) + self.Wr(x_prev)
        return soft_threshold(z, F.softplus(self.threshold))


# ---------------------------------------------------------------------------
# PINN-LISTA-LNN model
# ---------------------------------------------------------------------------

class PINNLISTALNNModel(nn.Module):
    """
    LNN temporal encoder → LISTA sparse refinement → per-appliance heads.

    Forward pass:
        1. BasicLiquidTimeLayer processes (batch, WIN, 1) sequentially
           → hidden state h  (batch, hidden_size)
        2. h is projected to x^(0) as a warm start for LISTA
        3. K LISTA layers refine x^(0) → x^(K)  (batch, n_atoms)
        4. Per-appliance linear heads → p̂  (batch, n_appliances)

    LNN cell (Basic, fixed tau):
        tau   = softplus(tau_param)
        f_t   = tanh(W_in·x_t + W_rec·h)
        dh    = (-h / tau + f_t) * dt
        h_new = clamp(h + dh, -10, 10)
    """

    def __init__(self, signal_len: int = WIN,
                 hidden_size: int = HIDDEN_SIZE,
                 n_atoms: int = N_ATOMS,
                 k_layers: int = K_LAYERS,
                 n_appliances: int = len(APPLIANCES),
                 dt: float = DT):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_atoms     = n_atoms
        self.dt          = dt

        self.lnn_input = nn.Linear(1, hidden_size)
        self.lnn_tau   = nn.Parameter(torch.ones(hidden_size))
        self.lnn_rec   = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.lnn_rec)
        self.lnn_norm  = nn.LayerNorm(hidden_size)

        self.h_to_x0 = nn.Linear(hidden_size, n_atoms)

        self.lista_layers = nn.ModuleList([
            LISTALayer(signal_len, n_atoms) for _ in range(k_layers)
        ])

        self.D = nn.Parameter(torch.randn(n_atoms, signal_len) * 0.01)

        self.heads = nn.ModuleList([
            nn.Linear(n_atoms, 1) for _ in range(n_appliances)
        ])

    def forward(self, y: torch.Tensor):
        if y.dim() == 2:
            y = y.unsqueeze(-1)

        batch_size, seq_len, _ = y.size()

        h   = torch.zeros(batch_size, self.hidden_size, device=y.device)
        tau = F.softplus(self.lnn_tau).unsqueeze(0)

        for t in range(seq_len):
            x_t = y[:, t, :]
            f_t = torch.tanh(self.lnn_norm(
                      self.lnn_input(x_t) + torch.matmul(h, self.lnn_rec)))
            dh  = (-h / tau + f_t) * self.dt
            h   = (h + dh).clamp(-10.0, 10.0)

        x      = self.h_to_x0(h)
        y_flat = y.squeeze(-1)
        for layer in self.lista_layers:
            x = layer(y_flat, x)

        power = torch.cat([head(x) for head in self.heads], dim=1)
        return power, x

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.D


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiApplianceDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_redd_specific_splits():
    print("Loading REDD data with specific splits...")
    with open('data/redd/train_small.pkl', 'rb') as f:
        train_data = pickle.load(f)[0]
    with open('data/redd/val_small.pkl', 'rb') as f:
        val_data = pickle.load(f)[0]
    with open('data/redd/test_small.pkl', 'rb') as f:
        test_data = pickle.load(f)[0]

    print(f"Train date range: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Val   date range: {val_data.index.min()} to {val_data.index.max()}")
    print(f"Test  date range: {test_data.index.min()} to {test_data.index.max()}")
    print(f"Available columns: {list(train_data.columns)}")
    return {'train': train_data, 'val': val_data, 'test': test_data}


def create_sequences(data, window_size=WIN):
    mains    = data['main'].values
    app_vals = {app: data[app].values for app in APPLIANCES}
    X, Y = [], []
    for i in range(0, len(mains) - window_size, STRIDE):
        X.append(mains[i:i + window_size])
        mid = i + window_size // 2
        Y.append([app_vals[app][mid] for app in APPLIANCES])
    return (
        np.array(X, dtype=np.float32).reshape(-1, window_size, 1),
        np.array(Y, dtype=np.float32),
    )


def compute_per_appliance_metrics(y_true, y_pred, y_scalers):
    metrics = {}
    for i, app in enumerate(APPLIANCES):
        raw_true = y_scalers[i].inverse_transform(y_true[:, i:i+1]).flatten()
        raw_pred = y_scalers[i].inverse_transform(y_pred[:, i:i+1]).flatten()
        metrics[app] = calculate_nilm_metrics(raw_true, raw_pred, threshold=THRESHOLDS[app])
    return metrics


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_pinn_lista_model(data_dict, save_dir,
                           hidden_size=HIDDEN_SIZE, dt=DT,
                           n_atoms=N_ATOMS, k_layers=K_LAYERS,
                           lambda_recon=LAMBDA_RECON,
                           lambda_sparse=LAMBDA_SPARSE,
                           lambda_phys=LAMBDA_PHYS,
                           epsilon_w=EPSILON_W):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"hidden={hidden_size}  dt={dt}  n_atoms={n_atoms}  K={k_layers}  "
          f"λ_recon={lambda_recon}  λ_sparse={lambda_sparse}  "
          f"λ_phys={lambda_phys}  ε={epsilon_w}W")

    X_tr, Y_tr = create_sequences(data_dict['train'], WIN)
    X_va, Y_va = create_sequences(data_dict['val'],   WIN)
    X_te, Y_te = create_sequences(data_dict['test'],  WIN)

    x_scaler = MinMaxScaler()
    X_tr = x_scaler.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_va = x_scaler.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te = x_scaler.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_tr[:, i:i+1] = ys.fit_transform(Y_tr[:, i:i+1])
        Y_va[:, i:i+1] = ys.transform(Y_va[:, i:i+1])
        Y_te[:, i:i+1] = ys.transform(Y_te[:, i:i+1])
        y_scalers.append(ys)

    thresholds_scaled = [
        (THRESHOLDS[app] - float(y_scalers[i].data_min_[0]))
        / float(y_scalers[i].data_range_[0])
        for i, app in enumerate(APPLIANCES)
    ]

    print(f"Train: {X_tr.shape} → {Y_tr.shape}")
    print(f"Val:   {X_va.shape} → {Y_va.shape}")
    print(f"Test:  {X_te.shape} → {Y_te.shape}")

    tr_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_va, Y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_te, Y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    model = PINNLISTALNNModel(
        signal_len=WIN, hidden_size=hidden_size, dt=dt,
        n_atoms=n_atoms, k_layers=k_layers,
        n_appliances=len(APPLIANCES)
    ).to(device)

    phys_criterion = PhysicsConsistencyLoss(
        x_scaler, y_scalers, APPLIANCES, epsilon_w=epsilon_w
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    history = {
        'train_loss': [], 'train_mse': [], 'train_phys': [],
        'val_loss':   [], 'val_mse':   [], 'val_phys':   [],
        'val_metrics': [],
    }
    best_val_loss = float('inf')
    best_state    = None
    counter       = 0

    print("Starting PINN-LISTA-LNN training (all appliances simultaneously)...")

    for epoch in range(EPOCHS):
        model.train()
        ep_mse = ep_phys = ep_total = 0.0
        bar = tqdm(tr_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for xb, yb in bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            power, x    = model(xb)
            mse_loss    = F.mse_loss(power, yb)
            y_flat      = xb.squeeze(-1)
            recon_loss  = F.mse_loss(model.reconstruct(x), y_flat)
            sparse_loss = x.abs().mean()

            if epoch < WARMUP_EPOCHS:
                loss = mse_loss + lambda_recon * recon_loss + lambda_sparse * sparse_loss
                phys_loss = torch.tensor(0.0, device=device)
            else:
                x_mid     = xb[:, WIN // 2, 0]
                phys_loss = phys_criterion(x_mid, power)

                bce_loss = torch.tensor(0.0, device=device)
                for i, app in enumerate(APPLIANCES):
                    if BCE_LAMBDA[app] > 0:
                        pred_prob = torch.sigmoid(
                            power[:, i:i+1] / (thresholds_scaled[i] + 1e-8))
                        y_bin = (yb[:, i:i+1] > thresholds_scaled[i]).float()
                        w     = torch.where(y_bin == 1,
                                            torch.full_like(y_bin, BCE_ALPHA[app]),
                                            torch.ones_like(y_bin))
                        bce_loss = bce_loss + BCE_LAMBDA[app] * F.binary_cross_entropy(
                            pred_prob.clamp(1e-7, 1 - 1e-7), y_bin, weight=w)

                loss = (mse_loss
                        + lambda_recon  * recon_loss
                        + lambda_sparse * sparse_loss
                        + lambda_phys   * phys_loss
                        + bce_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_mse   += mse_loss.item()
            ep_phys  += phys_loss.item()
            ep_total += loss.item()
            bar.set_postfix({'mse': f'{mse_loss.item():.5f}',
                             'phys': f'{phys_loss.item():.5f}'})

        avg_tr_mse   = ep_mse   / len(tr_loader)
        avg_tr_phys  = ep_phys  / len(tr_loader)
        avg_tr_total = ep_total / len(tr_loader)
        history['train_mse'].append(avg_tr_mse)
        history['train_phys'].append(avg_tr_phys)
        history['train_loss'].append(avg_tr_total)

        model.eval()
        vl_mse = vl_phys = vl_total = 0.0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                power, x    = model(xb)
                mse_loss    = F.mse_loss(power, yb)
                y_flat      = xb.squeeze(-1)
                recon_loss  = F.mse_loss(model.reconstruct(x), y_flat)
                sparse_loss = x.abs().mean()
                x_mid       = xb[:, WIN // 2, 0]
                phys_loss   = phys_criterion(x_mid, power)
                loss        = (mse_loss
                               + lambda_recon  * recon_loss
                               + lambda_sparse * sparse_loss
                               + lambda_phys   * phys_loss)

                vl_mse   += mse_loss.item()
                vl_phys  += phys_loss.item()
                vl_total += loss.item()
                val_preds.append(power.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        avg_va_mse   = vl_mse   / len(va_loader)
        avg_va_phys  = vl_phys  / len(va_loader)
        avg_va_total = vl_total / len(va_loader)
        history['val_mse'].append(avg_va_mse)
        history['val_phys'].append(avg_va_phys)
        history['val_loss'].append(avg_va_total)
        scheduler.step(avg_va_mse)

        y_pred_all = np.concatenate(val_preds)
        y_true_all = np.concatenate(val_trues)
        per_app_metrics = compute_per_appliance_metrics(y_true_all, y_pred_all, y_scalers)
        history['val_metrics'].append(per_app_metrics)

        avg_f1  = np.mean([per_app_metrics[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([per_app_metrics[a]['mae'] for a in APPLIANCES])

        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train={avg_tr_total:.5f} (mse={avg_tr_mse:.5f} phys={avg_tr_phys:.5f})  "
              f"val={avg_va_total:.5f} (mse={avg_va_mse:.5f} phys={avg_va_phys:.5f})  "
              f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")
        for app in APPLIANCES:
            m = per_app_metrics[app]
            print(f"    {app:<14}  F1={m['f1']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"MAE={m['mae']:.2f}  SAE={m['sae']:.4f}")

        if avg_va_mse < best_val_loss:
            best_val_loss = avg_va_mse
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            counter       = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print("Training completed!")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_preds, test_trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            power, _ = model(xb.to(device))
            test_preds.append(power.cpu().numpy())
            test_trues.append(yb.numpy())

    y_pred_te = np.concatenate(test_preds)
    y_true_te = np.concatenate(test_trues)
    test_metrics = compute_per_appliance_metrics(y_true_te, y_pred_te, y_scalers)

    print(f"\n{'Appliance':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} "
          f"{'MAE':>8} {'SAE':>8}")
    print("-" * 65)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"{app:<15} {m['f1']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")

    _plot_training(history, test_metrics, save_dir)

    config = {
        'dataset': 'REDD',
        'model': 'PINNLISTALNNModel',
        'description': 'BasicLNN encoder (warm-start) → K-layer LISTA + PhysicsConsistency (Σp̂ ≤ P_agg + ε)',
        'loss': 'MSE + λ_recon·Recon + λ_sparse·L1 + λ_phys·Phys [+ BCE stage2]',
        'window_size': WIN,
        'model_params': {
            'signal_len': WIN, 'hidden_size': hidden_size, 'dt': dt,
            'n_atoms': n_atoms, 'k_layers': k_layers,
            'n_appliances': len(APPLIANCES),
        },
        'train_params': {
            'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
            'lambda_recon': lambda_recon, 'lambda_sparse': lambda_sparse,
            'lambda_phys': lambda_phys, 'epsilon_w': epsilon_w,
            'warmup_epochs': WARMUP_EPOCHS,
            'bce_lambda': BCE_LAMBDA, 'bce_alpha': BCE_ALPHA,
        },
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
    }
    with open(os.path.join(save_dir, 'pinn_lista_lnn_redd_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_training(history, test_metrics, save_dir):
    epochs_x = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_x, history['train_loss'], label='Train total', color='blue')
    plt.plot(epochs_x, history['val_loss'],   label='Val total',   color='red')
    plt.title('Total Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_x, history['train_mse'], label='Train MSE', color='blue')
    plt.plot(epochs_x, history['val_mse'],   label='Val MSE',   color='red')
    plt.title('MSE Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_x, history['train_phys'], label='Train Phys', color='blue')
    plt.plot(epochs_x, history['val_phys'],   label='Val Phys',   color='red')
    plt.title('Physics Consistency Loss')
    plt.xlabel('Epoch'); plt.ylabel('L_phys')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lista_lnn_redd_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('PINN-LISTA-LNN REDD — Per-Appliance Val Metrics', fontsize=13)

    for row, app in enumerate(APPLIANCES):
        f1_series  = [m[app]['f1']  for m in history['val_metrics']]
        mae_series = [m[app]['mae'] for m in history['val_metrics']]

        axes[row][0].plot(epochs_x, f1_series, color='blue', linewidth=1.5)
        axes[row][0].axhline(test_metrics[app]['f1'], color='green',
                             linestyle='--', label='Test F1')
        axes[row][0].set_title(f'{app} — F1')
        axes[row][0].set_xlabel('Epoch'); axes[row][0].set_ylabel('F1')
        axes[row][0].legend(); axes[row][0].grid(True, alpha=0.3)

        axes[row][1].plot(epochs_x, mae_series, color='red', linewidth=1.5)
        axes[row][1].axhline(test_metrics[app]['mae'], color='green',
                             linestyle='--', label='Test MAE')
        axes[row][1].set_title(f'{app} — MAE (W)')
        axes[row][1].set_xlabel('Epoch'); axes[row][1].set_ylabel('MAE (W)')
        axes[row][1].legend(); axes[row][1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lista_lnn_redd_per_appliance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for f in ['data/redd/train_small.pkl', 'data/redd/val_small.pkl',
              'data/redd/test_small.pkl']:
        if not os.path.exists(f):
            print(f"Error: {f} not found!")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir  = f"models/pinn_lista_lnn_redd_{timestamp}"

    data_dict = load_redd_specific_splits()

    test_metrics, history = train_pinn_lista_model(
        data_dict,
        save_dir      = save_dir,
        hidden_size   = HIDDEN_SIZE,
        dt            = DT,
        n_atoms       = N_ATOMS,
        k_layers      = K_LAYERS,
        lambda_recon  = LAMBDA_RECON,
        lambda_sparse = LAMBDA_SPARSE,
        lambda_phys   = LAMBDA_PHYS,
        epsilon_w     = EPSILON_W,
    )

    print(f"\nResults saved to {save_dir}")
