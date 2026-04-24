"""
Physics-Informed LTC LNN (PINN-LTC) for NILM -- AMPds enriched splits (P + Q).

Same as test_pinn_basic_lnn_ampds_enriched_splits.py except the cell uses
input-dependent tau (full Liquid Time-Constant network) instead of a fixed
learned scalar.

  Fixed tau (basic):      tau   = softplus(tau_param)        -- constant after training
  Input-dependent tau:    tau_t = softplus(W_tau · x_t + b)  -- recomputed every step

This lets the cell self-regulate its memory timescale based on the current
signal: fast tau on transients (device switching), slow tau on steady-state
hold periods.

Architecture:
    Input (batch, WIN, 2)  -- scaled [P, Q] mains window
         |
    LTC cell (input-dependent tau)
         |
    LayerNorm(hidden)
         |
    +----+----+----+----+
    | DW | FR | BS | HP |  -- one Linear head per appliance
    +----+----+----+----+
    output: (batch, 4)

Loss:
    Stage 1 (epochs 1-WARMUP_EPOCHS): MSE only
    Stage 2 (remaining epochs):       MSE + lambda * L_phys + weighted BCE

Physics constraint (active-power only, channel 0):
    ReLU(sum_i p_hat_i_raw - P_agg_raw - epsilon) = 0
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

EPOCHS        = 80
PATIENCE      = 20
LR            = 1e-3
BATCH         = 32
WIN           = 100
STRIDE        = 5
INPUT_SIZE    = 2        # main (W)  +  main_Q (VAR)

LAMBDA_PHYS   = 0.01
EPSILON_W     = 50.0
WARMUP_EPOCHS = 20

DATA_DIR = 'data/AMPds_enriched_data'

APPLIANCES = ['dish washer', 'fridge', 'basement', 'heat pump']

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'basement':     10.0,
    'heat pump':    10.0,
}

BCE_LAMBDA = {
    'dish washer':  0.3,
    'fridge':       0.3,
    'basement':     0.3,
    'heat pump':    0.5,
}

BCE_ALPHA = {
    'dish washer':  1.5,
    'fridge':       1.5,
    'basement':     4.0,
    'heat pump':    5.0,
}


# ---------------------------------------------------------------------------
# Physics Consistency Loss  (active-power channel only)
# ---------------------------------------------------------------------------

class PhysicsConsistencyLoss(nn.Module):
    """
    Soft one-sided penalty:  ReLU(sum_i p_hat_i_raw - P_agg_raw - epsilon)

    x_mid_scaled comes from channel 0 (active power P).
    x_scaler_P is the MinMaxScaler fitted on the P channel.
    """

    def __init__(self, x_scaler_P, y_scalers, appliances, epsilon_w=EPSILON_W):
        super().__init__()
        self.epsilon = epsilon_w

        x_min   = float(x_scaler_P.data_min_[0])
        x_range = float(x_scaler_P.data_range_[0])
        self.register_buffer('x_min',   torch.tensor(x_min,   dtype=torch.float32))
        self.register_buffer('x_range', torch.tensor(x_range, dtype=torch.float32))

        y_mins   = [float(y_scalers[i].data_min_[0])   for i in range(len(appliances))]
        y_ranges = [float(y_scalers[i].data_range_[0]) for i in range(len(appliances))]
        self.register_buffer('y_mins',   torch.tensor(y_mins,   dtype=torch.float32))
        self.register_buffer('y_ranges', torch.tensor(y_ranges, dtype=torch.float32))

    def forward(self, x_mid_scaled, pred_scaled):
        x_raw     = x_mid_scaled * self.x_range + self.x_min   # (batch,)
        p_raw     = pred_scaled  * self.y_ranges + self.y_mins  # (batch, n_apps)
        p_sum     = p_raw.sum(dim=1)
        violation = F.relu(p_sum - x_raw - self.epsilon)
        return violation.mean()


# ---------------------------------------------------------------------------
# LTC LNN Model (input-dependent tau)
# ---------------------------------------------------------------------------

class PhysicsInformedLTCLiquidNetworkModel(nn.Module):
    """
    LTC cell with input-dependent tau -> per-appliance linear heads.

    Cell update:
        tau_t     = softplus(W_tau · x_t + b_tau)   -- recomputed every step
        f_t       = tanh(LayerNorm(W_in·x_t + W_rec·h))
        dh        = (-h / tau_t + f_t) * dt
        h_new     = clamp(h + dh, -10, 10)

    Compared to BasicLNN, the only change is that tau_t is a learned linear
    function of the current input x_t rather than a fixed parameter vector.
    """

    def __init__(self, input_size, hidden_size, n_appliances, dt=0.1):
        super().__init__()
        self.hidden_size  = hidden_size
        self.n_appliances = n_appliances
        self.dt           = dt

        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.W_tau       = nn.Linear(input_size, hidden_size)   # input-dependent tau
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)

        self.intra_norm = nn.LayerNorm(hidden_size)
        self.norm       = nn.LayerNorm(hidden_size)

        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_appliances)
        ])

    def forward(self, x):
        """x: (batch, seq_len, input_size)"""
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            x_t        = x[:, t, :]
            tau_t      = F.softplus(self.W_tau(x_t))           # (batch, hidden)
            input_proj = self.input_proj(x_t)
            rec_proj   = torch.matmul(h, self.rec_weights)
            f_t        = torch.tanh(self.intra_norm(input_proj + rec_proj))
            dh         = (-h / tau_t + f_t) * self.dt
            h          = (h + dh).clamp(-10.0, 10.0)

        h = self.norm(h)
        return torch.cat([head(h) for head in self.heads], dim=1)


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

def load_data():
    print("Loading AMPds enriched data (P + Q)...")
    splits = {}
    for split in ('train', 'val', 'test'):
        with open(os.path.join(DATA_DIR, f'{split}.pkl'), 'rb') as f:
            splits[split] = pickle.load(f)[0]

    for name, df in splits.items():
        print(f"  {name}: {df.shape}  {df.index.min()} -> {df.index.max()}")
    print(f"  Columns: {list(splits['train'].columns)}")
    return splits


def create_sequences(data, window_size=WIN):
    """Two-channel input [P, Q]; midpoint targeting for appliance labels."""
    mains = np.stack(
        [data['main'].values, data['main_Q'].values], axis=-1
    ).astype(np.float32)                                    # (T, 2)
    app_vals = {app: data[app].values for app in APPLIANCES}
    X, Y = [], []
    for i in range(0, len(mains) - window_size, STRIDE):
        X.append(mains[i:i + window_size])                  # (WIN, 2)
        mid = i + window_size // 2
        Y.append([app_vals[app][mid] for app in APPLIANCES])
    return (
        np.array(X, dtype=np.float32),                      # (N, WIN, 2)
        np.array(Y, dtype=np.float32),                      # (N, 4)
    )


def _scale_X(X_tr, X_va, X_te):
    """Per-channel MinMaxScaler on train; returns scaled arrays + list of scalers."""
    n_tr, n_va, n_te = X_tr.shape[0], X_va.shape[0], X_te.shape[0]
    X_tr_n = X_tr.copy()
    X_va_n = X_va.copy()
    X_te_n = X_te.copy()
    scalers = []
    for ch in range(INPUT_SIZE):
        sc = MinMaxScaler()
        X_tr_n[:, :, ch] = sc.fit_transform(
            X_tr[:, :, ch].reshape(-1, 1)).reshape(n_tr, WIN)
        X_va_n[:, :, ch] = sc.transform(
            X_va[:, :, ch].reshape(-1, 1)).reshape(n_va, WIN)
        X_te_n[:, :, ch] = sc.transform(
            X_te[:, :, ch].reshape(-1, 1)).reshape(n_te, WIN)
        scalers.append(sc)
    return X_tr_n, X_va_n, X_te_n, scalers


# ---------------------------------------------------------------------------
# Per-appliance metrics helper
# ---------------------------------------------------------------------------

def compute_per_appliance_metrics(y_true, y_pred, y_scalers):
    metrics = {}
    for i, app in enumerate(APPLIANCES):
        raw_true = y_scalers[i].inverse_transform(
            y_true[:, i:i+1]).flatten()
        raw_pred = y_scalers[i].inverse_transform(
            y_pred[:, i:i+1]).flatten()
        metrics[app] = calculate_nilm_metrics(
            raw_true, raw_pred, threshold=THRESHOLDS[app])
    return metrics


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_pinn_model(data_dict, save_dir,
                     hidden_size=64, dt=0.1,
                     lambda_phys=LAMBDA_PHYS, epsilon_w=EPSILON_W):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"lambda_phys={lambda_phys}  epsilon={epsilon_w} W  hidden={hidden_size}  dt={dt}")

    # ── Sequences ──
    X_tr, Y_tr = create_sequences(data_dict['train'], WIN)
    X_va, Y_va = create_sequences(data_dict['val'],   WIN)
    X_te, Y_te = create_sequences(data_dict['test'],  WIN)

    # ── Scale X per channel; keep x_scalers[0] for physics loss (P channel) ──
    X_tr, X_va, X_te, x_scalers = _scale_X(X_tr, X_va, X_te)

    # ── Scale Y per appliance ──
    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_tr[:, i:i+1] = ys.fit_transform(Y_tr[:, i:i+1])
        Y_va[:, i:i+1] = ys.transform(Y_va[:, i:i+1])
        Y_te[:, i:i+1] = ys.transform(Y_te[:, i:i+1])
        y_scalers.append(ys)

    thresholds_scaled = []
    for i, app in enumerate(APPLIANCES):
        r = float(y_scalers[i].data_range_[0])
        if r == 0.0:
            thresholds_scaled.append(float('inf'))  # all-zero appliance; BCE always off
        else:
            thresholds_scaled.append(
                (THRESHOLDS[app] - float(y_scalers[i].data_min_[0])) / r
            )

    print(f"Train: {X_tr.shape} -> {Y_tr.shape}")
    print(f"Val:   {X_va.shape} -> {Y_va.shape}")
    print(f"Test:  {X_te.shape} -> {Y_te.shape}")

    tr_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_va, Y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_te, Y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    # ── Model + losses ──
    model = PhysicsInformedLTCLiquidNetworkModel(
        input_size=INPUT_SIZE, hidden_size=hidden_size,
        n_appliances=len(APPLIANCES), dt=dt,
    ).to(device)

    mse_criterion  = nn.MSELoss()
    phys_criterion = PhysicsConsistencyLoss(
        x_scalers[0], y_scalers, APPLIANCES, epsilon_w=epsilon_w
    ).to(device)

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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print("Starting PINN-LTC training (all appliances simultaneously)...")

    for epoch in range(EPOCHS):
        # ── Training ──
        model.train()
        ep_mse = ep_phys = ep_total = 0.0
        progress_bar = tqdm(tr_loader,
                            desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for xb, yb in progress_bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            pred = model(xb)                                # (batch, n_apps)

            mse_loss  = mse_criterion(pred, yb)
            x_mid     = xb[:, WIN // 2, 0]                 # channel 0 = P
            phys_loss = phys_criterion(x_mid, pred)

            if epoch < WARMUP_EPOCHS:
                loss = mse_loss
            else:
                bce_loss = torch.tensor(0.0, device=device)
                for i, app in enumerate(APPLIANCES):
                    if BCE_LAMBDA[app] > 0:
                        pred_i = pred[:, i].clamp(1e-7, 1 - 1e-7)
                        thr_s  = thresholds_scaled[i]
                        y_bin  = (yb[:, i] > thr_s).float()
                        w      = torch.where(y_bin == 1,
                                             torch.full_like(y_bin, BCE_ALPHA[app]),
                                             torch.ones_like(y_bin))
                        bce_loss = bce_loss + BCE_LAMBDA[app] * F.binary_cross_entropy(
                            pred_i, y_bin, weight=w)
                loss = mse_loss + lambda_phys * phys_loss + bce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_mse   += mse_loss.item()
            ep_phys  += phys_loss.item()
            ep_total += loss.item()
            progress_bar.set_postfix({
                'mse':  f'{mse_loss.item():.5f}',
                'phys': f'{phys_loss.item():.5f}',
            })

        avg_tr_mse   = ep_mse   / len(tr_loader)
        avg_tr_phys  = ep_phys  / len(tr_loader)
        avg_tr_total = ep_total / len(tr_loader)
        history['train_mse'].append(avg_tr_mse)
        history['train_phys'].append(avg_tr_phys)
        history['train_loss'].append(avg_tr_total)

        # ── Validation ──
        model.eval()
        vl_mse = vl_phys = vl_total = 0.0
        val_preds, val_trues = [], []

        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)

                mse_loss  = mse_criterion(pred, yb)
                x_mid     = xb[:, WIN // 2, 0]
                phys_loss = phys_criterion(x_mid, pred)
                loss      = mse_loss + lambda_phys * phys_loss

                vl_mse   += mse_loss.item()
                vl_phys  += phys_loss.item()
                vl_total += loss.item()
                val_preds.append(pred.cpu().numpy())
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

        per_app_metrics = compute_per_appliance_metrics(
            y_true_all, y_pred_all, y_scalers)
        history['val_metrics'].append(per_app_metrics)

        avg_f1  = np.mean([per_app_metrics[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([per_app_metrics[a]['mae'] for a in APPLIANCES])

        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS}  "
            f"train={avg_tr_total:.5f} (mse={avg_tr_mse:.5f} phys={avg_tr_phys:.5f})  "
            f"val={avg_va_total:.5f} (mse={avg_va_mse:.5f} phys={avg_va_phys:.5f})  "
            f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
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

    # ── Test evaluation ──
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_preds, test_trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            test_preds.append(model(xb.to(device)).cpu().numpy())
            test_trues.append(yb.cpu().numpy())

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

    # ── Plots ──
    _plot_training(history, test_metrics, save_dir)

    # ── Save JSON ──
    config = {
        'dataset':     'AMPds_enriched',
        'model':       'PhysicsInformedLTCLiquidNetworkModel',
        'description': 'LTC LNN (input-dependent tau) + per-appliance heads + L_phys, input=[P,Q]',
        'loss':        f'MSE + {lambda_phys} * PhysicsConsistency(epsilon={epsilon_w}W) [stage2 only]',
        'input_size':  INPUT_SIZE,
        'window_size': WIN,
        'model_params': {
            'input_size': INPUT_SIZE, 'hidden_size': hidden_size,
            'n_appliances': len(APPLIANCES), 'dt': dt,
        },
        'train_params': {
            'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
            'lambda_phys': lambda_phys, 'epsilon_w': epsilon_w,
            'warmup_epochs': WARMUP_EPOCHS,
        },
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
    }
    with open(os.path.join(save_dir, 'pinn_ltc_lnn_ampds_enriched_results.json'),
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
    plt.title('Total Loss (MSE + lambda*Phys)')
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
    plt.savefig(os.path.join(save_dir, 'pinn_ltc_lnn_ampds_enriched_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('PINN-LTC AMPds enriched (P+Q) -- Per-Appliance Val Metrics',
                 fontsize=13)

    for row, app in enumerate(APPLIANCES):
        f1_series  = [m[app]['f1']  for m in history['val_metrics']]
        mae_series = [m[app]['mae'] for m in history['val_metrics']]

        ax_f1  = axes[row][0]
        ax_mae = axes[row][1]

        ax_f1.plot(epochs_x, f1_series, color='blue', linewidth=1.5)
        ax_f1.axhline(test_metrics[app]['f1'], color='green',
                      linestyle='--', label='Test F1')
        ax_f1.set_title(f'{app} -- F1')
        ax_f1.set_xlabel('Epoch'); ax_f1.set_ylabel('F1')
        ax_f1.legend(); ax_f1.grid(True, alpha=0.3)

        ax_mae.plot(epochs_x, mae_series, color='red', linewidth=1.5)
        ax_mae.axhline(test_metrics[app]['mae'], color='green',
                       linestyle='--', label='Test MAE')
        ax_mae.set_title(f'{app} -- MAE (W)')
        ax_mae.set_xlabel('Epoch'); ax_mae.set_ylabel('MAE (W)')
        ax_mae.legend(); ax_mae.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_ltc_lnn_ampds_enriched_per_appliance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for split in ('train', 'val', 'test'):
        fp = os.path.join(DATA_DIR, f'{split}.pkl')
        if not os.path.exists(fp):
            print(f"Error: {fp} not found! Run prepare_ampds_enriched.py first.")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir  = f"models/pinn_ltc_lnn_ampds_enriched_{timestamp}"

    data_dict = load_data()

    test_metrics, history = train_pinn_model(
        data_dict,
        save_dir    = save_dir,
        hidden_size = 64,
        dt          = 0.1,
        lambda_phys = LAMBDA_PHYS,
        epsilon_w   = EPSILON_W,
    )

    print(f"\nResults saved to {save_dir}")
