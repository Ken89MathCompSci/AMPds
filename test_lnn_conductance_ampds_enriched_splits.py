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


# ── Conductance-based LTC ──────────────────────────────────────────────────────
#
#   tau(x) * dh/dt = -h + sigma(W_A*x + U_A*h + b_A) * (E_rev - h)
#
#   Euler step:
#       tau   = softplus(tau_base + W_tau*x)      input-dependent time constant
#       A     = sigma(W_A*x + U_A*h + b_A)        conductance gate in (0,1)
#       h    <- h + dt/tau * (-h + A*(E_rev-h))
#
# Improvement over basic LTC:
#   * E_rev acts as self-regulating attractor -- no hard clamp needed
#   * tau(x) shrinks on spikes, expands on flat baseline
#   * conductance gate suppresses irrelevant timesteps

class ConductanceLTCLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dt=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt          = dt
        self.W_A         = nn.Linear(input_size,  hidden_size)
        self.U_A         = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_tau       = nn.Linear(input_size,  hidden_size, bias=False)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.E_rev       = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h):
        tau  = F.softplus(self.tau_base + self.W_tau(x))
        A    = torch.sigmoid(self.W_A(x) + self.U_A(h))
        dhdt = (-h + A * (self.E_rev - h)) / tau
        return h + self.dt * dhdt


class ConductanceLTCModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=1, dt=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell        = ConductanceLTCLayer(input_size, hidden_size, dt)
        self.fc          = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t, :], h)
        return self.fc(h)


# ── Constants ──────────────────────────────────────────────────────────────────

INPUT_SIZE = 2       # main (W)  +  main_Q (VAR)
DATA_DIR   = 'data/AMPds_enriched_data'

APPLIANCES = ['dish washer', 'fridge', 'oven', 'washer dryer']


# ── Dataset ────────────────────────────────────────────────────────────────────

class AMPdsDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_ampds_enriched_splits():
    print("Loading AMPds enriched data (P + Q)...")

    splits = {}
    for split in ('train', 'val', 'test'):
        with open(os.path.join(DATA_DIR, f'{split}.pkl'), 'rb') as f:
            splits[split] = pickle.load(f)[0]

    for name, df in splits.items():
        print(f"  {name}: {df.shape}  {df.index.min()} -> {df.index.max()}")
    print(f"  Columns: {list(splits['train'].columns)}")

    return splits


def create_sequences(data, window_size=100):
    # Two-channel input: active power (P) and reactive power (Q)
    mains = np.stack(
        [data['main'].values, data['main_Q'].values], axis=-1
    ).astype(np.float32)                                    # (T, 2)
    X = []
    stride = 5
    for i in range(0, len(mains) - window_size + 1, stride):
        X.append(mains[i:i + window_size])                  # (WIN, 2)
    return np.array(X, dtype=np.float32)                    # (N, WIN, 2)


def _scale_X(X_tr, X_va, X_te, window_size):
    """Fit MinMaxScaler per channel on train, apply to all splits."""
    n_tr, n_va, n_te = X_tr.shape[0], X_va.shape[0], X_te.shape[0]
    X_tr_n = X_tr.copy()
    X_va_n = X_va.copy()
    X_te_n = X_te.copy()
    for ch in range(INPUT_SIZE):
        sc = MinMaxScaler()
        X_tr_n[:, :, ch] = sc.fit_transform(
            X_tr[:, :, ch].reshape(-1, 1)).reshape(n_tr, window_size)
        X_va_n[:, :, ch] = sc.transform(
            X_va[:, :, ch].reshape(-1, 1)).reshape(n_va, window_size)
        X_te_n[:, :, ch] = sc.transform(
            X_te[:, :, ch].reshape(-1, 1)).reshape(n_te, window_size)
    return X_tr_n, X_va_n, X_te_n


def get_threshold_for_appliance(appliance_name):
    return 10.0


def train_on_appliance(data_dict, appliance_name, window_size=100,
                       hidden_size=64, dt=0.1,
                       epochs=80, lr=0.001, patience=20,
                       save_dir='models/conductance_lnn_ampds_enriched_specific'):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_data = data_dict['train']
    val_data   = data_dict['val']
    test_data  = data_dict['test']

    print(f"Creating sequences for {appliance_name}...")
    X_train = create_sequences(train_data, window_size)
    X_val   = create_sequences(val_data,   window_size)
    X_test  = create_sequences(test_data,  window_size)

    X_train, X_val, X_test = _scale_X(X_train, X_val, X_test, window_size)

    stride = 5
    y_train = train_data[appliance_name].iloc[::stride].values.reshape(-1, 1)[:len(X_train)]
    y_val   = val_data[appliance_name].iloc[::stride].values.reshape(-1, 1)[:len(X_val)]
    y_test  = test_data[appliance_name].iloc[::stride].values.reshape(-1, 1)[:len(X_test)]

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_val   = y_scaler.transform(y_val)
    y_test  = y_scaler.transform(y_test)

    print(f"Training sequences:   {X_train.shape} -> {y_train.shape}")
    print(f"Validation sequences: {X_val.shape} -> {y_val.shape}")
    print(f"Test sequences:       {X_test.shape} -> {y_test.shape}")

    train_loader = torch.utils.data.DataLoader(
        AMPdsDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        AMPdsDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        AMPdsDataset(X_test, y_test), batch_size=32, shuffle=False)

    model = ConductanceLTCModel(
        input_size=INPUT_SIZE,
        hidden_size=hidden_size,
        output_size=1,
        dt=dt
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val_loss = float('inf')
    best_state    = None
    counter = 0

    print(f"Starting ConductanceLTCModel training for {appliance_name}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        all_targets, all_outputs = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        threshold = get_threshold_for_appliance(appliance_name)
        raw_tgts = y_scaler.inverse_transform(
            np.concatenate(all_targets).reshape(-1, 1)).flatten()
        raw_outs = y_scaler.inverse_transform(
            np.concatenate(all_outputs).reshape(-1, 1)).flatten()
        metrics = calculate_nilm_metrics(raw_tgts, raw_outs, threshold=threshold)
        history['val_metrics'].append(metrics)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}, Val MAE: {metrics['mae']:.2f}, "
              f"Val SAE: {metrics['sae']:.2f}, Val F1: {metrics['f1']:.4f}, "
              f"Val Precision: {metrics['precision']:.4f}, Val Recall: {metrics['recall']:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            counter = 0
            best_model_path = os.path.join(
                save_dir, f"conductance_lnn_ampds_enriched_{appliance_name.replace(' ', '_')}_best.pth")
            save_model(model,
                       {'input_size': INPUT_SIZE, 'output_size': 1,
                        'hidden_size': hidden_size, 'dt': dt},
                       {'lr': lr, 'epochs': epochs, 'patience': patience,
                        'window_size': window_size, 'appliance': appliance_name},
                       metrics, best_model_path)
            print(f"Model saved to {best_model_path}")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping triggered")
                break

    print("Training completed!")

    if best_state is not None:
        model.load_state_dict(best_state)

    print("Evaluating on test set...")
    model.eval()
    all_test_targets, all_test_outputs = [], []
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
            all_test_targets.append(targets.cpu().numpy())
            all_test_outputs.append(outputs.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    threshold = get_threshold_for_appliance(appliance_name)
    all_test_targets = y_scaler.inverse_transform(
        np.concatenate(all_test_targets).reshape(-1, 1)).flatten()
    all_test_outputs = y_scaler.inverse_transform(
        np.concatenate(all_test_outputs).reshape(-1, 1)).flatten()
    test_metrics = calculate_nilm_metrics(all_test_targets, all_test_outputs, threshold=threshold)

    val_mae_series       = [m['mae']       for m in history['val_metrics']]
    val_sae_series       = [m['sae']       for m in history['val_metrics']]
    val_f1_series        = [m['f1']        for m in history['val_metrics']]
    val_precision_series = [m['precision'] for m in history['val_metrics']]
    val_recall_series    = [m['recall']    for m in history['val_metrics']]

    aggregates = {
        'train_loss_mean':      float(np.mean(history['train_loss'])),
        'train_loss_var':       float(np.var(history['train_loss'])),
        'val_loss_mean':        float(np.mean(history['val_loss'])),
        'val_loss_var':         float(np.var(history['val_loss'])),
        'val_mae_mean':         float(np.mean(val_mae_series)),
        'val_mae_var':          float(np.var(val_mae_series)),
        'val_sae_mean':         float(np.mean(val_sae_series)),
        'val_sae_var':          float(np.var(val_sae_series)),
        'val_f1_mean':          float(np.mean(val_f1_series)),
        'val_f1_var':           float(np.var(val_f1_series)),
        'val_precision_mean':   float(np.mean(val_precision_series)),
        'val_precision_var':    float(np.var(val_precision_series)),
        'val_recall_mean':      float(np.mean(val_recall_series)),
        'val_recall_var':       float(np.var(val_recall_series)),
        'test_mae':             float(test_metrics['mae']),
        'test_sae':             float(test_metrics['sae']),
        'test_f1':              float(test_metrics['f1']),
        'test_precision':       float(test_metrics['precision']),
        'test_recall':          float(test_metrics['recall']),
        'test_loss':            float(avg_test_loss)
    }

    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Test Metrics: {test_metrics}")

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'],   label='Val Loss',   color='red')
    plt.title(f'Loss - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(val_mae_series, label='Val MAE', color='red')
    plt.axhline(test_metrics['mae'], label='Test MAE', color='green', linestyle='--')
    plt.title(f'MAE - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MAE (W)')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(val_sae_series, label='Val SAE', color='red')
    plt.axhline(test_metrics['sae'], label='Test SAE', color='green', linestyle='--')
    plt.title(f'SAE - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('SAE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(val_f1_series,        label='Val F1',        color='red')
    plt.plot(val_precision_series, label='Val Precision', color='blue')
    plt.plot(val_recall_series,    label='Val Recall',    color='orange')
    plt.axhline(test_metrics['f1'],        color='red',    linestyle='--', alpha=0.5)
    plt.axhline(test_metrics['precision'], color='blue',   linestyle='--', alpha=0.5)
    plt.axhline(test_metrics['recall'],    color='orange', linestyle='--', alpha=0.5)
    plt.title(f'F1 / Precision / Recall - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('Score')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,
        f"conductance_lnn_ampds_enriched_{appliance_name.replace(' ', '_')}_metrics.png"),
        dpi=150, bbox_inches='tight')
    plt.close()

    config = {
        'appliance':  appliance_name,
        'dataset':    'AMPds_enriched',
        'model':      'ConductanceLTCModel',
        'ode':        'tau(x)*dh/dt = -h + sigma(W_A*x + U_A*h)*(E_rev - h)',
        'input_size': INPUT_SIZE,
        'norm':       'per-channel MinMaxScaler',
        'loss':       'MSE',
        'window_size': window_size,
        'model_params': {
            'input_size': INPUT_SIZE, 'output_size': 1,
            'hidden_size': hidden_size, 'dt': dt
        },
        'train_params': {'lr': lr, 'epochs': epochs, 'patience': patience},
        'final_metrics': {
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'aggregates':   aggregates
        }
    }
    with open(os.path.join(save_dir,
            f'conductance_lnn_ampds_enriched_{appliance_name.replace(" ", "_")}_history.json'),
            'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics


def test_on_all_appliances(window_size=100, hidden_size=64, dt=0.1,
                           epochs=80, lr=0.001, patience=20):
    data_dict = load_ampds_enriched_splits()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f"models/conductance_lnn_ampds_enriched_specific_test_{timestamp}"

    all_results = {}

    for appliance_name in APPLIANCES:
        print(f"\n{'='*60}")
        print(f"Testing ConductanceLTCModel (P+Q) on {appliance_name}")
        print(f"{'='*60}\n")

        appliance_dir = os.path.join(base_save_dir, appliance_name.replace(' ', '_'))
        os.makedirs(appliance_dir, exist_ok=True)

        try:
            model, history, test_metrics = train_on_appliance(
                data_dict,
                appliance_name=appliance_name,
                window_size=window_size,
                hidden_size=hidden_size,
                dt=dt,
                epochs=epochs,
                lr=lr,
                patience=patience,
                save_dir=appliance_dir
            )
            if model is not None:
                all_results[appliance_name] = {
                    'model_path': os.path.join(
                        appliance_dir,
                        f"conductance_lnn_ampds_enriched_{appliance_name.replace(' ', '_')}_best.pth"),
                    'final_metrics': {k: float(v) for k, v in test_metrics.items()}
                }
        except Exception as e:
            print(f"Error on {appliance_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    summary = {
        'timestamp':   timestamp,
        'dataset':     'AMPds_enriched',
        'model':       'ConductanceLTCModel',
        'ode':         'tau(x)*dh/dt = -h + sigma(W_A*x + U_A*h)*(E_rev - h)',
        'input_size':  INPUT_SIZE,
        'window_size': window_size,
        'model_params': {'hidden_size': hidden_size, 'dt': dt},
        'train_params': {'epochs': epochs, 'lr': lr, 'patience': patience},
        'results':     all_results
    }

    os.makedirs(base_save_dir, exist_ok=True)
    with open(os.path.join(base_save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nConductanceLTCModel AMPds enriched testing completed. Results saved to {base_save_dir}\n")
    print(f"{'Appliance':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} {'MAE':>8} {'SAE':>8}")
    print("-" * 65)
    for app in APPLIANCES:
        if app in all_results:
            m = all_results[app]['final_metrics']
            print(f"{app:<15} {m['f1']:>8.4f} {m['precision']:>10.4f} "
                  f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")

    return all_results


if __name__ == "__main__":
    print("Testing ConductanceLTCModel on AMPds enriched dataset (P + Q)...")

    for split in ('train', 'val', 'test'):
        fp = os.path.join(DATA_DIR, f'{split}.pkl')
        if not os.path.exists(fp):
            print(f"Error: {fp} not found! Run prepare_ampds_enriched.py first.")
            sys.exit(1)

    results = test_on_all_appliances(
        window_size=100,
        hidden_size=64,
        dt=0.1,
        epochs=80,
        lr=0.001,
        patience=20
    )

    print(f"\nSummary of ConductanceLTCModel testing on AMPds enriched dataset:")
    print(f"Total appliances tested: {len(results)}")
    for appliance, result in results.items():
        print(f"  {appliance}:")
        print(f"    Test MAE:       {result['final_metrics']['mae']:.4f}")
        print(f"    Test SAE:       {result['final_metrics']['sae']:.4f}")
        print(f"    Test F1:        {result['final_metrics']['f1']:.4f}")
        print(f"    Test Precision: {result['final_metrics']['precision']:.4f}")
        print(f"    Test Recall:    {result['final_metrics']['recall']:.4f}")
