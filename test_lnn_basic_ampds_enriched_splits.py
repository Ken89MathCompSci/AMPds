"""
Basic LNN (LiquidNetworkModel) -- AMPds enriched splits (P + Q)
===============================================================
Same as test_lnn_basic_ampds_specific_splits.py but uses the enriched
dataset (data/AMPds_enriched_data/) with two input channels:
  channel 0 : main   -- mains active power  (W)
  channel 1 : main_Q -- mains reactive power (VAR)

input_size = 2

Architecture: LiquidTimeLayer unrolled over WIN timesteps -> final hidden state -> FC

  LiquidTimeLayer : Euler ODE  dh/dt = -h/tau + tanh(Wx + Uh),  dt=0.1
  Hidden state    : clamped to [-10, 10] each step
  FC head         : Linear(hidden -> 1)

Loss schedule:
  Epochs 0 ... WARMUP_EPOCHS-1 : MSE only
  Epochs WARMUP_EPOCHS ...     : MSE + lambda_bce x weighted-BCE(sigma(yhat/thr), s*)

LR schedule:
  Epochs 0 ... WARMUP_EPOCHS-1 : constant LR (Adam default)
  Epochs WARMUP_EPOCHS ...     : CosineAnnealingLR  (smooth decay -> eta_min=1e-5)

Gradient clipping: max_norm=1.0 every step.

Model selection:
  Warmup phase -- best val_loss
  BCE phase    -- best val F1  (avoids always-ON/OFF collapse)

Splits (1-min resolution, AMPds2):
  Train : 2013-11-21  (November)
  Val   : 2013-12-31  (December)
  Test  : 2012-08-23  (August)

Usage:
    !python test_lnn_basic_ampds_enriched_splits.py --appliance fridge
    !python test_lnn_basic_ampds_enriched_splits.py --appliance oven
    !python test_lnn_basic_ampds_enriched_splits.py --appliance "dish washer"
    !python test_lnn_basic_ampds_enriched_splits.py --appliance "washer dryer"
    !python test_lnn_basic_ampds_enriched_splits.py --plot
"""

import sys
import os
import time
import argparse
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source Code'))
from models import LiquidNetworkModel
from utils import calculate_nilm_metrics

# ── Constants ──────────────────────────────────────────────────────────────────

EPOCHS        = 80
PATIENCE      = 20
LR            = 1e-3
BATCH         = 128
WIN           = 100
STRIDE        = 5
WARMUP_EPOCHS = 15
INPUT_SIZE    = 2       # main (W)  +  main_Q (VAR)

DATA_DIR = 'data/AMPds_enriched_data'
SAVE_DIR = os.path.join('results', 'lnn_basic_ampds_enriched')
COLOR    = '#FF7F0E'

APPLIANCES = ['dish washer', 'fridge', 'oven', 'washer dryer']
APP_LABELS = ['Dish Washer', 'Fridge', 'Oven', 'Washer Dryer']

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'oven':         10.0,
    'washer dryer': 10.0,
}

BCE_LAMBDA = {
    'dish washer':  0.3,
    'fridge':       0.3,
    'oven':         0.3,
    'washer dryer': 0.5,
}

BCE_ALPHA = {
    'dish washer':  1.5,
    'fridge':       1.5,
    'oven':         4.0,
    'washer dryer': 5.0,
}


# ── Data ───────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading AMPds enriched data (P + Q)...")
    splits = {}
    for split in ('train', 'val', 'test'):
        with open(os.path.join(DATA_DIR, f'{split}.pkl'), 'rb') as f:
            splits[split] = pickle.load(f)[0]
    print(f"  Train: {splits['train'].index.min()} -> {splits['train'].index.max()}")
    print(f"  Val  : {splits['val'].index.min()} -> {splits['val'].index.max()}")
    print(f"  Test : {splits['test'].index.min()} -> {splits['test'].index.max()}")
    print(f"  Columns: {list(splits['train'].columns)}")
    return splits


def create_sequences(data, appliance_name, window_size=WIN):
    # Two-channel input: active power (P) and reactive power (Q)
    mains = np.stack(
        [data['main'].values, data['main_Q'].values], axis=-1
    ).astype(np.float32)                                   # (T, 2)
    app = data[appliance_name].values.astype(np.float32)
    X, y = [], []
    for i in range(0, len(mains) - window_size, STRIDE):
        X.append(mains[i:i + window_size])                 # (WIN, 2)
        y.append(app[i + window_size // 2])
    return (
        np.array(X, dtype=np.float32),                     # (N, WIN, 2)
        np.array(y, dtype=np.float32).reshape(-1, 1),
    )


def _scale_X(X_tr, X_va, X_te):
    """Fit MinMaxScaler per channel on train, apply to all splits."""
    n_tr, n_va, n_te = X_tr.shape[0], X_va.shape[0], X_te.shape[0]
    scalers = []
    X_tr_n = X_tr.copy()
    X_va_n = X_va.copy()
    X_te_n = X_te.copy()
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


class NILMDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Fast per-epoch metrics ─────────────────────────────────────────────────────

def _fast_metrics(y_true, y_pred, threshold):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mae    = float(np.mean(np.abs(y_true - y_pred)))
    t_bin  = y_true > threshold
    p_bin  = y_pred > threshold
    tp = int(np.sum(t_bin & p_bin))
    fp = int(np.sum(~t_bin & p_bin))
    fn = int(np.sum(t_bin & ~p_bin))
    pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
    return {'mae': mae, 'f1': f1, 'precision': pr, 'recall': rc}


# ── Training ───────────────────────────────────────────────────────────────────

def train_appliance(appliance_name, splits, device, epochs, hidden_size):
    thr       = THRESHOLDS[appliance_name]
    bce_lam   = BCE_LAMBDA[appliance_name]
    bce_alpha = BCE_ALPHA[appliance_name]

    print(f"\n{'='*60}")
    print(f"  {appliance_name}  |  hidden={hidden_size}  input_size={INPUT_SIZE}")
    print(f"  lambda_bce={bce_lam}  alpha_bce={bce_alpha}  warmup={WARMUP_EPOCHS}")
    print(f"{'='*60}")

    X_tr, y_tr = create_sequences(splits['train'], appliance_name)
    X_va, y_va = create_sequences(splits['val'],   appliance_name)
    X_te, y_te = create_sequences(splits['test'],  appliance_name)

    X_tr_n, X_va_n, X_te_n, _ = _scale_X(X_tr, X_va, X_te)

    y_scaler = MinMaxScaler()
    y_tr_n = y_scaler.fit_transform(y_tr)
    y_va_n = y_scaler.transform(y_va)
    y_te_n = y_scaler.transform(y_te)

    thr_scaled = float((thr - y_scaler.data_min_[0]) / y_scaler.data_range_[0])

    print(f"  Train: {X_tr_n.shape}  Val: {X_va_n.shape}  Test: {X_te_n.shape}")

    tr_loader = torch.utils.data.DataLoader(
        NILMDataset(X_tr_n, y_tr_n), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        NILMDataset(X_va_n, y_va_n), batch_size=BATCH, shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        NILMDataset(X_te_n, y_te_n), batch_size=BATCH, shuffle=False, drop_last=False)

    model = LiquidNetworkModel(
        input_size=INPUT_SIZE, hidden_size=hidden_size, output_size=1, dt=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - WARMUP_EPOCHS), eta_min=1e-5)

    def _loss(out, yb, epoch):
        mse = F.mse_loss(out, yb)
        if epoch < WARMUP_EPOCHS:
            return mse
        prob  = torch.sigmoid(out / (thr_scaled + 1e-8))
        y_bin = (yb > thr_scaled).float()
        w     = torch.where(y_bin == 1,
                            torch.full_like(y_bin, bce_alpha),
                            torch.ones_like(y_bin))
        bce   = F.binary_cross_entropy(prob.clamp(1e-7, 1 - 1e-7), y_bin, weight=w)
        return mse + bce_lam * bce

    train_losses, val_losses = [], []
    val_mae_h, val_f1_h, val_p_h, val_r_h = [], [], [], []
    val_time_h = []

    best_val    = float('inf')
    best_val_f1 = -1.0
    best_state  = None
    no_improve  = 0

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        bar = tqdm(tr_loader, desc=f"  Epoch {epoch+1}/{epochs}", leave=False)
        for xb, yb in bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(xb)
            loss = _loss(out, yb, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
            bar.set_postfix({'loss': f'{loss.item():.5f}'})
        avg_tr = ep_loss / len(tr_loader)
        train_losses.append(avg_tr)

        # ── Validation ──
        vt0 = time.time()
        model.eval()
        vl_loss = 0.0
        preds_v, trues_v = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                out     = model(xb)
                vl_loss += _loss(out, yb, epoch).item()
                preds_v.append(out.cpu().numpy())
                trues_v.append(yb.cpu().numpy())
        avg_va = vl_loss / len(va_loader)
        val_losses.append(avg_va)

        if epoch >= WARMUP_EPOCHS:
            cosine_scheduler.step()

        raw_pred = y_scaler.inverse_transform(
            np.concatenate(preds_v).reshape(-1, 1)).flatten()
        raw_true = y_scaler.inverse_transform(
            np.concatenate(trues_v).reshape(-1, 1)).flatten()
        vm = _fast_metrics(raw_true, raw_pred, threshold=thr)
        val_mae_h.append(vm['mae'])
        val_f1_h.append(vm['f1'])
        val_p_h.append(vm['precision'])
        val_r_h.append(vm['recall'])
        val_time_h.append(time.time() - vt0)

        print(f"  Epoch {epoch+1:3d}/{epochs}  "
              f"train={avg_tr:.5f}  val={avg_va:.5f}  "
              f"F1={vm['f1']:.4f}  P={vm['precision']:.4f}  R={vm['recall']:.4f}  "
              f"MAE={vm['mae']:.1f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if epoch == WARMUP_EPOCHS:
            best_val    = float('inf')
            best_val_f1 = -1.0
            no_improve  = 0

        improved = (avg_va < best_val) if epoch < WARMUP_EPOCHS \
                   else (vm['f1'] > best_val_f1)

        if improved:
            best_val    = avg_va
            best_val_f1 = vm['f1']
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # ── Test ──
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    infer_t0 = time.time()
    preds_t, trues_t = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            preds_t.append(model(xb.to(device)).cpu().numpy())
            trues_t.append(yb.numpy())
    infer_s = time.time() - infer_t0

    raw_pred_te = y_scaler.inverse_transform(
        np.concatenate(preds_t).reshape(-1, 1)).flatten()
    raw_true_te = y_scaler.inverse_transform(
        np.concatenate(trues_t).reshape(-1, 1)).flatten()
    test_metrics = calculate_nilm_metrics(raw_true_te, raw_pred_te, threshold=thr)

    print(f"\n  Test  F1={test_metrics['f1']:.4f}  P={test_metrics['precision']:.4f}  "
          f"R={test_metrics['recall']:.4f}  MAE={test_metrics['mae']:.2f}  "
          f"SAE={test_metrics['sae']:.4f}")

    return {
        'metrics':      test_metrics,
        'train_losses': train_losses,
        'val_losses':   val_losses,
        'val_mae_h':    val_mae_h,
        'val_f1_h':     val_f1_h,
        'val_p_h':      val_p_h,
        'val_r_h':      val_r_h,
        'val_time_h':   val_time_h,
        'infer_s':      infer_s,
        'epochs_run':   len(train_losses),
        'num_params':   n_params,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_training_curves(results, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, app in zip(axes.flatten(), APPLIANCES):
        if app not in results:
            continue
        r  = results[app]
        ep = range(1, r['epochs_run'] + 1)
        ax.plot(ep, r['train_losses'], label='Train', color='steelblue', lw=1.5)
        ax.plot(ep, r['val_losses'],   label='Val',   color='tomato',    lw=1.5, ls='--')
        ax.axvline(WARMUP_EPOCHS, color='grey', ls=':', lw=1, label='BCE start')
        ax.set_title(app.title())
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle('Training Curves -- Basic LNN (AMPds enriched P+Q)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_epoch_metrics(results, save_dir):
    fig, axes = plt.subplots(len(APPLIANCES), 3,
                              figsize=(15, 4 * len(APPLIANCES)))
    for row, app in enumerate(APPLIANCES):
        if app not in results:
            continue
        r  = results[app]
        ep = range(1, r['epochs_run'] + 1)
        for col, (key, label) in enumerate(
                [('val_mae_h', 'MAE (W)'), ('val_f1_h', 'F1'),
                 ('val_p_h', 'Precision / Recall')]):
            ax = axes[row, col]
            ax.plot(ep, r[key], color=COLOR, lw=1.5, label=label)
            if label == 'Precision / Recall':
                ax.plot(ep, r['val_r_h'], color='orange', lw=1.5,
                        ls='--', label='Recall')
                ax.legend(fontsize=7)
            ax.axvline(WARMUP_EPOCHS, color='grey', ls=':', lw=1)
            ax.set_title(f'{app.title()} -- {label}')
            ax.set_xlabel('Epoch'); ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
    fig.suptitle('Val Metrics per Epoch -- Basic LNN (AMPds enriched P+Q)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'epoch_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_bar_chart(results, save_dir):
    metrics_cfg = [('mae', 'MAE (W)'), ('sae', 'SAE'), ('f1', 'F1'),
                   ('precision', 'Precision'), ('recall', 'Recall')]
    x   = np.arange(len(APPLIANCES))
    fig, axes = plt.subplots(1, len(metrics_cfg), figsize=(18, 5))
    for ax, (mk, ml) in zip(axes, metrics_cfg):
        vals = [results[app]['metrics'].get(mk, np.nan)
                if app in results else np.nan for app in APPLIANCES]
        bars = ax.bar(x, vals, color=COLOR, alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(APP_LABELS, rotation=12, ha='right')
        ax.set_ylabel(ml); ax.set_title(ml)
        ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
    fig.suptitle('Final Test Metrics -- Basic LNN (AMPds enriched P+Q)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'bar_chart.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def print_table(results):
    divider = '-' * 70
    for metric, label in [('f1', 'F1'), ('precision', 'Precision'),
                           ('recall', 'Recall'), ('mae', 'MAE'), ('sae', 'SAE')]:
        print(f"\n{'='*70}")
        print(f"  {label} -- Basic LNN (AMPds enriched P+Q)")
        print(f"{'='*70}")
        print(f"  {'Appliance':<20}{'Value':>12}  (epochs)")
        print(divider)
        vals = []
        for app in APPLIANCES:
            if app in results:
                v  = results[app]['metrics'].get(metric, float('nan'))
                ep = results[app]['epochs_run']
                print(f"  {app.title():<20}{v:>12.4f}  ({ep})")
                vals.append(v)
        print(divider)
        if vals:
            print(f"  {'Average':<20}{np.nanmean(vals):>12.4f}")


# ── JSON helpers ───────────────────────────────────────────────────────────────

def _save_json(app, r, hidden, epochs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{app.replace(" ", "_")}.json')
    with open(path, 'w') as f:
        json.dump({
            'appliance':     app,
            'dataset':       'AMPds_enriched',
            'architecture':  'LiquidNetworkModel',
            'input_size':    INPUT_SIZE,
            'hidden_size':   hidden,
            'epochs':        epochs,
            'epochs_run':    r['epochs_run'],
            'num_params':    r['num_params'],
            'bce_lambda':    BCE_LAMBDA[app],
            'bce_alpha':     BCE_ALPHA[app],
            'warmup_epochs': WARMUP_EPOCHS,
            'metrics':       {k: float(v) for k, v in r['metrics'].items()},
            'train_losses':  r['train_losses'],
            'val_losses':    r['val_losses'],
            'val_mae_h':     r['val_mae_h'],
            'val_f1_h':      r['val_f1_h'],
            'val_p_h':       r['val_p_h'],
            'val_r_h':       r['val_r_h'],
            'val_time_h':    r['val_time_h'],
            'infer_s':       r['infer_s'],
        }, f, indent=2)
    print(f'  JSON saved -> {path}')


def _load_all_jsons(save_dir):
    results = {}
    for app in APPLIANCES:
        path = os.path.join(save_dir, f'{app.replace(" ", "_")}.json')
        if os.path.exists(path):
            with open(path) as f:
                results[app] = json.load(f)
            print(f'  Loaded {app} <- {path}')
        else:
            print(f'  Missing: {path}  (run --appliance "{app}" first)')
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Basic LNN -- AMPds enriched (P+Q)')
    parser.add_argument('--epochs',    type=int, default=EPOCHS)
    parser.add_argument('--hidden',    type=int, default=256,
                        help='Hidden size (default 256)')
    parser.add_argument('--appliance', type=str, default=None,
                        choices=APPLIANCES,
                        help='Train a single appliance and save its JSON.')
    parser.add_argument('--plot',      action='store_true',
                        help='Skip training -- load saved JSONs and plot.')
    parser.add_argument('--save_dir',  type=str, default=SAVE_DIR)
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    if args.plot:
        print('Plot mode -- loading saved results...')
        results = _load_all_jsons(save_dir)
        if not results:
            print('No results found. Run at least one appliance first.')
            sys.exit(1)
        plot_training_curves(results, save_dir)
        plot_epoch_metrics(results, save_dir)
        plot_bar_chart(results, save_dir)
        print_table(results)
        return

    for split in ('train', 'val', 'test'):
        fp = os.path.join(DATA_DIR, f'{split}.pkl')
        if not os.path.exists(fp):
            print(f'ERROR: {fp} not found. Run prepare_ampds_enriched.py first.')
            sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  Hidden: {args.hidden}  |  Epochs: {args.epochs}  |  input_size: {INPUT_SIZE}')
    print('Architecture: LiquidTimeLayer (ODE unroll) -> final hidden state -> FC')

    splits      = load_data()
    apps_to_run = [args.appliance] if args.appliance else APPLIANCES

    total_t0 = time.time()
    all_results: dict = {}

    for app in apps_to_run:
        t0 = time.time()
        r  = train_appliance(app, splits, device, args.epochs, args.hidden)
        r['time_s'] = time.time() - t0
        m = r['metrics']
        print(f"\n  DONE {app:<15} | F1={m['f1']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  MAE={m['mae']:.2f}  SAE={m['sae']:.4f}  "
              f"({r['num_params']:,} params  {r['time_s']:.0f}s)")
        _save_json(app, r, args.hidden, args.epochs, save_dir)
        all_results[app] = r

    total_s = time.time() - total_t0
    print(f'\nTotal time: {total_s:.0f}s ({total_s/60:.1f} min)')

    all_done = all(
        os.path.exists(os.path.join(save_dir, f'{a.replace(" ", "_")}.json'))
        for a in APPLIANCES
    )
    if all_done:
        full = _load_all_jsons(save_dir)
        print('\nAll appliances complete -- generating plots...')
        plot_training_curves(full, save_dir)
        plot_epoch_metrics(full, save_dir)
        plot_bar_chart(full, save_dir)
        print_table(full)
    else:
        missing = [a for a in APPLIANCES
                   if not os.path.exists(
                       os.path.join(save_dir, f'{a.replace(" ", "_")}.json'))]
        print(f'\nStill missing: {missing}')
        print('Run those, then: python test_lnn_basic_ampds_enriched_splits.py --plot')

    print(f'\nAll outputs -> {save_dir}/')


if __name__ == '__main__':
    main()
