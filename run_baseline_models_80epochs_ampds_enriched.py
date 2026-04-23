"""
Baseline Model Comparison -- 80 Epochs (AMPds enriched: P + Q)
===============================================================
Same as run_baseline_models_80epochs_ampds.py but uses the enriched
dataset (data/AMPds_enriched_data/) which adds mains reactive power
(main_Q) as a second input channel alongside active power (main).

input_size = 2  : [main (W), main_Q (VAR)]

Splits (1-min resolution, AMPds2):
  Train : 2013-11-21  (November)
  Val   : 2013-12-31  (December)
  Test  : 2012-08-23  (August)

Run ONE model per session:
    !python run_baseline_models_80epochs_ampds_enriched.py --model gru
    !python run_baseline_models_80epochs_ampds_enriched.py --model lstm
    !python run_baseline_models_80epochs_ampds_enriched.py --model resnet
    !python run_baseline_models_80epochs_ampds_enriched.py --model tcn
    !python run_baseline_models_80epochs_ampds_enriched.py --model transformer

Run ALL models in one go:
    !python run_baseline_models_80epochs_ampds_enriched.py --model all

Generate comparison graphs from saved results:
    !python run_baseline_models_80epochs_ampds_enriched.py --plot

Each model saves its own JSON to results/baseline_80epochs_ampds_enriched/<model>.json
so results accumulate across separate Colab sessions.
"""

import sys
import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime
from tqdm import tqdm

sys.path.append('Source Code')

from models import (
    GRUModel,
    LSTMModel,
    ResNetModel,
    TCNModel,
    SimpleTransformerModel,
)

# ── Constants ─────────────────────────────────────────────────────────────────

EPOCHS     = 80
PATIENCE   = 20
LR         = 1e-3
BATCH      = 32
WIN        = 100
STRIDE     = 5
INPUT_SIZE = 2          # main (W) + main_Q (VAR)

DATA_DIR = 'data/AMPds_enriched_data'
SAVE_DIR = os.path.join('results', 'baseline_80epochs_ampds_enriched')

APPLIANCES  = ['dish washer', 'fridge', 'oven', 'washer dryer']
APP_LABELS  = ['Dish Washer', 'Fridge', 'Oven', 'Washer Dryer']

THRESHOLDS = {
    'dish washer':  10.0,
    'fridge':       10.0,
    'oven':         10.0,
    'washer dryer': 10.0,
}

MODEL_LABELS = {
    'gru':         'GRU',
    'lstm':        'LSTM',
    'resnet':      'ResNet',
    'tcn':         'TCN',
    'transformer': 'Transformer',
}

MODEL_COLORS = {
    'gru':         '#4C72B0',
    'lstm':        '#DD8452',
    'resnet':      '#55A868',
    'tcn':         '#C44E52',
    'transformer': '#8172B2',
}

# ── Data ──────────────────────────────────────────────────────────────────────

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


def create_sequences(df, appliance_name, window_size=WIN, stride=STRIDE):
    # Two-channel input: active power (P) and reactive power (Q)
    mains = np.stack([df['main'].values, df['main_Q'].values], axis=-1).astype(np.float32)
    targets = df[appliance_name].values.astype(np.float32)
    X, y = [], []
    for i in range(0, len(mains) - window_size, stride):
        X.append(mains[i:i + window_size])          # shape (window_size, 2)
        midpoint = i + window_size // 2
        y.append(targets[midpoint])
    return (
        np.array(X, dtype=np.float32),              # (N, window_size, 2)
        np.array(y, dtype=np.float32).reshape(-1, 1),
    )


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Metrics ───────────────────────────────────────────────────────────────────

def calculate_metrics(y_true, y_pred, threshold):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mae = float(np.mean(np.abs(y_true - y_pred)))

    N = 100
    num_periods = len(y_true) // N
    diff = sum(
        abs(np.sum(y_true[i*N:(i+1)*N]) - np.sum(y_pred[i*N:(i+1)*N]))
        for i in range(num_periods)
    )
    sae = float(diff / (N * num_periods)) if num_periods > 0 else 0.0

    t_bin = (y_true > threshold).astype(int)
    p_bin = (y_pred > threshold).astype(int)
    tp = int(np.sum((t_bin == 1) & (p_bin == 1)))
    fp = int(np.sum((t_bin == 0) & (p_bin == 1)))
    fn = int(np.sum((t_bin == 1) & (p_bin == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'mae': mae, 'sae': sae, 'f1': float(f1),
            'precision': float(precision), 'recall': float(recall)}


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(model_key):
    if model_key == 'gru':
        return GRUModel(input_size=INPUT_SIZE, hidden_size=128, num_layers=2,
                        output_size=1, bidirectional=True)
    if model_key == 'lstm':
        return LSTMModel(input_size=INPUT_SIZE, hidden_size=128, num_layers=2,
                         output_size=1, bidirectional=True)
    if model_key == 'resnet':
        return ResNetModel(input_size=INPUT_SIZE, output_size=1,
                           layers=[2, 2, 2], base_width=32)
    if model_key == 'tcn':
        return TCNModel(input_size=INPUT_SIZE, output_size=1,
                        num_channels=[32, 64, 128], kernel_size=3, dropout=0.2)
    if model_key == 'transformer':
        return SimpleTransformerModel(input_size=INPUT_SIZE, hidden_size=128, output_size=1,
                                      num_layers=3, num_heads=4, dropout=0.1)
    raise ValueError(f"Unknown model: {model_key}")


# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(model_key, appliance_name, splits, device, optimizer_key='adam'):
    thr = THRESHOLDS[appliance_name]

    X_tr, y_tr = create_sequences(splits['train'], appliance_name)
    X_va, y_va = create_sequences(splits['val'],   appliance_name)
    X_te, y_te = create_sequences(splits['test'],  appliance_name)

    # Normalise each channel independently using train statistics
    mu    = X_tr.mean(axis=(0, 1), keepdims=True)          # shape (1, 1, 2)
    sigma = X_tr.std(axis=(0, 1),  keepdims=True) + 1e-8
    X_tr = (X_tr - mu) / sigma
    X_va = (X_va - mu) / sigma
    X_te = (X_te - mu) / sigma

    tr_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_va, y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        SimpleDataset(X_te, y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    model = build_model(model_key).to(device)
    criterion = torch.nn.MSELoss()
    _optim_map = {
        'adam':    torch.optim.Adam,
        'adamw':   torch.optim.AdamW,
        'sgd':     lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
        'rmsprop': torch.optim.RMSprop,
    }
    optimizer = _optim_map[optimizer_key](model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    best_val  = float('inf')
    best_state = None
    no_improve = 0
    train_losses, val_losses = [], []
    val_mae_hist, val_sae_hist, val_f1_hist = [], [], []
    val_precision_hist, val_recall_hist = [], []

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        ep_loss = 0.0
        for xb, yb in tqdm(tr_loader, desc=f"[{MODEL_LABELS[model_key]} | {appliance_name}] Epoch {epoch+1}/{EPOCHS}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
        avg_tr = ep_loss / len(tr_loader)
        train_losses.append(avg_tr)

        # ── Validate ──
        model.eval()
        vl_loss = 0.0
        val_preds_ep, val_trues_ep = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vl_loss += criterion(out, yb).item()
                val_preds_ep.append(out.cpu().numpy())
                val_trues_ep.append(yb.cpu().numpy())
        avg_va = vl_loss / len(va_loader)
        val_losses.append(avg_va)
        scheduler.step(avg_va)

        ep_pred = np.concatenate(val_preds_ep)
        ep_true = np.concatenate(val_trues_ep)
        ep_m = calculate_metrics(ep_true, ep_pred, thr)
        val_mae_hist.append(ep_m['mae'])
        val_sae_hist.append(ep_m['sae'])
        val_f1_hist.append(ep_m['f1'])
        val_precision_hist.append(ep_m['precision'])
        val_recall_hist.append(ep_m['recall'])

        print(f"  [{MODEL_LABELS[model_key]} | {appliance_name}] "
              f"Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train={avg_tr:.5f}  val={avg_va:.5f}  "
              f"F1={ep_m['f1']:.4f}  MAE={ep_m['mae']:.2f}  SAE={ep_m['sae']:.4f}  "
              f"P={ep_m['precision']:.4f}  R={ep_m['recall']:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if avg_va < best_val:
            best_val = avg_va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # ── Test with best weights ──
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            trues.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    metrics = calculate_metrics(y_true, y_pred, thr)

    return metrics, train_losses, val_losses, val_mae_hist, val_sae_hist, val_f1_hist, val_precision_hist, val_recall_hist


# ── Plotting helpers ──────────────────────────────────────────────────────────

TAG = 'AMPds enriched (P+Q), 80 epochs'

def bar_chart(all_results, metric_key, metric_label, save_path):
    model_keys = list(MODEL_LABELS.keys())
    x = np.arange(len(APPLIANCES))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, mk in enumerate(model_keys):
        vals = [all_results.get(mk, {}).get(app, {}).get(metric_key, np.nan)
                for app in APPLIANCES]
        offset = (i - len(model_keys) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=MODEL_LABELS[mk],
                      color=MODEL_COLORS[mk], alpha=0.85)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005 * ax.get_ylim()[1],
                        f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    ax.set_xlabel('Appliance')
    ax.set_ylabel(metric_label)
    ax.set_title(f'{metric_label} per Appliance -- Baseline Models ({TAG})')
    ax.set_xticks(x)
    ax.set_xticklabels(APP_LABELS)
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def summary_chart(all_results, save_path):
    metrics = [('mae', 'MAE'), ('sae', 'SAE'), ('f1', 'F1')]
    model_keys = list(MODEL_LABELS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (mk_key, mk_label) in zip(axes, metrics):
        vals = []
        for model in model_keys:
            scores = [all_results.get(model, {}).get(app, {}).get(mk_key, np.nan)
                      for app in APPLIANCES]
            vals.append(np.nanmean(scores))
        colors = [MODEL_COLORS[m] for m in model_keys]
        bars = ax.bar(list(MODEL_LABELS.values()), vals, color=colors, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005 * max(v for v in vals if not np.isnan(v)),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_title(f'Average {mk_label}')
        ax.set_ylabel(mk_label)
        ax.tick_params(axis='x', rotation=15)
    fig.suptitle(f'Summary -- Baseline Models Averaged Over All Appliances ({TAG})', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def f1_heatmap(all_results, save_path):
    model_keys = list(MODEL_LABELS.keys())
    data = np.full((len(APPLIANCES), len(model_keys)), np.nan)
    for j, mk in enumerate(model_keys):
        for i, app in enumerate(APPLIANCES):
            data[i, j] = all_results.get(mk, {}).get(app, {}).get('f1', np.nan)
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(model_keys)))
    ax.set_xticklabels(list(MODEL_LABELS.values()), fontsize=9)
    ax.set_yticks(range(len(APPLIANCES)))
    ax.set_yticklabels(APP_LABELS, fontsize=9)
    for i in range(len(APPLIANCES)):
        for j in range(len(model_keys)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9,
                        color='black' if 0.3 < val < 0.75 else 'white')
    plt.colorbar(im, ax=ax, label='F1 Score')
    ax.set_title(f'F1 Score Heatmap -- Baseline Models ({TAG})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def training_curves(model_key, curves, save_path):
    n = len(curves)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (app, (tl, vl)) in zip(axes, curves.items()):
        ax.plot(tl, label='Train', color='steelblue')
        ax.plot(vl, label='Val',   color='tomato')
        ax.set_title(app.title())
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f'Training Curves -- {MODEL_LABELS[model_key]} ({TAG})', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def epoch_metric_curves(model_key, epoch_metrics, save_path):
    metric_info = [('mae', 'MAE (W)'), ('sae', 'SAE'), ('f1', 'F1'), ('precision', 'Precision'), ('recall', 'Recall')]
    n = len(APPLIANCES)
    fig, axes = plt.subplots(n, 5, figsize=(25, 4 * n))
    color = MODEL_COLORS[model_key]
    for row, app in enumerate(APPLIANCES):
        em = epoch_metrics.get(app, {})
        for col, (mk_key, mk_label) in enumerate(metric_info):
            ax = axes[row, col]
            vals = em.get(mk_key, [])
            if vals:
                ax.plot(range(1, len(vals) + 1), vals, color=color)
            ax.set_title(f'{app.title()} -- {mk_label}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(mk_label)
            ax.grid(True, alpha=0.3)
    fig.suptitle(f'Val Metrics per Epoch -- {MODEL_LABELS[model_key]} ({TAG})', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


def combined_epoch_metric_curves(all_epoch_metrics, save_path):
    metric_info = [('mae', 'MAE (W)'), ('sae', 'SAE'), ('f1', 'F1'), ('precision', 'Precision'), ('recall', 'Recall')]
    n = len(APPLIANCES)
    fig, axes = plt.subplots(n, 5, figsize=(25, 4 * n))
    for row, app in enumerate(APPLIANCES):
        for col, (mk_key, mk_label) in enumerate(metric_info):
            ax = axes[row, col]
            for model_key in MODEL_LABELS:
                em = all_epoch_metrics.get(model_key, {}).get(app, {})
                vals = em.get(mk_key, [])
                if vals:
                    ax.plot(range(1, len(vals) + 1), vals,
                            label=MODEL_LABELS[model_key],
                            color=MODEL_COLORS[model_key])
            ax.set_title(f'{app.title()} -- {mk_label}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(mk_label)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    fig.suptitle(f'Val Metrics per Epoch -- All Baseline Models ({TAG})', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved -> {save_path}")


# ── Console summary table ─────────────────────────────────────────────────────

def print_table(all_results):
    model_keys = list(MODEL_LABELS.keys())
    divider = '-' * 80

    for metric, label in [('f1', 'F1'), ('precision', 'Precision'), ('recall', 'Recall'), ('mae', 'MAE'), ('sae', 'SAE')]:
        print(f"\n{'='*80}")
        print(f"  {label} Comparison (AMPds enriched P+Q)")
        print(f"{'='*80}")
        header = f"  {'Model':<16}" + ''.join(f"{a.title():<16}" for a in APP_LABELS) + f"{'Avg':<10}"
        print(header)
        print(divider)
        for mk in model_keys:
            vals = [all_results.get(mk, {}).get(app, {}).get(metric, float('nan'))
                    for app in APPLIANCES]
            avg  = np.nanmean(vals)
            row  = f"  {MODEL_LABELS[mk]:<16}" + ''.join(f"{v:<16.4f}" for v in vals) + f"{avg:<10.4f}"
            print(row)


# ── Per-model run ─────────────────────────────────────────────────────────────

def run_one_model(mk, splits, device, optimizer_key='adam'):
    os.makedirs(SAVE_DIR, exist_ok=True)

    results = {}
    curves  = {}
    epoch_metrics = {}

    print(f"\n{'#'*70}")
    print(f"#  {MODEL_LABELS[mk]}  (input: P + Q)")
    print(f"{'#'*70}")

    for app in APPLIANCES:
        print(f"\n  >> {app}")
        try:
            m, tl, vl, mae_h, sae_h, f1_h, prec_h, rec_h = train_model(mk, app, splits, device, optimizer_key)
            results[app] = m
            curves[app]  = (tl, vl)
            epoch_metrics[app] = {'mae': mae_h, 'sae': sae_h, 'f1': f1_h,
                                  'precision': prec_h, 'recall': rec_h}
            print(f"  OK {MODEL_LABELS[mk]:12s} | {app:15s} | "
                  f"F1={m['f1']:.4f}  MAE={m['mae']:.2f}  SAE={m['sae']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}")
        except Exception as exc:
            import traceback
            print(f"  FAIL {MODEL_LABELS[mk]} / {app}: {exc}")
            traceback.print_exc()

    json_path = os.path.join(SAVE_DIR, f'{mk}.json')
    with open(json_path, 'w') as f:
        json.dump({'model': mk, 'label': MODEL_LABELS[mk],
                   'input_size': INPUT_SIZE,
                   'epochs': EPOCHS, 'results': results,
                   'epoch_metrics': epoch_metrics}, f, indent=2)
    print(f"\n  JSON saved -> {json_path}")

    if curves:
        training_curves(mk, curves,
                        os.path.join(SAVE_DIR, f'training_curves_{mk}.png'))

    if epoch_metrics:
        epoch_metric_curves(mk, epoch_metrics,
                            os.path.join(SAVE_DIR, f'epoch_metrics_{mk}.png'))


def load_all_results():
    all_results = {}
    all_epoch_metrics = {}
    for mk in MODEL_LABELS:
        path = os.path.join(SAVE_DIR, f'{mk}.json')
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            all_results[mk] = data['results']
            all_epoch_metrics[mk] = data.get('epoch_metrics', {})
            print(f"  Loaded {MODEL_LABELS[mk]} from {path}")
        else:
            print(f"  Missing: {path}  (run --model {mk} first)")
    return all_results, all_epoch_metrics


def generate_plots(all_results, all_epoch_metrics=None):
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("\nGenerating graphs...")
    bar_chart(all_results, 'mae', 'MAE (W)',
              os.path.join(SAVE_DIR, 'mae_per_appliance.png'))
    bar_chart(all_results, 'sae', 'SAE',
              os.path.join(SAVE_DIR, 'sae_per_appliance.png'))
    bar_chart(all_results, 'f1',  'F1 Score',
              os.path.join(SAVE_DIR, 'f1_per_appliance.png'))
    bar_chart(all_results, 'precision', 'Precision',
              os.path.join(SAVE_DIR, 'precision_per_appliance.png'))
    bar_chart(all_results, 'recall', 'Recall',
              os.path.join(SAVE_DIR, 'recall_per_appliance.png'))
    summary_chart(all_results, os.path.join(SAVE_DIR, 'summary_all_metrics.png'))
    f1_heatmap(all_results,    os.path.join(SAVE_DIR, 'f1_heatmap.png'))
    if all_epoch_metrics and any(all_epoch_metrics.values()):
        combined_epoch_metric_curves(
            all_epoch_metrics,
            os.path.join(SAVE_DIR, 'epoch_metrics_all_models.png'))
    print_table(all_results)

    json_path = os.path.join(SAVE_DIR, 'results_summary.json')
    with open(json_path, 'w') as f:
        json.dump({'results': all_results}, f, indent=2)
    print(f"\nCombined JSON -> {json_path}")
    print(f"All graphs in  -> {SAVE_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        choices=['gru', 'lstm', 'resnet', 'tcn', 'transformer', 'all'],
        default='all',
        help='Which model to train. Use "all" to run every model sequentially.'
    )
    parser.add_argument(
        '--optimizer',
        choices=['adam', 'adamw', 'sgd', 'rmsprop'],
        default='adam',
        help='Optimizer to use (default: adam).'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Skip training -- just load saved JSONs and regenerate graphs.'
    )
    args = parser.parse_args()

    if args.plot:
        print("Plot mode -- loading saved results...")
        all_results, all_epoch_metrics = load_all_results()
        if not any(all_results.values()):
            print("No results found. Run at least one model first.")
            sys.exit(1)
        generate_plots(all_results, all_epoch_metrics)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    for split in ('train', 'val', 'test'):
        p = os.path.join(DATA_DIR, f'{split}.pkl')
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run prepare_ampds_enriched.py first.")
            sys.exit(1)

    splits = load_data()

    models_to_run = list(MODEL_LABELS.keys()) if args.model == 'all' else [args.model]

    for mk in models_to_run:
        run_one_model(mk, splits, device, args.optimizer)

    all_done = all(
        os.path.exists(os.path.join(SAVE_DIR, f'{mk}.json'))
        for mk in MODEL_LABELS
    )
    if all_done:
        print("\nAll models complete -- generating combined graphs...")
        all_results, all_epoch_metrics = load_all_results()
        generate_plots(all_results, all_epoch_metrics)
    else:
        missing = [mk for mk in MODEL_LABELS
                   if not os.path.exists(os.path.join(SAVE_DIR, f'{mk}.json'))]
        print(f"\nStill missing: {missing}")
        print("Run those models then regenerate plots with:  --plot")


if __name__ == '__main__':
    main()
