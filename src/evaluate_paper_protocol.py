"""
Downstream evaluation using the protocol from origin/main's classification.ipynb,
which is what produced the paper's reported numbers.

Protocol (per trained encoder):
  1. dev = the encoder's training patients, test = its held-out fold.
  2. Encode both once; the encoder is FROZEN from here on.
  3. StratifiedGroupKFold(5) over dev, grouped by patient, stratified on the
     patient-level label. The reported mean +- sd is classifier-refit variance
     with a fixed encoder -- not encoder-retraining variance.
  4. Per fold, train SimpleMLP (Adam, lr 5e-3, wd 1e-4, batch 32, 30 epochs)
     with BCEWithLogitsLoss(pos_weight = n_neg/n_pos).

Two AUCs are reported per fold:

  auc_selected -- the value main reports: the model is checkpointed on the
      maximum validation AUC over the 30 epochs and that same maximum is
      reported. Selecting on the scored split biases this upward, and with only
      a few dozen positives per fold the bias is not small.

  auc_final -- the same run scored at the last epoch, with no epoch selection.
      The difference between the two is the selection optimism.

Also reports held-out AUC: trained on all of dev (epoch chosen on an inner
split that is never scored) and evaluated on the untouched test patients.

Usage:
    python src/evaluate_paper_protocol.py --project-root . \
        --embeddings data/embeddings/emb_med_gte_hybrid_de.npy \
        --run-dir runs/paper_aligned
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from evaluate import PAPER_TARGETS, TASKS, extract_representations
from models import MultiModal, SimpleMLP, TimeAwareAttentionEncoder
from preprocessing import (
    collate_fn,
    create_dataset_splits,
    create_medication_df,
    create_notes_df,
    create_static_df,
    create_ts_data,
    create_vitals_df,
    get_dfs,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_threshold(y_true, probs):
    y_true = np.asarray(y_true).astype(int)
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 19), probs]))
    best_thr, best_f1 = 0.5, -1.0
    for thr in candidates:
        f1 = f1_score(y_true, (probs >= thr).astype(int), zero_division=0)
        if f1 > best_f1 or (np.isclose(f1, best_f1) and abs(thr - 0.5) < abs(best_thr - 0.5)):
            best_f1, best_thr = f1, float(thr)
    return best_thr


def predict_proba(model, X_t):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X_t)).cpu().numpy().reshape(-1)


class WideMLP(nn.Module):
    """Higher-capacity probe. If the Nexus holds more signal than SimpleMLP
    extracts, this should recover it; if not, the representation is the limit."""

    def __init__(self, input_dim, hidden=(512, 128), dropout=0.3):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_probe(kind, input_dim):
    if kind == 'mlp':
        return SimpleMLP(input_dim=input_dim)
    if kind == 'wide':
        return WideMLP(input_dim, hidden=(512, 128), dropout=0.3)
    if kind == 'deep':
        return WideMLP(input_dim, hidden=(1024, 256, 64), dropout=0.3)
    raise ValueError(kind)


def fit_logreg(X_train, y_train, X_val, y_val):
    """Linear probe. No epoch selection, so there is no selection optimism:
    auc_selected and auc_final are identical by construction."""
    y_val_int = np.asarray(y_val).astype(int)
    if len(set(y_val_int)) < 2:
        return {'auc_selected': float('nan'), 'auc_final': float('nan'),
                'best_epoch': 0, 'n_pos_train': int(np.sum(y_train == 1)),
                'n_pos_val': int(y_val_int.sum()), 'state': None, 'threshold': 0.5}
    n_pos = max(1, int(np.sum(y_train == 1)))
    n_neg = int(np.sum(y_train == 0))
    clf = LogisticRegression(class_weight={0: 1, 1: n_neg / n_pos}, max_iter=1000).fit(X_train, y_train)
    probs = clf.predict_proba(X_val)[:, 1]
    auc = float(roc_auc_score(y_val_int, probs))
    return {'auc_selected': auc, 'auc_final': auc, 'best_epoch': 0,
            'n_pos_train': n_pos, 'n_pos_val': int(y_val_int.sum()),
            'state': None, 'threshold': select_threshold(y_val_int, probs)}


def train_mlp(X_train, y_train, X_val, y_val, epochs=30, batch_size=32, lr=5e-3, probe='mlp'):
    """
    Mirrors train_and_eval_mlp_fold from origin/main, but returns both the
    epoch-selected and the final-epoch validation AUC.

    Note: `use_upsampling=True` in the original does not resample -- it sets
    pos_weight on the loss. It is cost-sensitive weighting, not oversampling.
    """
    if probe == 'lr':
        return fit_logreg(X_train, y_train, X_val, y_val)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    model = build_probe(probe, X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    num_pos = max(1, int(np.sum(y_train == 1)))
    num_neg = int(np.sum(y_train == 0))
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    y_val_int = np.asarray(y_val).astype(int)
    two_classes = len(set(y_val_int)) > 1

    best_auc, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

        if two_classes:
            auc = roc_auc_score(y_val_int, predict_proba(model, X_val_t))
            if auc > best_auc:
                best_auc, best_epoch = auc, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    final_probs = predict_proba(model, X_val_t)
    auc_final = roc_auc_score(y_val_int, final_probs) if two_classes else float('nan')

    return {
        'auc_selected': best_auc if best_auc >= 0 else float('nan'),
        'auc_final': auc_final,
        'best_epoch': best_epoch,
        'n_pos_train': num_pos,
        'n_pos_val': int(y_val_int.sum()),
        'state': best_state,
        'threshold': select_threshold(y_val_int, final_probs),
    }


def cv_for_task(X, y, pids, args):
    """StratifiedGroupKFold over patients, as in origin/main."""
    uniq = np.unique(pids)
    pid_label = {p: int(y[pids == p].any()) for p in uniq}
    strat_y = np.array([pid_label[p] for p in pids])

    n_pos_patients = sum(pid_label.values())
    if n_pos_patients < args.folds or (len(uniq) - n_pos_patients) < args.folds:
        return None, {'reason': f'too few positive patients ({n_pos_patients})'}

    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=42)
    folds = []
    for train_idx, val_idx in skf.split(X, strat_y, groups=pids):
        if len(np.unique(y[val_idx])) < 2 or len(np.unique(y[train_idx])) < 2:
            continue
        r = train_mlp(X[train_idx], y[train_idx], X[val_idx], y[val_idx],
                      epochs=args.epochs, probe=args.probe)
        r.pop('state', None)
        folds.append(r)

    if not folds:
        return None, {'reason': 'no usable folds'}

    sel = [f['auc_selected'] for f in folds if not np.isnan(f['auc_selected'])]
    fin = [f['auc_final'] for f in folds if not np.isnan(f['auc_final'])]
    return {
        'auc_selected_mean': float(np.mean(sel)), 'auc_selected_std': float(np.std(sel)),
        'auc_final_mean': float(np.mean(fin)), 'auc_final_std': float(np.std(fin)),
        'n_folds': len(folds),
        'n_pos_patients': n_pos_patients,
        'folds': folds,
    }, {}


def heldout_for_task(Xd, yd, pd_, Xt, yt, args):
    """Train on all of dev with an inner split for epoch choice; score on test."""
    if len(np.unique(yt)) < 2 or len(np.unique(yd)) < 2:
        return None
    uniq = np.unique(pd_)
    pid_label = {p: int(yd[pd_ == p].any()) for p in uniq}
    if sum(pid_label.values()) < 2:
        return None
    try:
        tr_p, va_p = train_test_split(
            uniq, test_size=0.15, random_state=42,
            stratify=[pid_label[p] for p in uniq],
        )
    except ValueError:
        return None
    tr = np.isin(pd_, tr_p)
    va = np.isin(pd_, va_p)
    if len(np.unique(yd[va])) < 2 or len(np.unique(yd[tr])) < 2:
        return None

    r = train_mlp(Xd[tr], yd[tr], Xd[va], yd[va], epochs=args.epochs, probe=args.probe)
    if r['state'] is None:
        return {'auc': None, 'epoch': r['best_epoch'], 'n_pos_test': int(np.asarray(yt).sum())}
    model = build_probe(args.probe, Xd.shape[1]).to(device)
    model.load_state_dict(r['state'])
    probs = predict_proba(model, torch.tensor(Xt, dtype=torch.float32).to(device))
    return {'auc': float(roc_auc_score(np.asarray(yt).astype(int), probs)),
            'epoch': r['best_epoch'], 'n_pos_test': int(np.asarray(yt).sum())}


def main():
    parser = argparse.ArgumentParser(description="Evaluate TFN using origin/main's downstream protocol")
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--horizons', type=int, nargs='+', default=[30, 90, 180, 360])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--probe', choices=['mlp', 'wide', 'deep', 'lr'], default='mlp',
                        help="Downstream probe: mlp=SimpleMLP (main's), wide/deep=higher capacity, lr=linear")
    parser.add_argument('--min-history-days', type=int, default=90)
    parser.add_argument('--max-days', type=int, default=720)
    parser.add_argument('--max-samples-per-patient', type=int, default=5)
    parser.add_argument('--sampling', choices=['first', 'uniform', 'all'], default='first')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--require-notes', action='store_true', default=False)
    parser.add_argument('--encoders', type=int, nargs='*', default=None,
                        help="Which encoder folds to use (default: all checkpoints found)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoints = sorted(run_dir.glob('fold*.pt'))
    if args.encoders is not None:
        checkpoints = [c for c in checkpoints if int(c.stem.replace('fold', '')) in args.encoders]
    if not checkpoints:
        raise SystemExit(f"no checkpoints in {run_dir}")
    print(f"device={device} encoders={[c.name for c in checkpoints]}", flush=True)

    print("loading data...", flush=True)
    dfs = get_dfs(args.project_root)
    static_df = create_static_df(dfs)
    medication_df = create_medication_df(dfs)
    vitals_ca, vitals_lab = create_vitals_df(dfs)
    biopsy_df = dfs['biopsy']
    ts_data = create_ts_data(vitals_ca, vitals_lab, medication_df, merge_lab=True, merge_med=True, static_df=static_df)
    notes = create_notes_df(dfs, filename=args.embeddings)

    results = []
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        enc_fold = ckpt['fold']
        print(f"\n=== encoder fold {enc_fold} ({ckpt_path.name}) ===", flush=True)

        sid = dict(ckpt['patient_ids'])
        dev_ids = np.concatenate([np.asarray(sid['train']), np.asarray(sid['val'])])

        splits = create_dataset_splits(
            static_df=static_df, ts_data=ts_data, notes_df=notes, biopsy_df=biopsy_df,
            patient_split_ids={'selected': sid['selected'], 'train': dev_ids,
                               'val': [], 'test': sid['test']},
            preprocessing_artifacts=ckpt['preprocessing_artifacts'],
            require_notes=args.require_notes,
        )

        model = MultiModal(
            TimeAwareAttentionEncoder(use_temporal_attention=True),
            categorical_cardinalities=splits['train'].categorical_cardinalities,
            use_static=ckpt['use_static'], use_notes=ckpt['use_notes'],
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])

        dev_loader = DataLoader(splits['train'], batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(splits['test'], batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        dev_reps = extract_representations(dev_loader, model, device, args.horizons, args)
        test_reps = extract_representations(test_loader, model, device, args.horizons, args)

        entry = {'encoder_fold': enc_fold, 'tasks': {}}
        for task in TASKS:
            entry['tasks'][task] = {}
            for H in args.horizons:
                d, t = dev_reps[task][H], test_reps[task][H]
                cv, info = cv_for_task(d['X'], d['y'], d['pid'], args)
                ho = heldout_for_task(d['X'], d['y'], d['pid'], t['X'], t['y'], args)
                entry['tasks'][task][str(H)] = {'cv': cv or info, 'heldout': ho}

                target = PAPER_TARGETS.get(task, {}).get(H)
                if cv:
                    ho_s = f"{ho['auc']:.4f}" if ho else "n/a"
                    print(f"  {task:16s} H={H:3d}  CV(selected)={cv['auc_selected_mean']:.4f}+-{cv['auc_selected_std']:.4f}"
                          f"  CV(final)={cv['auc_final_mean']:.4f}+-{cv['auc_final_std']:.4f}"
                          f"  heldout={ho_s}  paper={target}", flush=True)
                else:
                    print(f"  {task:16s} H={H:3d}  n/a ({info.get('reason')})", flush=True)
        results.append(entry)

    # Aggregate across encoders
    print("\n============ across encoders ============", flush=True)
    summary = {}
    for task in TASKS:
        summary[task] = {}
        for H in args.horizons:
            sel, fin, ho = [], [], []
            for e in results:
                c = e['tasks'][task][str(H)]['cv']
                if isinstance(c, dict) and 'auc_selected_mean' in c:
                    sel.append(c['auc_selected_mean']); fin.append(c['auc_final_mean'])
                h = e['tasks'][task][str(H)]['heldout']
                if h: ho.append(h['auc'])
            target = PAPER_TARGETS.get(task, {}).get(H)
            summary[task][str(H)] = {
                'cv_selected': round(float(np.mean(sel)), 4) if sel else None,
                'cv_final': round(float(np.mean(fin)), 4) if fin else None,
                'heldout': round(float(np.mean(ho)), 4) if ho else None,
                'selection_optimism': round(float(np.mean(sel) - np.mean(fin)), 4) if sel and fin else None,
                'paper': target,
                'delta_vs_paper': round(float(np.mean(sel)) - target, 4) if sel and target else None,
            }
            s = summary[task][str(H)]
            print(f"  {task:16s} H={H:3d}  CV_sel={s['cv_selected']}  CV_final={s['cv_final']}"
                  f"  optimism={s['selection_optimism']}  heldout={s['heldout']}"
                  f"  paper={target}  delta={s['delta_vs_paper']}", flush=True)

    suffix = '' if args.probe == 'mlp' else f'_{args.probe}'
    if args.sampling != 'first':
        suffix += f'_{args.sampling}'
    out = run_dir / f'paper_protocol_results{suffix}.json'
    with open(out, 'w') as f:
        json.dump({'args': vars(args), 'summary': summary, 'encoders': results}, f, indent=2, default=str)
    print(f"\nwrote {out}", flush=True)


if __name__ == '__main__':
    main()
