"""
Sensitivity of downstream AUC to the sampling-window parameters.

min_history_days, max_days and max_samples_per_patient define the prediction
TASK, not the model: they decide which timepoints are scored and therefore how
many positive events exist. The published numbers used 90 / 720 / 5, where the
function defaults were 90 / 180 / 5.

Reporting a sensitivity curve is safer than tuning these. Picking the window
that maximises AUC is the same failure mode as picking the best of 30 epochs,
one level up.

Efficiency: the encoder is run ONCE per checkpoint and the per-patient
embeddings are cached, then each window is applied by re-sampling the cache.
Re-encoding per window would dominate the runtime for no benefit.

Probe: logistic regression by default -- fast, and with no epoch selection it
carries none of the optimism documented in evaluate_paper_protocol.py. Use it
to find the SHAPE of the curve, then confirm a chosen point with the MLP.

Usage:
    python src/sweep_window.py --project-root . \
        --embeddings data/embeddings/emb_med_gte_hybrid_de.npy \
        --run-dir runs/paper_aligned --max-encoders 2
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate import PAPER_TARGETS, TASKS, _as_scalar
from models import MultiModal, TimeAwareAttentionEncoder
from preprocessing import (
    collate_fn, create_dataset_splits, create_medication_df, create_notes_df,
    create_static_df, create_ts_data, create_vitals_df, get_dfs,
)
from utils import build_elapsed_times


@torch.no_grad()
def encode_once(loader, model, device):
    """Cache per-patient embeddings and event times; window-independent."""
    model.eval()
    cache = []
    for batch in loader:
        cat = batch['static_categorical_features'].to(device)
        num = batch['static_numerical_features'].to(device)
        ts = batch['ts_features'].to(device)
        timesteps = batch['timesteps'].to(device)
        step_mask = batch['mask'].to(device)
        value_mask = batch['value_mask'].to(device)
        elapsed = build_elapsed_times(timesteps, step_mask)

        _, Z, _, _, _ = model(
            x=ts, elapsed_times=elapsed, timesteps=timesteps,
            notes_embeddings=batch['notes_embeddings'].to(device),
            notes_timesteps=batch['notes_timesteps'].to(device),
            static_features=(cat, num), mask=step_mask,
            notes_mask=batch['notes_mask'].to(device),
            value_mask=value_mask, future_elapsed=None,
        )
        Z = Z.cpu().numpy().astype(np.float32)
        # Raw observed values at each timestep, as a baseline feature set:
        # if these beat the Nexus, the embedding is losing signal rather
        # than adding any.
        raw = ts.cpu().numpy().astype(np.float32)
        times = timesteps.cpu().numpy()

        for i in range(Z.shape[0]):
            slen = int(batch['seq_len'][i])
            if slen < 2:
                continue
            events = {}
            for task, spec in TASKS.items():
                if spec['rel_days_key'] is not None:
                    events[task] = (_as_scalar(batch[spec['label_key']][i]),
                                    _as_scalar(batch[spec['rel_days_key']][i]))
                else:
                    days = batch[spec['label_key']][i]
                    events[task] = (None, list(days) if days is not None else [])
            cache.append({'pid': batch['patient_id'][i], 'Z': Z[i, :slen - 1],
                          'raw': raw[i, :slen - 1],
                          'days': times[i, :slen], 'events': events})
    return cache


def build_dataset(cache, task, H, min_history, max_days, max_samples, strategy='first', features='nexus'):
    """
    strategy controls WHICH eligible timesteps are kept per patient:
      'first'   -- the earliest max_samples (what the published protocol did).
                   Combined with a max_days cap this systematically excludes
                   late events: median graft loss is ~day 1420 and median death
                   ~day 2387, while the published window ends at day 720.
      'uniform' -- max_samples spread evenly across the eligible range, so late
                   follow-up is represented.
      'all'     -- every eligible timestep.
    """
    X, y, pids = [], [], []
    spec = TASKS[task]
    for rec in cache:
        eligible = [k for k in range(rec['Z'].shape[0])
                    if rec['days'][k + 1] >= min_history
                    and (max_days is None or rec['days'][k + 1] <= max_days)]
        if not eligible:
            continue
        if strategy == 'all' or len(eligible) <= max_samples:
            sel = eligible
        elif strategy == 'uniform':
            idx = np.linspace(0, len(eligible) - 1, max_samples).astype(int)
            sel = [eligible[i] for i in idx]
        else:
            sel = eligible[:max_samples]

        for k in sel:
            cur = rec['days'][k + 1]
            if spec['rel_days_key'] is not None:
                lbl, day = rec['events'][task]
                label = int(lbl == 1 and 0 < (day - cur) <= H)
            else:
                _, dl = rec['events'][task]
                label = int(any(0 < (d - cur) <= H for d in dl))
            X.append(rec['Z' if features == 'nexus' else 'raw'][k])
            y.append(label); pids.append(rec['pid'])
    return np.asarray(X), np.asarray(y), np.asarray(pids)


def cv_auc(X, y, pids, folds=5):
    """Patient-grouped CV with a linear probe. No epoch selection."""
    if len(X) == 0 or len(np.unique(y)) < 2:
        return None, 0
    uniq = np.unique(pids)
    pid_label = {p: int(y[pids == p].any()) for p in uniq}
    n_pos = sum(pid_label.values())
    if n_pos < folds or (len(uniq) - n_pos) < folds:
        return None, n_pos
    strat = np.array([pid_label[p] for p in pids])
    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    aucs = []
    for tr, va in skf.split(X, strat, groups=pids):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        npos = max(1, int(y[tr].sum())); nneg = int((y[tr] == 0).sum())
        clf = LogisticRegression(class_weight={0: 1, 1: nneg / npos}, max_iter=1000).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[va], clf.predict_proba(X[va])[:, 1]))
    if not aucs:
        return None, n_pos
    return (float(np.mean(aucs)), float(np.std(aucs))), n_pos


def main():
    p = argparse.ArgumentParser(description="Sampling-window sensitivity sweep")
    p.add_argument('--project-root', default='.')
    p.add_argument('--embeddings', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--max-encoders', type=int, default=2)
    p.add_argument('--horizons', type=int, nargs='+', default=[30, 90])
    p.add_argument('--min-history-grid', type=int, nargs='+', default=[30, 90, 180])
    p.add_argument('--max-days-grid', type=int, nargs='+', default=[180, 360, 720, 1440, 100000])
    p.add_argument('--max-samples-grid', type=int, nargs='+', default=[1, 5, 20])
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--sampling', nargs='+', default=['first', 'uniform'],
                   choices=['first', 'uniform', 'all'])
    p.add_argument('--features', choices=['nexus', 'raw'], default='nexus',
                   help="'raw' uses the observed time-series values as a baseline")
    p.add_argument('--out', default='window_sweep.json')
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    ckpts = sorted(run_dir.glob('fold*.pt'))[:args.max_encoders]
    if not ckpts:
        raise SystemExit(f"no checkpoints in {run_dir}")

    print("loading data...", flush=True)
    dfs = get_dfs(args.project_root)
    static_df = create_static_df(dfs)
    med = create_medication_df(dfs)
    vca, vlab = create_vitals_df(dfs)
    biopsy_df = dfs['biopsy']
    ts_data = create_ts_data(vca, vlab, med, merge_lab=True, merge_med=True, static_df=static_df)
    notes = create_notes_df(dfs, filename=args.embeddings)

    caches = []
    for cp in ckpts:
        ck = torch.load(cp, weights_only=False, map_location=device)
        sid = dict(ck['patient_ids'])
        dev_ids = np.concatenate([np.asarray(sid['train']), np.asarray(sid['val'])])
        splits = create_dataset_splits(
            static_df=static_df, ts_data=ts_data, notes_df=notes, biopsy_df=biopsy_df,
            patient_split_ids={'selected': sid['selected'], 'train': dev_ids,
                               'val': [], 'test': sid['test']},
            preprocessing_artifacts=ck['preprocessing_artifacts'], require_notes=False)
        model = MultiModal(
            TimeAwareAttentionEncoder(use_temporal_attention=True),
            categorical_cardinalities=splits['train'].categorical_cardinalities,
            use_static=ck['use_static'], use_notes=ck['use_notes']).to(device)
        model.load_state_dict(ck['model_state_dict'])
        loader = DataLoader(splits['train'], batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)
        print(f"  encoding {cp.name} ...", flush=True)
        caches.append(encode_once(loader, model, device))
        print(f"  cached {len(caches[-1])} patients", flush=True)

    results = []
    print(f"\n{'task':16s} {'H':>4s} {'minH':>5s} {'maxD':>7s} {'maxS':>5s} {'samp':>8s} "
          f"{'AUC':>7s} {'sd':>7s} {'posPt':>6s} {'nPos':>7s} {'paper':>7s}", flush=True)
    for task in TASKS:
        for H in args.horizons:
            for mh in args.min_history_grid:
                for md in args.max_days_grid:
                    for ms in args.max_samples_grid:
                      for samp in args.sampling:
                        per_enc, npos_last, nrows_pos = [], 0, 0
                        for cache in caches:
                            X, y, pids = build_dataset(cache, task, H, mh, md, ms, samp, args.features)
                            res, npos = cv_auc(X, y, pids)
                            npos_last = npos
                            nrows_pos = int(y.sum()) if len(y) else 0
                            if res:
                                per_enc.append(res[0])
                        if not per_enc:
                            continue
                        mean = float(np.mean(per_enc))
                        sd = float(np.std(per_enc))
                        tgt = PAPER_TARGETS.get(task, {}).get(H)
                        results.append({'task': task, 'H': H, 'min_history': mh,
                                        'max_days': md, 'max_samples': ms, 'sampling': samp,
                                        'auc': round(mean, 4), 'sd_across_encoders': round(sd, 4),
                                        'n_pos_patients': npos_last, 'n_pos_rows': nrows_pos,
                                        'paper': tgt})
                        print(f"{task:16s} {H:4d} {mh:5d} {md:7d} {ms:5d} {samp:>8s} "
                              f"{mean:7.4f} {sd:7.4f} {npos_last:6d} {nrows_pos:7d} "
                              f"{(tgt if tgt else 0):7.4f}", flush=True)

    with open(args.out, 'w') as f:
        json.dump({'args': vars(args), 'results': results}, f, indent=2)

    print("\n=== best window per task/horizon ===", flush=True)
    for task in TASKS:
        for H in args.horizons:
            sub = [r for r in results if r['task'] == task and r['H'] == H]
            if not sub:
                continue
            b = max(sub, key=lambda r: r['auc'])
            print(f"  {task:16s} H={H:3d}  best AUC={b['auc']:.4f} at "
                  f"minH={b['min_history']} maxD={b['max_days']} maxS={b['max_samples']} samp={b['sampling']}"
                  f"  (paper {b['paper']})", flush=True)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
