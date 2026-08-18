"""
Downstream evaluation of trained Temporal Fusion Nexus folds.

Reproduces the paper's downstream protocol: Nexus representations are taken at
observation times between --min-history-days and --max-days (at most
--max-samples-per-patient per patient), labelled by whether the event occurs
within H days, and fed to a small binary classifier. AUC is reported as
mean +- sd across the cross-validation folds.

Usage:
    python src/evaluate.py --project-root . \
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from models import MultiModal, TimeAwareAttentionEncoder
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
from utils import build_elapsed_times

# Reported AUCs for the full multi-modal model with med-gte-hybrid-de
# embeddings (data/results/final_res.json, "time_series_static_notes").
# These are identical to the "disentangled_model" entries in
# data/results/disent_vs_ent.json, i.e. the headline numbers are the model
# trained WITH the auxiliary losses -- the configuration the released code
# could not produce, since all three lambdas were zero.
PAPER_TARGETS = {
    'graft_loss': {30: 0.9620, 90: 0.9648, 180: 0.9622, 360: 0.9496},
    'graft_rejection': {30: 0.8439, 90: 0.8471, 180: 0.8488, 360: 0.8235},
    'mortality': {30: 0.8601, 90: 0.8587, 180: 0.8514, 360: 0.8387},
}

# The ablation trained without the auxiliary losses, for reference.
ENTANGLED_TARGETS = {
    'graft_loss': {30: 0.9665, 90: 0.9687, 180: 0.9615, 360: 0.9552},
    'graft_rejection': {30: 0.8484, 90: 0.8504, 180: 0.8467, 360: 0.8206},
    'mortality': {30: 0.8645, 90: 0.8622, 180: 0.8505, 360: 0.8408},
}

TASKS = {
    'graft_loss': {'label_key': 'graft_loss_label', 'rel_days_key': 'loss_rel_days'},
    'graft_rejection': {'label_key': 'rej_rel_days', 'rel_days_key': None},
    'mortality': {'label_key': 'death_label', 'rel_days_key': 'death_rel_days'},
}


def _as_scalar(v):
    if isinstance(v, torch.Tensor):
        return float(v.reshape(-1)[0].item()) if v.numel() >= 1 else float('nan')
    return float(v)


@torch.no_grad()
def extract_representations(loader, model, device, horizons, args):
    """
    Returns per-task, per-horizon dicts of representations, labels and patient ids.

    The encoder is causal (time-aware LSTM, causal temporal self-attention,
    causal note cross-attention), so the Nexus at step k depends only on steps
    <= k. Sequences are therefore encoded once per batch and sliced afterwards,
    which is equivalent to the per-patient truncation the notebooks used.
    """
    model.eval()
    out = {
        task: {H: {'X': [], 'y': [], 'pid': []} for H in horizons}
        for task in TASKS
    }
    counts = {task: {H: {} for H in horizons} for task in TASKS}

    for batch in loader:
        cat_static = batch['static_categorical_features'].to(device)
        num_static = batch['static_numerical_features'].to(device)
        ts_values = batch['ts_features'].to(device)
        timesteps = batch['timesteps'].to(device)
        step_mask = batch['mask'].to(device)
        value_mask = batch['value_mask'].to(device)
        notes_embeddings = batch['notes_embeddings'].to(device)
        notes_timesteps = batch['notes_timesteps'].to(device)
        notes_mask = batch['notes_mask'].to(device)

        elapsed = build_elapsed_times(timesteps, step_mask)

        _, Z, _, _, _ = model(
            x=ts_values,
            elapsed_times=elapsed,
            timesteps=timesteps,
            notes_embeddings=notes_embeddings,
            notes_timesteps=notes_timesteps,
            static_features=(cat_static, num_static),
            mask=step_mask,
            notes_mask=notes_mask,
            value_mask=value_mask,
            future_elapsed=None,  # skip the decoder; only the embedding is needed
        )
        Z = Z.cpu().numpy()

        seq_lens = batch['seq_len']
        pids = batch['patient_id']
        times = timesteps.cpu().numpy()

        for i in range(Z.shape[0]):
            pid = pids[i]
            slen = int(seq_lens[i])
            if slen < 2:
                continue
            time_arr = times[i]

            event_info = {}
            for task, spec in TASKS.items():
                if spec['rel_days_key'] is not None:
                    event_info[task] = (
                        _as_scalar(batch[spec['label_key']][i]),
                        _as_scalar(batch[spec['rel_days_key']][i]),
                    )
                else:
                    days = batch[spec['label_key']][i]
                    event_info[task] = (None, list(days) if days is not None else [])

            # Which eligible timesteps to keep. 'first' is the published rule and
            # systematically excludes late events; see docs/paper_alignment.md 3e.
            eligible = [k for k in range(slen - 1)
                        if args.min_history_days <= time_arr[k + 1] <= args.max_days]
            strategy = getattr(args, 'sampling', 'first')
            cap = args.max_samples_per_patient
            if strategy == 'all' or len(eligible) <= cap:
                selected = eligible
            elif strategy == 'uniform':
                selected = [eligible[j] for j in
                            np.linspace(0, len(eligible) - 1, cap).astype(int)]
            else:
                selected = eligible[:cap]

            for k in selected:
                cur_day = time_arr[k + 1]
                rep = Z[i, k, :]

                for task, spec in TASKS.items():
                    for H in horizons:
                        # `selected` already applies the per-patient cap, so no
                        # further counting here -- doing both would silently
                        # defeat the 'all' strategy.
                        seen = counts[task][H].setdefault(pid, 0)

                        if spec['rel_days_key'] is not None:
                            event_label, event_day = event_info[task]
                            label = int(event_label == 1 and 0 < (event_day - cur_day) <= H)
                        else:
                            _, day_list = event_info[task]
                            label = int(any(0 < (d - cur_day) <= H for d in day_list))

                        out[task][H]['X'].append(rep)
                        out[task][H]['y'].append(label)
                        out[task][H]['pid'].append(pid)
                        counts[task][H][pid] = seen + 1

    for task in TASKS:
        for H in horizons:
            out[task][H]['X'] = np.asarray(out[task][H]['X'])
            out[task][H]['y'] = np.asarray(out[task][H]['y'])
            out[task][H]['pid'] = np.asarray(out[task][H]['pid'])
    return out


def fit_and_score(train_split, test_split, args):
    X_tr, y_tr = train_split['X'], train_split['y']
    X_te, y_te = test_split['X'], test_split['y']

    if len(X_tr) == 0 or len(X_te) == 0:
        return None, {'reason': 'empty split'}
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None, {'reason': 'single class', 'n_pos_train': int(y_tr.sum()),
                      'n_pos_test': int(y_te.sum())}

    clf = LogisticRegression(
        class_weight={0: 1, 1: args.positive_weight}, max_iter=1000
    ).fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, probs)
    return float(auc), {
        'n_train': int(len(y_tr)), 'n_test': int(len(y_te)),
        'n_pos_train': int(y_tr.sum()), 'n_pos_test': int(y_te.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate TFN folds on the downstream tasks")
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--horizons', type=int, nargs='+', default=[30, 90, 180, 360])
    parser.add_argument('--min-history-days', type=int, default=90)
    parser.add_argument('--max-days', type=int, default=720)
    parser.add_argument('--max-samples-per-patient', type=int, default=5)
    parser.add_argument('--positive-weight', type=float, default=10.0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--require-notes', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    checkpoints = sorted(run_dir.glob('fold*.pt'))
    if not checkpoints:
        raise SystemExit(f"no fold checkpoints in {run_dir}")
    print(f"device={device} checkpoints={[c.name for c in checkpoints]}", flush=True)

    print("loading data...", flush=True)
    dfs = get_dfs(args.project_root)
    static_df = create_static_df(dfs)
    medication_df = create_medication_df(dfs)
    vitals_ca, vitals_lab = create_vitals_df(dfs)
    biopsy_df = dfs['biopsy']
    ts_data = create_ts_data(vitals_ca, vitals_lab, medication_df, merge_lab=True, merge_med=True, static_df=static_df)
    notes = create_notes_df(dfs, filename=args.embeddings)

    per_fold = []

    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        fold = ckpt['fold']
        print(f"\n=== fold {fold} ({ckpt_path.name}, best epoch {ckpt['best_epoch']}) ===", flush=True)

        split_ids = dict(ckpt['patient_ids'])
        # The classifier trains on the full 80% (encoder-train plus the
        # early-stopping slice) and is scored on the untouched 20% fold.
        clf_train_ids = np.concatenate([np.asarray(split_ids['train']), np.asarray(split_ids['val'])])

        splits = create_dataset_splits(
            static_df=static_df, ts_data=ts_data, notes_df=notes, biopsy_df=biopsy_df,
            patient_split_ids={
                'selected': split_ids['selected'],
                'train': clf_train_ids,
                'val': [],
                'test': split_ids['test'],
            },
            preprocessing_artifacts=ckpt['preprocessing_artifacts'],
            require_notes=args.require_notes,
        )

        encoder = TimeAwareAttentionEncoder(use_temporal_attention=True)
        model = MultiModal(
            encoder,
            categorical_cardinalities=splits['train'].categorical_cardinalities,
            use_static=ckpt['use_static'],
            use_notes=ckpt['use_notes'],
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])

        train_loader = DataLoader(splits['train'], batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(splits['test'], batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        train_reps = extract_representations(train_loader, model, device, args.horizons, args)
        test_reps = extract_representations(test_loader, model, device, args.horizons, args)

        fold_result = {'fold': fold, 'best_epoch': ckpt['best_epoch'], 'tasks': {}}
        for task in TASKS:
            fold_result['tasks'][task] = {}
            for H in args.horizons:
                auc, info = fit_and_score(train_reps[task][H], test_reps[task][H], args)
                fold_result['tasks'][task][str(H)] = {'auc': auc, **info}
                shown = f"{auc:.4f}" if auc is not None else f"n/a ({info.get('reason')})"
                print(f"  {task:16s} H={H:3d}  AUC={shown}", flush=True)
        per_fold.append(fold_result)

    # Aggregate across folds
    summary = {}
    print("\n================ summary (mean +- sd across folds) ================", flush=True)
    for task in TASKS:
        summary[task] = {}
        for H in args.horizons:
            aucs = [f['tasks'][task][str(H)]['auc'] for f in per_fold
                    if f['tasks'][task][str(H)]['auc'] is not None]
            if not aucs:
                summary[task][str(H)] = {'mean': None, 'std': None, 'n_folds': 0}
                continue
            mean, std = float(np.mean(aucs)), float(np.std(aucs))
            target = PAPER_TARGETS.get(task, {}).get(H)
            summary[task][str(H)] = {
                'mean': round(mean, 4), 'std': round(std, 4),
                'n_folds': len(aucs), 'paper': target,
                'paper_entangled': ENTANGLED_TARGETS.get(task, {}).get(H),
                'delta': round(mean - target, 4) if target is not None else None,
            }
            tgt = f"paper {target:.4f}  delta {mean - target:+.4f}" if target is not None else ""
            print(f"  {task:16s} H={H:3d}  {mean:.4f} +- {std:.4f}  (n={len(aucs)})  {tgt}", flush=True)

    out_path = run_dir / 'downstream_results.json'
    with open(out_path, 'w') as f:
        json.dump({'args': vars(args), 'summary': summary, 'folds': per_fold}, f, indent=2, default=str)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == '__main__':
    main()
