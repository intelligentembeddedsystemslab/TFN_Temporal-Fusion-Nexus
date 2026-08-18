"""
Disentanglement metrics on trained Temporal Fusion Nexus checkpoints.

Reports the STANDARD definitions of MIG and SAP alongside the lenient variants
currently in src/utils.py, so the two can be compared directly.

Why this matters
----------------
Standard SAP (Kumar et al. 2018) is a GAP: for each factor, the score of the
best single latent dimension minus the score of the second best. The gap is
what "disentangled" means -- one factor should be captured by one dimension.

`utils.compute_sap` instead reports the absolute R^2 of the best five
dimensions fitted jointly, and its own docstring says it is "more lenient ...
easier for models to achieve good scores". That measures how PREDICTABLE a
factor is, not how SEPARATED the representation is. A fully entangled
representation, where every dimension encodes everything, scores high on it.

Standard MIG (Chen et al. 2018) is likewise the normalised gap between the
top-1 and top-2 mutual information. `utils.compute_mig` uses overlapping groups
of three dimensions, which is also more permissive.

Per-factor scoring here:
  - numerical / time-series factors: R^2 of a single latent dimension, which
    for one predictor equals the squared Pearson correlation.
  - categorical factors: the correlation ratio eta^2, the standard
    continuous-vs-categorical analogue of R^2.

Usage:
    python src/disentanglement.py --project-root . \
        --embeddings data/embeddings/emb_med_gte_hybrid_de.npy \
        --runs runs/paper_aligned runs/recon_only
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mutual_info_score
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from models import MultiModal, TimeAwareAttentionEncoder
from preprocessing import (
    collate_fn, create_dataset_splits, create_medication_df, create_notes_df,
    create_static_df, create_ts_data, create_vitals_df, get_dfs,
)
from utils import (build_elapsed_times, compute_dci, compute_mig_lenient,
                   compute_sap_lenient, get_last_valid_step)
from utils import compute_mig as _std_mig_utils  # noqa: F401


# ----------------------------------------------------------------------------
# standard metrics
# ----------------------------------------------------------------------------

def _discretize(x, bins=20):
    edges = np.histogram_bin_edges(x, bins=bins)
    return np.digitize(x, edges[1:-1])


def _entropy(codes):
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def standard_mig(Z, F, is_cat, bins=20):
    """Normalised gap between the top-1 and top-2 mutual information."""
    Zd = np.stack([_discretize(Z[:, j], bins) for j in range(Z.shape[1])], axis=1)
    per_factor = []
    for k in range(F.shape[1]):
        fk = F[:, k].astype(int) if is_cat[k] else _discretize(F[:, k], bins)
        H = _entropy(fk)
        if H <= 1e-8:
            continue
        mi = np.array([mutual_info_score(Zd[:, j], fk) for j in range(Z.shape[1])])
        top = np.sort(mi)[::-1]
        per_factor.append((top[0] - top[1]) / H)
    if not per_factor:
        return float('nan'), []
    return float(np.mean(per_factor)), per_factor


def _eta_squared(z, codes):
    """Correlation ratio: between-group variance over total variance."""
    total_var = z.var()
    if total_var <= 1e-12:
        return 0.0
    grand = z.mean()
    num = 0.0
    for c in np.unique(codes):
        g = z[codes == c]
        if g.size == 0:
            continue
        num += g.size * (g.mean() - grand) ** 2
    return float(num / (z.size * total_var))


def standard_sap(Z, F, is_cat):
    """Gap between the best and second-best SINGLE latent dimension per factor."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    Zs = Zc.std(axis=0) + 1e-12
    per_factor = []
    for k in range(F.shape[1]):
        if is_cat[k]:
            codes = F[:, k].astype(int)
            scores = np.array([_eta_squared(Z[:, j], codes) for j in range(Z.shape[1])])
        else:
            f = F[:, k] - F[:, k].mean()
            fs = f.std() + 1e-12
            corr = (Zc * f[:, None]).mean(axis=0) / (Zs * fs)
            scores = corr ** 2  # R^2 of a single predictor
        top = np.sort(scores)[::-1]
        per_factor.append(float(top[0] - top[1]))
    if not per_factor:
        return float('nan'), []
    return float(np.mean(per_factor)), per_factor


# ----------------------------------------------------------------------------

@torch.no_grad()
def collect(loader, model, device):
    """Nexus embedding and generative factors at each patient's last valid step."""
    model.eval()
    Zs, cats, nums, tss = [], [], [], []
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
        Zs.append(get_last_valid_step(Z, step_mask).cpu().numpy())
        cats.append(cat.cpu().numpy())
        nums.append(num.cpu().numpy())
        tss.append(get_last_valid_step(ts, step_mask).cpu().numpy())

    Z = np.concatenate(Zs)
    F = np.concatenate([np.concatenate(cats, 0), np.concatenate(nums, 0), np.concatenate(tss, 0)], axis=1)
    n_cat = len(CONFIG['static_categorical_cols'])
    is_cat = [True] * n_cat + [False] * (F.shape[1] - n_cat)
    names = (CONFIG['static_categorical_cols'] + CONFIG['static_numerical_cols'] + CONFIG['ts_features'])
    return Z, F, is_cat, names


def main():
    p = argparse.ArgumentParser(description="Standard vs lenient disentanglement metrics")
    p.add_argument('--project-root', default='.')
    p.add_argument('--embeddings', required=True)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--bins', type=int, default=20)
    p.add_argument('--max-encoders', type=int, default=2)
    p.add_argument('--out', default='disentanglement_results.json')
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading data...", flush=True)
    dfs = get_dfs(args.project_root)
    static_df = create_static_df(dfs)
    med = create_medication_df(dfs)
    vca, vlab = create_vitals_df(dfs)
    biopsy_df = dfs['biopsy']
    ts_data = create_ts_data(vca, vlab, med, merge_lab=True, merge_med=True, static_df=static_df)
    notes = create_notes_df(dfs, filename=args.embeddings)

    results = {}
    for run in args.runs:
        run_dir = Path(run)
        ckpts = sorted(run_dir.glob('fold*.pt'))[:args.max_encoders]
        if not ckpts:
            print(f"!! no checkpoints in {run}", flush=True)
            continue
        results[run] = []
        for cp in ckpts:
            ck = torch.load(cp, weights_only=False, map_location=device)
            sid = dict(ck['patient_ids'])
            splits = create_dataset_splits(
                static_df=static_df, ts_data=ts_data, notes_df=notes, biopsy_df=biopsy_df,
                patient_split_ids={'selected': sid['selected'], 'train': sid['test'],
                                   'val': [], 'test': []},
                preprocessing_artifacts=ck['preprocessing_artifacts'], require_notes=False,
            )
            model = MultiModal(
                TimeAwareAttentionEncoder(use_temporal_attention=True),
                categorical_cardinalities=splits['train'].categorical_cardinalities,
                use_static=ck['use_static'], use_notes=ck['use_notes'],
            ).to(device)
            model.load_state_dict(ck['model_state_dict'])

            loader = DataLoader(splits['train'], batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_fn)
            Z, F, is_cat, names = collect(loader, model, device)

            mig_std, _ = standard_mig(Z, F, is_cat, bins=args.bins)
            sap_std, _ = standard_sap(Z, F, is_cat)
            mig_len, _ = compute_mig_lenient(Z, F)
            sap_len, _ = compute_sap_lenient(Z, F)
            dci = compute_dci(Z, F, is_cat)

            # Null control: permute the rows of Z so the latent-factor
            # relationship is destroyed while every marginal is preserved.
            # Scores at or below this floor mean no disentanglement at all.
            rng = np.random.default_rng(0)
            Zp = Z[rng.permutation(Z.shape[0])]
            mig_null, _ = standard_mig(Zp, F, is_cat, bins=args.bins)
            sap_null, _ = standard_sap(Zp, F, is_cat)
            dci_null = compute_dci(Zp, F, is_cat)

            row = {'checkpoint': cp.name, 'n_patients': int(Z.shape[0]),
                   'mig_standard': round(mig_std, 4), 'sap_standard': round(sap_std, 4),
                   'mig_lenient': round(float(mig_len), 4), 'sap_lenient': round(float(sap_len), 4),
                   'mig_null': round(mig_null, 4), 'sap_null': round(sap_null, 4),
                   'dci_D': round(dci['disentanglement'], 4),
                   'dci_C': round(dci['completeness'], 4),
                   'dci_I': round(dci['informativeness'], 4),
                   'dci_D_null': round(dci_null['disentanglement'], 4),
                   'dci_C_null': round(dci_null['completeness'], 4),
                   'dci_I_null': round(dci_null['informativeness'], 4)}
            results[run].append(row)
            print(f"  {run:28s} {cp.name}  MIG std={row['mig_standard']:.4f} "
                  f"lenient={row['mig_lenient']:.4f} null={row['mig_null']:.4f} | "
                  f"SAP std={row['sap_standard']:.4f} lenient={row['sap_lenient']:.4f} "
                  f"null={row['sap_null']:.4f} | DCI D={row['dci_D']:.3f} "
                  f"C={row['dci_C']:.3f} I={row['dci_I']:.3f} "
                  f"(null D={row['dci_D_null']:.3f} C={row['dci_C_null']:.3f} "
                  f"I={row['dci_I_null']:.3f})", flush=True)

    print("\n================ mean per run ================", flush=True)
    print(f"{'run':30s} {'MIG std':>9s} {'MIGnull':>9s} {'MIG len':>9s} {'SAP std':>9s} {'SAPnull':>9s} {'SAP len':>9s} {'DCI-D':>7s} {'Dnull':>7s} {'DCI-C':>7s} {'Cnull':>7s} {'DCI-I':>7s} {'Inull':>7s}")
    for run, rows in results.items():
        if not rows:
            continue
        m = lambda k: float(np.mean([r[k] for r in rows]))
        print(f"{run:30s} {m('mig_standard'):9.4f} {m('mig_null'):9.4f} {m('mig_lenient'):9.4f} "
              f"{m('sap_standard'):9.4f} {m('sap_null'):9.4f} {m('sap_lenient'):9.4f} "
              f"{m('dci_D'):7.3f} {m('dci_D_null'):7.3f} {m('dci_C'):7.3f} "
              f"{m('dci_C_null'):7.3f} {m('dci_I'):7.3f} {m('dci_I_null'):7.3f}", flush=True)

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
