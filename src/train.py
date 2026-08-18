"""
Reproducible Temporal Fusion Nexus training.

Implements the paper's training setup:
  - L_total = L_recon + lambda_decorr * L_decorr + lambda_disent * L_disent  (Eq. 6)
  - L_recon averaged over H = 10 future steps                                (Eq. 7)
  - five-fold cross-validation, 80% train / 20% evaluation per fold
  - Adam, lr 5e-4, batch size 32, 4 attention heads                          (Supplement B)

Early stopping on a validation slice carved out of the training folds, so the
evaluation fold is never touched during model selection. The paper does not
state an epoch count; the epoch actually reached is recorded per fold in
metrics.json.

Usage:
    python src/train.py --project-root . --embeddings data/embeddings/emb_med_gte_hybrid_de.npy \
        --out-dir runs/paper_aligned
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
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
    get_valid_patient_ids,
)
from utils import (
    FeatureDecoders,
    build_elapsed_times,
    build_future_windows,
    correlation_loss,
    feature_prediction_loss,
    flatten_valid_timesteps,
    get_last_valid_step,
    multistep_reconstruction_loss,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subsample_origins(mask: torch.Tensor, max_origins: int | None, generator: torch.Generator | None = None):
    """
    Choose which (batch, timestep) positions to decode from.

    Returns a boolean tensor shaped like `mask`. With max_origins set, a random
    subset of valid timesteps is kept so decoder memory stays bounded on
    batches that contain very long sequences.
    """
    valid = mask.bool()
    if not max_origins or valid.sum().item() <= max_origins:
        return valid

    flat = valid.reshape(-1)
    idx = flat.nonzero(as_tuple=True)[0]
    perm = torch.randperm(idx.numel(), device=idx.device, generator=generator)[:max_origins]
    keep = torch.zeros_like(flat)
    keep[idx[perm]] = True
    return keep.reshape(valid.shape)


def run_batch(batch, model, feature_decoders, device, args, train: bool):
    cat_features = batch['static_categorical_features'].to(device)
    num_features = batch['static_numerical_features'].to(device)

    ts_values = batch['ts_features'].to(device)          # (B, T, F)
    timesteps = batch['timesteps'].to(device)            # (B, T)
    value_mask = batch['value_mask'].to(device)          # (B, T, F)
    step_mask = batch['mask'].to(device)                 # (B, T)

    notes_embeddings = batch['notes_embeddings'].to(device)
    notes_timesteps = batch['notes_timesteps'].to(device)
    notes_mask = batch['notes_mask'].to(device)

    elapsed_times = build_elapsed_times(timesteps, step_mask)

    future_elapsed, future_values, target_mask = build_future_windows(
        timesteps=timesteps,
        ts_values=ts_values,
        value_mask=value_mask,
        step_mask=step_mask,
        horizon=CONFIG['prediction_horizon'],
    )

    # Decoding from every timestep is quadratic in sequence length; cap the
    # number of origins so a batch containing one very long patient cannot
    # blow up decoder memory. The subset is random, so L_recon stays unbiased.
    origins = subsample_origins(step_mask, args.max_decode_origins if train else None)
    target_mask = target_mask * origins.view(*origins.shape, 1, 1).to(target_mask.dtype)

    predictions, Z, attn, static_encoding, static_weights = model(
        x=ts_values,
        elapsed_times=elapsed_times,
        timesteps=timesteps,
        notes_embeddings=notes_embeddings,
        notes_timesteps=notes_timesteps,
        static_features=(cat_features, num_features),
        mask=step_mask,
        notes_mask=notes_mask,
        value_mask=value_mask,
        future_elapsed=future_elapsed,
        decode_mask=origins,
    )

    # L_recon (Eq. 7)
    recon = multistep_reconstruction_loss(predictions, future_values, target_mask)

    # L_decorr (Eq. 8), over the Nexus embedding at every valid timestep
    z_valid = flatten_valid_timesteps(Z, step_mask)
    decorr = correlation_loss(z_valid, reduction=args.decorr_reduction)

    # L_disent (Eq. 9), factors predicted from the patient's final representation
    last_hidden = get_last_valid_step(Z, step_mask)
    last_ts = get_last_valid_step(ts_values, step_mask)
    feature_preds, soft_masks = feature_decoders(last_hidden)
    disent = feature_prediction_loss(
        feature_preds, soft_masks, cat_features, num_features, last_ts,
        alpha=CONFIG['alpha_disent'],
    )

    total = recon + args.lambda_decorr * decorr + args.lambda_disent * disent
    return total, {'recon': recon.item(), 'decorr': decorr.item(), 'disent': disent.item()}


def evaluate(loader, model, feature_decoders, device, args):
    model.eval()
    feature_decoders.eval()
    sums = {'total': 0.0, 'recon': 0.0, 'decorr': 0.0, 'disent': 0.0}
    n = 0
    with torch.no_grad():
        for batch in loader:
            total, parts = run_batch(batch, model, feature_decoders, device, args, train=False)
            sums['total'] += total.item()
            for k, v in parts.items():
                sums[k] += v
            n += 1
    if n == 0:
        return {k: float('nan') for k in sums}
    return {k: v / n for k, v in sums.items()}


def train_fold(fold, split_ids, data, args, device):
    set_seed(args.seed + fold)

    splits = create_dataset_splits(
        static_df=data['static_df'],
        ts_data=data['ts_data'],
        notes_df=data['notes'],
        biopsy_df=data['biopsy_df'],
        patient_split_ids=split_ids,
        require_notes=args.require_notes,
        random_state=args.seed,
    )

    train_dataset = splits['train']
    val_dataset = splits['val']
    test_dataset = splits['test']

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)

    encoder = TimeAwareAttentionEncoder(use_temporal_attention=True)
    model = MultiModal(
        encoder,
        categorical_cardinalities=train_dataset.categorical_cardinalities,
        use_static=True,
        use_notes=args.use_notes,
    ).to(device)

    feature_decoders = FeatureDecoders(
        CONFIG['lstm_hidden_size'], CONFIG,
        categorical_cardinalities=train_dataset.categorical_cardinalities,
    ).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(feature_decoders.parameters()),
        lr=CONFIG['learning_rate'],
    )

    best_val = float('inf')
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        feature_decoders.train()
        epoch_start = time.time()
        running = {'total': 0.0, 'recon': 0.0, 'decorr': 0.0, 'disent': 0.0}
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            total, parts = run_batch(batch, model, feature_decoders, device, args, train=True)
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(feature_decoders.parameters()), args.grad_clip
                )
            optimizer.step()

            running['total'] += total.item()
            for k, v in parts.items():
                running[k] += v
            n_batches += 1

        train_metrics = {k: v / max(n_batches, 1) for k, v in running.items()}
        val_metrics = evaluate(val_loader, model, feature_decoders, device, args)

        history.append({'epoch': epoch, 'train': train_metrics, 'val': val_metrics,
                        'seconds': round(time.time() - epoch_start, 1)})
        print(
            f"[fold {fold}] epoch {epoch:3d} "
            f"train_total={train_metrics['total']:.4f} train_recon={train_metrics['recon']:.4f} "
            f"val_total={val_metrics['total']:.4f} val_recon={val_metrics['recon']:.4f} "
            f"({history[-1]['seconds']}s)",
            flush=True,
        )

        # Model selection on validation reconstruction loss.
        if val_metrics['recon'] < best_val - args.min_delta:
            best_val = val_metrics['recon']
            best_epoch = epoch
            best_state = {
                'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'feature_decoders': {k: v.detach().cpu().clone() for k, v in feature_decoders.state_dict().items()},
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"[fold {fold}] early stopping at epoch {epoch} (best epoch {best_epoch})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state['model'])
        feature_decoders.load_state_dict(best_state['feature_decoders'])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"fold{fold}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_decoders_state_dict': feature_decoders.state_dict(),
        'preprocessing_artifacts': splits['preprocessing_artifacts'],
        'patient_ids': splits['patient_ids'],
        'config': CONFIG,
        'model_class': 'MultiModal',
        'encoder_class': 'TimeAwareAttentionEncoder',
        'use_static': True,
        'use_notes': args.use_notes,
        'fold': fold,
        'best_epoch': best_epoch,
        'best_val_recon': best_val,
    }, checkpoint_path)

    return {
        'fold': fold,
        'best_epoch': best_epoch,
        'best_val_recon': best_val,
        'n_train': len(train_dataset),
        'n_val': len(val_dataset),
        'n_test': len(test_dataset),
        'checkpoint': str(checkpoint_path),
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Temporal Fusion Nexus with five-fold cross-validation")
    parser.add_argument('--project-root', default='.', help="Directory containing data/v1")
    parser.add_argument('--embeddings', required=True, help="Path to the note embedding .npy file")
    parser.add_argument('--out-dir', default='runs/paper_aligned')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=8, help="Epochs without val improvement before stopping")
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--val-fraction', type=float, default=0.1,
                        help="Fraction of the training folds held out for early stopping")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--lambda-decorr', type=float, default=CONFIG['lambda_decorr'])
    parser.add_argument('--lambda-disent', type=float, default=CONFIG['lambda_disent'])
    parser.add_argument('--decorr-reduction', choices=['sum', 'dim', 'mean'], default='sum',
                        help="How Eq. 8's off-diagonal terms are aggregated; 'sum' is Eq. 8 as printed")
    parser.add_argument('--use-notes', dest='use_notes', action='store_true', default=True)
    parser.add_argument('--no-notes', dest='use_notes', action='store_false')
    parser.add_argument('--require-notes', action='store_true', default=False)
    parser.add_argument('--max-patients', type=int, default=None, help="Smoke-test cap on cohort size")
    parser.add_argument('--max-decode-origins', type=int, default=8192,
                        help="Cap decoder origins per training batch to bound memory (0 = no cap)")
    parser.add_argument('--only-fold', type=int, default=None, help="Run a single fold (for parallel jobs)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} folds={args.folds} horizon={CONFIG['prediction_horizon']} "
          f"heads={CONFIG['num_heads']} lr={CONFIG['learning_rate']} batch={CONFIG['batch_size']} "
          f"lambda_decorr={args.lambda_decorr} lambda_disent={args.lambda_disent}", flush=True)

    set_seed(args.seed)

    print("loading data...", flush=True)
    dfs = get_dfs(args.project_root)
    static_df = create_static_df(dfs)
    medication_df = create_medication_df(dfs)
    vitals_ca, vitals_lab = create_vitals_df(dfs)
    biopsy_df = dfs['biopsy']
    ts_data = create_ts_data(vitals_ca, vitals_lab, medication_df, merge_lab=True, merge_med=True, static_df=static_df)
    notes = create_notes_df(dfs, filename=args.embeddings)

    eligible = get_valid_patient_ids(
        static_df=static_df, ts_data=ts_data, notes_df=notes, require_notes=args.require_notes
    )
    eligible = np.asarray(eligible)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(eligible)
    if args.max_patients is not None:
        eligible = eligible[:args.max_patients]
    print(f"eligible patients: {len(eligible)}", flush=True)

    data = {'static_df': static_df, 'ts_data': ts_data, 'notes': notes, 'biopsy_df': biopsy_df}

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(eligible)):
        if args.only_fold is not None and fold != args.only_fold:
            continue

        train_ids = eligible[train_idx]
        test_ids = eligible[test_idx]

        # Early-stopping split carved out of the training folds only, so the
        # 20% evaluation fold stays untouched by model selection.
        n_val = max(1, int(round(len(train_ids) * args.val_fraction)))
        fold_rng = np.random.default_rng(args.seed + fold)
        shuffled = train_ids.copy()
        fold_rng.shuffle(shuffled)
        val_ids = shuffled[:n_val]
        fit_ids = shuffled[n_val:]

        split_ids = {
            'selected': eligible,
            'train': fit_ids,
            'val': val_ids,
            'test': test_ids,
        }

        print(f"\n=== fold {fold}: train={len(fit_ids)} val={len(val_ids)} test={len(test_ids)} ===", flush=True)
        results.append(train_fold(fold, split_ids, data, args, device))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({'args': vars(args), 'config': CONFIG, 'folds': results}, f, indent=2, default=str)
    print(f"\nwrote {out_dir / 'metrics.json'}", flush=True)


if __name__ == '__main__':
    main()
