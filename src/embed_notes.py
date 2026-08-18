"""
Generate clinical-note embeddings for the Temporal Fusion Nexus.

Defaults to the paper's text encoder, Med-GTE-hybrid-de. The embedding array is
written in the row order of create_notes_df(dfs, filename=None), which is the
order create_notes_df expects when it later attaches the array.

Usage:
    python src/embed_notes.py --project-root . \
        --out data/embeddings/emb_med_gte_hybrid_de.npy
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from models import NotesEncoder
from preprocessing import create_notes_df, get_dfs


def main():
    parser = argparse.ArgumentParser(description="Embed clinical notes")
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--model', default=CONFIG['notes_encoder_model'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-shards', type=int, default=1,
                        help="Split the note list into contiguous shards for parallel GPUs")
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--merge', action='store_true',
                        help="Concatenate {out}.shard*.npy into {out} and exit")
    args = parser.parse_args()

    if args.merge:
        parts = []
        for k in range(args.num_shards):
            path = f"{args.out}.shard{k}.npy"
            if not os.path.exists(path):
                raise SystemExit(f"missing shard: {path}")
            parts.append(np.load(path))
        embeddings = np.concatenate(parts)
        np.save(args.out, embeddings)
        print(f"merged {args.num_shards} shards -> {args.out} with shape {embeddings.shape}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} model={args.model}", flush=True)

    dfs = get_dfs(args.project_root)
    notes = create_notes_df(dfs, filename=None)
    all_texts = notes['text'].tolist()

    # Contiguous shards so that merging in shard order restores the note order.
    total = len(all_texts)
    bounds = np.linspace(0, total, args.num_shards + 1).astype(int)
    start, end = bounds[args.shard], bounds[args.shard + 1]
    texts = all_texts[start:end]
    out_path = args.out if args.num_shards == 1 else f"{args.out}.shard{args.shard}.npy"
    print(f"notes total: {total}; shard {args.shard}/{args.num_shards} rows [{start}:{end}] "
          f"-> {out_path}", flush=True)

    encoder = NotesEncoder(model_name=args.model).to(device)
    encoder.model.eval()

    chunks = []
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch = texts[i:i + args.batch_size]
        chunks.append(encoder(batch).cpu().numpy())

    embeddings = np.concatenate(chunks)
    if embeddings.shape[0] != len(texts):
        raise RuntimeError(f"embedded {embeddings.shape[0]} rows for {len(texts)} notes")
    if embeddings.shape[1] != CONFIG['notes_embedding_dim']:
        raise RuntimeError(
            f"embedding dim {embeddings.shape[1]} != CONFIG['notes_embedding_dim'] "
            f"{CONFIG['notes_embedding_dim']}"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(out_path, embeddings)
    print(f"wrote {out_path} with shape {embeddings.shape}", flush=True)


if __name__ == '__main__':
    main()
