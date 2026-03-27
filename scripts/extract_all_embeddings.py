#!/usr/bin/env python3
"""Extract Perch embeddings for ALL train soundscapes (labeled + unlabeled).

Saves to cache/all_soundscapes_emb.npz + cache/all_soundscapes_meta.parquet.
This is the first step for self-supervised pretraining.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import SR, N_WINDOWS, WINDOW_SAMPLES, FILE_SAMPLES
from src.data.parsing import parse_soundscape_filename
from src.inference.audio import read_soundscape_60s
from tqdm.auto import tqdm


def main():
    p = argparse.ArgumentParser(description="Extract Perch embeddings for all soundscapes")
    p.add_argument("--data-dir", default="data/competition")
    p.add_argument("--model-dir", default=None, help="Perch model dir")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--batch-files", type=int, default=32)
    p.add_argument("--max-files", type=int, default=None, help="Limit for testing")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    out_emb = cache_dir / "all_soundscapes_emb.npz"
    out_meta = cache_dir / "all_soundscapes_meta.parquet"

    if out_emb.exists() and out_meta.exists():
        arr = np.load(out_emb)
        print(f"Cache exists: {out_emb} ({arr['embeddings'].shape})")
        print("Delete it to recompute.")
        return

    # Find all soundscape files
    soundscape_dir = data_dir / "train_soundscapes"
    all_files = sorted(soundscape_dir.glob("*.ogg"))
    if args.max_files:
        all_files = all_files[:args.max_files]
    print(f"Found {len(all_files)} soundscape files")

    # Load Perch
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    model_dir = args.model_dir
    if model_dir is None:
        import kagglehub
        model_dir = kagglehub.model_download(
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"
        )
    model_dir = Path(model_dir)

    print(f"Loading Perch from {model_dir}...")
    birdclassifier = tf.saved_model.load(str(model_dir))
    infer_fn = birdclassifier.signatures["serving_default"]
    print("Perch loaded.")

    # Extract embeddings in batches
    n_files = len(all_files)
    n_rows = n_files * N_WINDOWS
    batch_files = args.batch_files

    embeddings = np.zeros((n_rows, 1536), dtype=np.float32)
    filenames = []
    sites = []
    hours = []

    write_row = 0
    t0 = time.time()

    for start in tqdm(range(0, n_files, batch_files), desc="Perch batches"):
        batch_paths = all_files[start:start + batch_files]
        batch_n = len(batch_paths)

        x = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        x_pos = 0

        for path in batch_paths:
            y = read_soundscape_60s(path)
            x[x_pos:x_pos + N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
            x_pos += N_WINDOWS

            meta = parse_soundscape_filename(path.name)
            for _ in range(N_WINDOWS):
                filenames.append(path.name)
                sites.append(meta["site"])
                hours.append(int(meta["hour_utc"]))

        outputs = infer_fn(inputs=tf.convert_to_tensor(x[:x_pos]))
        emb = outputs["embedding"].numpy().astype(np.float32, copy=False)
        embeddings[write_row:write_row + x_pos] = emb
        write_row += x_pos

        del x, outputs, emb
        gc.collect()

        # Progress
        elapsed = time.time() - t0
        files_done = start + batch_n
        rate = files_done / elapsed
        eta = (n_files - files_done) / rate if rate > 0 else 0
        if (start // batch_files) % 10 == 0:
            print(f"  {files_done}/{n_files} files, {rate:.1f} files/s, ETA {eta/60:.0f}min")

    embeddings = embeddings[:write_row]

    # Save
    meta_df = pd.DataFrame({
        "filename": filenames,
        "site": sites,
        "hour_utc": hours,
    })

    np.savez_compressed(out_emb, embeddings=embeddings)
    meta_df.to_parquet(out_meta, index=False)

    elapsed = time.time() - t0
    n_unique = meta_df["filename"].nunique()
    print(f"\nDone: {n_unique} files, {len(embeddings)} windows in {elapsed/60:.1f} min")
    print(f"Saved: {out_emb} ({embeddings.shape})")
    print(f"Saved: {out_meta}")


if __name__ == "__main__":
    main()
