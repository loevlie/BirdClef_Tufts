"""Perch inference with embedding extraction and selective proxies."""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants import N_WINDOWS, WINDOW_SAMPLES
from src.data.parsing import parse_soundscape_filename
from src.inference.audio import read_soundscape_60s


def infer_perch_with_embeddings(paths, infer_fn, n_classes, mapped_pos, mapped_bc_indices,
                                proxy_pos_to_bc=None, batch_files=16, verbose=True,
                                proxy_reduce="max"):
    """Run Perch on a list of soundscape paths and return metadata, scores, and embeddings.

    Parameters
    ----------
    paths : list[str | Path]
        Soundscape OGG file paths.
    infer_fn : callable
        TF serving signature (``birdclassifier.signatures["serving_default"]``).
    n_classes : int
        Number of competition classes (N_CLASSES).
    mapped_pos : np.ndarray[int32]
        Indices into the competition label list that have a direct Perch mapping.
    mapped_bc_indices : np.ndarray[int32]
        Corresponding BirdClassifier label indices for *mapped_pos*.
    proxy_pos_to_bc : dict[int, np.ndarray] | None
        ``{competition_label_pos: array_of_bc_indices}`` for genus-proxy species.
    batch_files : int
        Number of files per GPU batch.
    verbose : bool
        Show a tqdm progress bar.
    proxy_reduce : str
        Aggregation over proxy BirdClassifier indices (``"max"`` or ``"mean"``).

    Returns
    -------
    meta_df : pd.DataFrame
        Columns: row_id, filename, site, hour_utc.
    scores : np.ndarray, shape (n_files * N_WINDOWS, n_classes)
    embeddings : np.ndarray, shape (n_files * N_WINDOWS, 1536)
    """
    if proxy_pos_to_bc is None:
        proxy_pos_to_bc = {}

    paths = [Path(p) for p in paths]
    n_files = len(paths)
    n_rows = n_files * N_WINDOWS

    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)

    scores = np.zeros((n_rows, n_classes), dtype=np.float32)
    embeddings = np.zeros((n_rows, 1536), dtype=np.float32)

    write_row = 0
    iterator = range(0, n_files, batch_files)
    if verbose:
        iterator = tqdm(iterator, total=(n_files + batch_files - 1) // batch_files, desc="Perch batches")

    for start in iterator:
        batch_paths = paths[start:start + batch_files]
        batch_n = len(batch_paths)

        x = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        batch_row_start = write_row
        x_pos = 0

        for path in batch_paths:
            y = read_soundscape_60s(path)
            x[x_pos:x_pos + N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)

            meta = parse_soundscape_filename(path.name)
            stem = path.stem

            row_ids[write_row:write_row + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
            filenames[write_row:write_row + N_WINDOWS] = path.name
            sites[write_row:write_row + N_WINDOWS] = meta["site"]
            hours[write_row:write_row + N_WINDOWS] = int(meta["hour_utc"])

            x_pos += N_WINDOWS
            write_row += N_WINDOWS

        import tensorflow as tf
        outputs = infer_fn(inputs=tf.convert_to_tensor(x))
        logits = outputs["label"].numpy().astype(np.float32, copy=False)
        emb = outputs["embedding"].numpy().astype(np.float32, copy=False)

        scores[batch_row_start:write_row, mapped_pos] = logits[:, mapped_bc_indices]
        embeddings[batch_row_start:write_row] = emb

        # Selected frog proxies
        for pos, bc_idx_arr in proxy_pos_to_bc.items():
            sub = logits[:, bc_idx_arr]
            if proxy_reduce == "max":
                proxy_score = sub.max(axis=1)
            elif proxy_reduce == "mean":
                proxy_score = sub.mean(axis=1)
            else:
                raise ValueError("proxy_reduce must be 'max' or 'mean'")
            scores[batch_row_start:write_row, pos] = proxy_score.astype(np.float32)

        del x, outputs, logits, emb
        gc.collect()

    meta_df = pd.DataFrame({
        "row_id": row_ids,
        "filename": filenames,
        "site": sites,
        "hour_utc": hours,
    })

    return meta_df, scores, embeddings
