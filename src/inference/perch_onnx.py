"""Perch v2 inference via ONNX Runtime — ~9x faster than TensorFlow."""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants import N_WINDOWS, WINDOW_SAMPLES
from src.data.parsing import parse_soundscape_filename
from src.inference.audio import read_soundscape_60s


def load_onnx_session(model_path):
    """Load ONNX model and return an inference session."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 4
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
    return session


def infer_perch_onnx(paths, session, n_classes, mapped_pos, mapped_bc_indices,
                     proxy_pos_to_bc=None, batch_files=32, verbose=True,
                     proxy_reduce="max"):
    """Run Perch v2 ONNX inference. Drop-in replacement for infer_perch_with_embeddings.

    Parameters
    ----------
    paths : list of Path
    session : onnxruntime.InferenceSession
    n_classes : int
    mapped_pos, mapped_bc_indices : ndarray
    proxy_pos_to_bc : dict or None
    batch_files : int
    verbose : bool
    proxy_reduce : str

    Returns
    -------
    meta_df : DataFrame
    scores : ndarray (n_rows, n_classes)
    embeddings : ndarray (n_rows, 1536)
    """
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
        iterator = tqdm(iterator, total=(n_files + batch_files - 1) // batch_files, desc="Perch ONNX batches")

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

        # ONNX inference
        outputs = session.run(None, {"inputs": x[:x_pos]})
        # outputs: [embedding, spatial_embedding, spectrogram, label]
        emb = outputs[0].astype(np.float32)
        logits = outputs[3].astype(np.float32)

        scores[batch_row_start:write_row, mapped_pos] = logits[:, mapped_bc_indices]
        embeddings[batch_row_start:write_row] = emb

        # Proxy scores
        if proxy_pos_to_bc:
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
        "row_id": row_ids[:write_row],
        "filename": filenames[:write_row],
        "site": sites[:write_row],
        "hour_utc": hours[:write_row],
    })

    return meta_df, scores[:write_row], embeddings[:write_row]
