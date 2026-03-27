"""Reshape flat window arrays into per-file blocks."""

from src.constants import N_WINDOWS


def reshape_to_files(flat_array, meta_df, n_windows=N_WINDOWS):
    filenames = meta_df["filename"].to_numpy()
    unique_files = []
    seen = set()
    for f in filenames:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)

    n_files = len(unique_files)
    assert len(flat_array) == n_files * n_windows, \
        f"Expected {n_files * n_windows} rows, got {len(flat_array)}"

    new_shape = (n_files, n_windows) + flat_array.shape[1:]
    return flat_array.reshape(new_shape), unique_files
