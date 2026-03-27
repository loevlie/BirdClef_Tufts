"""Build the final submission DataFrame."""

import numpy as np
import pandas as pd


def build_submission(probs, meta_test, primary_labels, test_paths, n_windows=12):
    """Create a competition-ready submission DataFrame.

    Parameters
    ----------
    probs : np.ndarray, shape (n_rows, n_classes)
        Calibrated probabilities.
    meta_test : pd.DataFrame
        Must contain a ``row_id`` column.
    primary_labels : list[str]
        Ordered species labels (column names).
    test_paths : list
        Test soundscape file paths (used for row-count assertion).
    n_windows : int
        Windows per file.

    Returns
    -------
    pd.DataFrame
    """
    submission = pd.DataFrame(probs, columns=primary_labels)
    submission.insert(0, "row_id", meta_test["row_id"].values)
    submission[primary_labels] = submission[primary_labels].astype(np.float32)

    expected_rows = len(test_paths) * n_windows
    assert len(submission) == expected_rows, f"Expected {expected_rows}, got {len(submission)}"
    assert not submission.isna().any().any(), "Submission contains NaN values"

    return submission
