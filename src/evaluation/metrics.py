"""Evaluation metrics for multi-label bird classification."""

from sklearn.metrics import roc_auc_score


def macro_auc_skip_empty(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")
