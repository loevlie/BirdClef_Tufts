"""Temporal-shift test-time augmentation for ProtoSSM."""

import gc

import numpy as np
import torch


def temporal_shift_tta(emb_files, logits_files, model, site_ids, hours, shifts=[0, 1, -1]):
    """TTA by circular-shifting the 12-window embedding sequence.
    Averages predictions from shifted versions for more robust output.
    Memory-efficient: accumulates running mean instead of storing all preds."""
    model.eval()
    running_sum = None

    site_t = torch.tensor(site_ids, dtype=torch.long)
    hour_t = torch.tensor(hours, dtype=torch.long)

    for shift in shifts:
        if shift == 0:
            e = torch.tensor(emb_files, dtype=torch.float32)
            l = torch.tensor(logits_files, dtype=torch.float32)
        else:
            e = torch.tensor(np.roll(emb_files, shift, axis=1), dtype=torch.float32)
            l = torch.tensor(np.roll(logits_files, shift, axis=1), dtype=torch.float32)

        with torch.no_grad():
            out, _, _ = model(e, l, site_ids=site_t, hours=hour_t)
            pred = out.numpy()

        del e, l, out
        gc.collect()

        if shift != 0:
            pred = np.roll(pred, -shift, axis=1)

        if running_sum is None:
            running_sum = pred
        else:
            running_sum += pred

    return running_sum / len(shifts)
