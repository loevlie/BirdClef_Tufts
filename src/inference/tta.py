"""Temporal-shift test-time augmentation for ProtoSSM."""

import numpy as np
import torch


def temporal_shift_tta(emb_files, logits_files, model, site_ids, hours, shifts=[0, 1, -1]):
    """TTA by circular-shifting the 12-window embedding sequence.
    Averages predictions from shifted versions for more robust output."""
    all_preds = []
    model.eval()

    for shift in shifts:
        if shift == 0:
            e = emb_files
            l = logits_files
        else:
            e = np.roll(emb_files, shift, axis=1)
            l = np.roll(logits_files, shift, axis=1)

        with torch.no_grad():
            out, _, _ = model(
                torch.tensor(e, dtype=torch.float32),
                torch.tensor(l, dtype=torch.float32),
                site_ids=torch.tensor(site_ids, dtype=torch.long),
                hours=torch.tensor(hours, dtype=torch.long),
            )
            pred = out.numpy()

        # Reverse the shift on predictions
        if shift != 0:
            pred = np.roll(pred, -shift, axis=1)

        all_preds.append(pred)

    return np.mean(all_preds, axis=0)
