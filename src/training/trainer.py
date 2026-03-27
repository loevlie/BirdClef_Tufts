"""Single-model ProtoSSM training with mixup, focal loss, and SWA."""

import numpy as np
import torch
import torch.nn.functional as F

from src.evaluation.metrics import macro_auc_skip_empty
from .losses import focal_bce_with_logits
from .augmentation import mixup_files


def train_proto_ssm_single(model, emb_train, logits_train, labels_train,
                           site_ids_train=None, hours_train=None,
                           emb_val=None, logits_val=None, labels_val=None,
                           site_ids_val=None, hours_val=None,
                           file_families_train=None, file_families_val=None,
                           cfg=None, verbose=True):
    """Train a single ProtoSSM v4 model with mixup, focal loss, and SWA.

    Parameters
    ----------
    model : ProtoSSMv2
        Model instance to train (modified in-place).
    emb_train : ndarray, shape (n_files, n_windows, d_emb)
        Training embeddings reshaped to file-level.
    logits_train : ndarray, shape (n_files, n_windows, n_classes)
        Training Perch logits reshaped to file-level.
    labels_train : ndarray, shape (n_files, n_windows, n_classes)
        Training labels reshaped to file-level.
    site_ids_train, hours_train : ndarray or None
        Per-file site and hour indices.
    emb_val, logits_val, labels_val : ndarray or None
        Validation data (same shapes, different files).
    site_ids_val, hours_val : ndarray or None
        Per-file site and hour indices for validation.
    file_families_train, file_families_val : ndarray or None
        Taxonomic family soft-labels for auxiliary loss.
    cfg : dict
        Training hyperparameters dict (proto_ssm_train section of CFG).
    verbose : bool
        Print progress every 20 epochs.

    Returns
    -------
    model : ProtoSSMv2
        Trained model with best/SWA weights loaded.
    history : dict
        Training history with train_loss, val_loss, val_auc lists.
    """
    if cfg is None:
        raise ValueError("cfg dict is required (pass CFG['proto_ssm_train'])")

    label_smoothing = cfg.get("label_smoothing", 0.0)
    mixup_alpha = cfg.get("mixup_alpha", 0.0)
    focal_gamma = cfg.get("focal_gamma", 0.0)
    swa_start_frac = cfg.get("swa_start_frac", 1.0)  # 1.0 = disabled
    n_epochs = cfg["n_epochs"]
    swa_start_epoch = int(n_epochs * swa_start_frac)

    # Convert to tensors (base -- unmixed)
    labels_np = labels_train.copy()

    # Apply label smoothing
    if label_smoothing > 0:
        labels_np = labels_np * (1.0 - label_smoothing) + label_smoothing / 2.0

    has_val = emb_val is not None
    if has_val:
        emb_v = torch.tensor(emb_val, dtype=torch.float32)
        logits_v = torch.tensor(logits_val, dtype=torch.float32)
        labels_v = torch.tensor(labels_val, dtype=torch.float32)
        site_v = torch.tensor(site_ids_val, dtype=torch.long) if site_ids_val is not None else None
        hour_v = torch.tensor(hours_val, dtype=torch.long) if hours_val is not None else None

    fam_v = torch.tensor(file_families_val, dtype=torch.float32) if (has_val and file_families_val is not None) else None

    # Class weights for imbalanced data
    labels_tr_t = torch.tensor(labels_np, dtype=torch.float32)
    pos_counts = labels_tr_t.sum(dim=(0, 1))
    total = labels_tr_t.shape[0] * labels_tr_t.shape[1]
    pos_weight = ((total - pos_counts) / (pos_counts + 1)).clamp(max=cfg["pos_weight_cap"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"],
        epochs=n_epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy='cos'
    )

    best_val_loss = float('inf')
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    # SWA state accumulator
    swa_state = None
    swa_count = 0

    for epoch in range(n_epochs):
        # === Mixup augmentation (per-epoch re-sampling) ===
        if mixup_alpha > 0 and epoch > 5:  # Skip mixup for first 5 epochs (warmup)
            emb_mix, logits_mix, labels_mix, _, _, fam_mix = mixup_files(
                emb_train, logits_train, labels_np,
                site_ids_train, hours_train, file_families_train,
                alpha=mixup_alpha,
            )
        else:
            emb_mix, logits_mix, labels_mix = emb_train, logits_train, labels_np
            fam_mix = file_families_train

        emb_tr = torch.tensor(emb_mix, dtype=torch.float32)
        logits_tr = torch.tensor(logits_mix, dtype=torch.float32)
        labels_tr = torch.tensor(labels_mix, dtype=torch.float32)
        site_tr = torch.tensor(site_ids_train, dtype=torch.long) if site_ids_train is not None else None
        hour_tr = torch.tensor(hours_train, dtype=torch.long) if hours_train is not None else None
        fam_tr = torch.tensor(fam_mix, dtype=torch.float32) if fam_mix is not None else None

        # === Train ===
        model.train()
        species_out, family_out, _ = model(emb_tr, logits_tr, site_ids=site_tr, hours=hour_tr)

        # Primary loss: focal BCE or weighted BCE
        if focal_gamma > 0:
            loss_main = focal_bce_with_logits(
                species_out, labels_tr,
                gamma=focal_gamma,
                pos_weight=pos_weight[None, None, :],
            )
        else:
            loss_main = F.binary_cross_entropy_with_logits(
                species_out, labels_tr,
                pos_weight=pos_weight[None, None, :]
            )

        # Knowledge distillation loss
        loss_distill = F.mse_loss(species_out, logits_tr)

        # Total loss
        loss = loss_main + cfg["distill_weight"] * loss_distill

        # Taxonomic auxiliary loss
        if family_out is not None and fam_tr is not None:
            loss_family = F.binary_cross_entropy_with_logits(family_out, fam_tr)
            loss = loss + 0.1 * loss_family

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # === SWA accumulation ===
        if epoch >= swa_start_epoch:
            if swa_state is None:
                swa_state = {k: v.clone() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k in swa_state:
                    swa_state[k] += model.state_dict()[k]
                swa_count += 1

        # === Validate ===
        model.eval()
        with torch.no_grad():
            if has_val:
                val_out, val_fam, _ = model(emb_v, logits_v, site_ids=site_v, hours=hour_v)
                val_loss = F.binary_cross_entropy_with_logits(
                    val_out, labels_v,
                    pos_weight=pos_weight[None, None, :]
                )

                val_pred = val_out.reshape(-1, val_out.shape[-1]).numpy()
                val_true = labels_v.reshape(-1, labels_v.shape[-1]).numpy()
                try:
                    val_auc = macro_auc_skip_empty(val_true, val_pred)
                except Exception:
                    val_auc = 0.0
            else:
                val_loss = loss
                val_auc = 0.0

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["val_auc"].append(val_auc)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and (epoch + 1) % 20 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            swa_info = f" swa={swa_count}" if swa_count > 0 else ""
            print(f"  Epoch {epoch+1:3d}: train={loss.item():.4f} val={val_loss.item():.4f} "
                  f"auc={val_auc:.4f} lr={lr_now:.6f} wait={wait}{swa_info}")

        if wait >= cfg["patience"]:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
            break

    # Apply SWA if we accumulated enough checkpoints
    if swa_state is not None and swa_count >= 3:
        if verbose:
            print(f"  Applying SWA (averaged {swa_count} checkpoints)")
        avg_state = {k: v / swa_count for k, v in swa_state.items()}
        model.load_state_dict(avg_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"  Training complete. Best val_loss={best_val_loss:.4f}")
        with torch.no_grad():
            alphas = torch.sigmoid(model.fusion_alpha).numpy()
            print(f"  Fusion alpha: mean={alphas.mean():.3f} min={alphas.min():.3f} max={alphas.max():.3f}")
            print(f"  Proto temperature: {F.softplus(model.proto_temp).item():.3f}")

    return model, history
