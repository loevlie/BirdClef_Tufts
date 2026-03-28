"""Self-supervised pretraining for ProtoSSM via masked window prediction.

Trains the SSM temporal encoder on ALL soundscapes (labeled + unlabeled)
to reconstruct randomly masked Perch embedding windows from context.
This teaches temporal patterns (dawn chorus, call-response, etc.)
before any labels are seen.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import N_WINDOWS
from src.models.ssm import SelectiveSSM


class MaskedSSMEncoder(nn.Module):
    """SSM encoder with masked window prediction head.

    Architecture matches ProtoSSMv2's encoder (input_proj + pos_enc +
    bidirectional SSM layers) so weights can be transferred.
    """

    def __init__(self, d_input=1536, d_model=128, d_state=16,
                 n_ssm_layers=2, n_windows=12, dropout=0.15,
                 n_sites=20, meta_dim=16):
        super().__init__()
        self.d_model = d_model
        self.n_windows = n_windows

        # Encoder (matches ProtoSSMv2)
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        # Site + hour metadata (matches ProtoSSMv2)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        # Bidirectional SSM layers (matches ProtoSSMv2)
        self.ssm_layers_fwd = nn.ModuleList()
        self.ssm_layers_bwd = nn.ModuleList()
        self.ssm_merges = nn.ModuleList()
        self.ssm_norms = nn.ModuleList()
        self.ssm_drops = nn.ModuleList()

        for _ in range(n_ssm_layers):
            self.ssm_layers_fwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_layers_bwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_merges.append(nn.Linear(2 * d_model, d_model))
            self.ssm_norms.append(nn.LayerNorm(d_model))
            self.ssm_drops.append(nn.Dropout(dropout))

        # Reconstruction head (SSL-specific, not transferred)
        self.reconstruct_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_input),
        )

    def encode(self, emb, site_ids=None, hours=None):
        """Run the encoder (shared with ProtoSSMv2). Returns (B, T, d_model)."""
        h = self.input_proj(emb)
        h = h + self.pos_enc[:, :h.shape[1], :]

        if site_ids is not None and hours is not None:
            s_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
            h_e = self.hour_emb(hours.clamp(0, 23))
            meta = self.meta_proj(torch.cat([s_e, h_e], dim=-1))
            h = h + meta.unsqueeze(1)

        for fwd, bwd, merge, norm, drop in zip(
            self.ssm_layers_fwd, self.ssm_layers_bwd,
            self.ssm_merges, self.ssm_norms, self.ssm_drops
        ):
            residual = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h = merge(torch.cat([h_f, h_b], dim=-1))
            h = drop(h)
            h = norm(h + residual)

        return h

    def forward(self, emb, mask, site_ids=None, hours=None):
        """Forward pass with masking.

        Parameters
        ----------
        emb : tensor (B, T, d_input)
            Perch embeddings.
        mask : tensor (B, T)
            Boolean mask. True = masked (to predict).
        site_ids : tensor (B,)
        hours : tensor (B,)

        Returns
        -------
        pred : tensor (B, T, d_input)
            Reconstructed embeddings for ALL windows.
        """
        # Zero out masked windows before encoding
        masked_emb = emb.clone()
        masked_emb[mask] = 0.0

        h = self.encode(masked_emb, site_ids=site_ids, hours=hours)
        pred = self.reconstruct_head(h)
        return pred


def random_window_mask(batch_size, n_windows=12, mask_ratio=0.25, min_mask=1, max_mask=4):
    """Generate random boolean masks. True = masked."""
    mask = torch.zeros(batch_size, n_windows, dtype=torch.bool)
    n_mask = max(min_mask, min(max_mask, int(n_windows * mask_ratio)))
    for i in range(batch_size):
        indices = torch.randperm(n_windows)[:n_mask]
        mask[i, indices] = True
    return mask


def ssl_pretrain(
    embeddings_files, site_ids, hours,
    ssm_cfg, n_epochs=30, lr=1e-3, batch_size=64,
    mask_ratio=0.25, verbose=True,
):
    """Pretrain the SSM encoder via masked window prediction.

    Parameters
    ----------
    embeddings_files : ndarray (n_files, 12, 1536)
    site_ids : ndarray (n_files,)
    hours : ndarray (n_files,)
    ssm_cfg : dict
        Architecture config (d_model, d_state, n_ssm_layers, etc.)
    n_epochs : int
    lr : float
    batch_size : int
    mask_ratio : float
        Fraction of windows to mask per file.

    Returns
    -------
    model : MaskedSSMEncoder (trained)
    history : dict with 'train_loss' per epoch
    """
    n_files = len(embeddings_files)
    d_input = embeddings_files.shape[2]

    model = MaskedSSMEncoder(
        d_input=d_input,
        d_model=ssm_cfg.get("d_model", 128),
        d_state=ssm_cfg.get("d_state", 16),
        n_ssm_layers=ssm_cfg.get("n_ssm_layers", 2),
        n_windows=N_WINDOWS,
        dropout=ssm_cfg.get("dropout", 0.15),
        n_sites=ssm_cfg.get("n_sites", 20),
        meta_dim=ssm_cfg.get("meta_dim", 16),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    emb_t = torch.tensor(embeddings_files, dtype=torch.float32)
    site_t = torch.tensor(site_ids, dtype=torch.long)
    hour_t = torch.tensor(hours, dtype=torch.long)

    history = {"train_loss": []}

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_files)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_files, batch_size):
            idx = perm[start:start + batch_size]
            emb_batch = emb_t[idx]
            site_batch = site_t[idx]
            hour_batch = hour_t[idx]

            mask = random_window_mask(len(idx), N_WINDOWS, mask_ratio)
            pred = model(emb_batch, mask, site_ids=site_batch, hours=hour_batch)

            # Loss only on masked windows
            loss = F.mse_loss(pred[mask], emb_batch[mask])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  SSL epoch {epoch+1}/{n_epochs}: loss={avg_loss:.6f} lr={scheduler.get_last_lr()[0]:.6f}")

    return model, history


def transfer_weights_to_proto_ssm(ssl_model, proto_ssm_model):
    """Transfer pretrained encoder weights from SSL model to ProtoSSMv2.

    Copies: input_proj, pos_enc, site_emb, hour_emb, meta_proj,
            all SSM layers (fwd, bwd, merge, norm, drop).
    Does NOT copy: reconstruct_head, output heads, prototypes, fusion.
    """
    transferred = 0
    skipped = 0

    ssl_state = ssl_model.state_dict() if hasattr(ssl_model, 'state_dict') else ssl_model
    proto_state = proto_ssm_model.state_dict()

    for key in ssl_state:
        # Skip reconstruction head and metadata embeddings
        if "reconstruct_head" in key or "site_emb" in key or "hour_emb" in key or "meta_proj" in key:
            skipped += 1
            continue

        # Map SSL key names to ProtoSSM key names
        proto_key = key
        # SSL: ssm_layers_fwd.0 -> ProtoSSM: ssm_fwd.0
        proto_key = proto_key.replace("ssm_layers_fwd.", "ssm_fwd.")
        proto_key = proto_key.replace("ssm_layers_bwd.", "ssm_bwd.")
        # SSL: ssm_merges/ssm_norms/ssm_drops (plural) -> ProtoSSM: ssm_merge/ssm_norm/ssm_drop
        proto_key = proto_key.replace("ssm_merges.", "ssm_merge.")
        proto_key = proto_key.replace("ssm_norms.", "ssm_norm.")
        proto_key = proto_key.replace("ssm_drops.", "ssm_drop.")

        if proto_key in proto_state and ssl_state[key].shape == proto_state[proto_key].shape:
            proto_state[proto_key] = ssl_state[key].cpu() if hasattr(ssl_state[key], 'cpu') else ssl_state[key]
            transferred += 1
        else:
            skipped += 1

    proto_ssm_model.load_state_dict(proto_state)
    print(f"  SSL transfer: {transferred} params transferred, {skipped} skipped")
    return proto_ssm_model
