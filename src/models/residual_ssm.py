import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm import SelectiveSSM


class ResidualSSM(nn.Module):
    # Lightweight SSM that takes first-pass scores + embeddings and predicts corrections.
    # Architecture: project(concat(emb, first_pass)) -> 1-layer BiSSM -> linear head

    def __init__(self, d_input=1536, d_scores=234, d_model=64, d_state=8,
                 n_classes=234, n_windows=12, dropout=0.1, n_sites=20, meta_dim=8):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes

        # Project embeddings + first-pass scores
        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Metadata
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        # Single bidirectional SSM layer (lightweight)
        self.ssm_fwd = SelectiveSSM(d_model, d_state)
        self.ssm_bwd = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm_drop = nn.Dropout(dropout)

        # Output: per-class correction (additive)
        self.output_head = nn.Linear(d_model, n_classes)

        # Initialize output near zero (corrections start small)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, emb, first_pass_scores, site_ids=None, hours=None):
        # emb: (B, T, d_input), first_pass_scores: (B, T, n_classes)
        B, T, _ = emb.shape

        # Concatenate embeddings with first-pass scores
        x = torch.cat([emb, first_pass_scores], dim=-1)  # (B, T, d_input + d_scores)
        h = self.input_proj(x)

        # Add metadata
        if site_ids is not None and hours is not None:
            site_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
            hour_e = self.hour_emb(hours.clamp(0, 23))
            meta = self.meta_proj(torch.cat([site_e, hour_e], dim=-1))
            h = h + meta.unsqueeze(1)

        h = h + self.pos_enc[:, :T, :]

        # Bidirectional SSM
        residual = h
        h_f = self.ssm_fwd(h)
        h_b = self.ssm_bwd(h.flip(1)).flip(1)
        h = self.ssm_merge(torch.cat([h_f, h_b], dim=-1))
        h = self.ssm_drop(h)
        h = self.ssm_norm(h + residual)

        # Output correction
        correction = self.output_head(h)  # (B, T, n_classes)
        return correction

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
