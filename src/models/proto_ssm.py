import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm import SelectiveSSM
from .attention import TemporalCrossAttention


class ProtoSSMv2(nn.Module):
    # Prototypical State Space Model v4 with cross-attention and metadata awareness.
    #
    # V16 additions:
    # - Cross-attention layer after SSM for non-local temporal patterns
    # - All other v2 features preserved (metadata, prototypes, gated fusion)

    def __init__(self, d_input=1536, d_model=192, d_state=16,
                 n_ssm_layers=2, n_classes=234, n_windows=12,
                 dropout=0.2, n_sites=20, meta_dim=16,
                 use_cross_attn=True, cross_attn_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_windows = n_windows

        # 1. Feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Learnable positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        # 3. Metadata embeddings
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        # 4. Bidirectional SSM layers
        self.ssm_fwd = nn.ModuleList()
        self.ssm_bwd = nn.ModuleList()
        self.ssm_merge = nn.ModuleList()
        self.ssm_norm = nn.ModuleList()
        for _ in range(n_ssm_layers):
            self.ssm_fwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_bwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_merge.append(nn.Linear(2 * d_model, d_model))
            self.ssm_norm.append(nn.LayerNorm(d_model))
        self.ssm_drop = nn.Dropout(dropout)

        # 4b. NEW: Cross-attention after SSM
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = TemporalCrossAttention(d_model, n_heads=cross_attn_heads, dropout=dropout)

        # 5. Learnable class prototypes
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))

        # 6. Per-class calibration bias
        self.class_bias = nn.Parameter(torch.zeros(n_classes))

        # 7. Per-class gated fusion with Perch logits
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

        # 8. Taxonomic auxiliary head
        self.n_families = 0
        self.family_head = None

    def init_prototypes_from_data(self, embeddings, labels):
        with torch.no_grad():
            h = self.input_proj(embeddings)
            for c in range(self.n_classes):
                mask = labels[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)

    def init_family_head(self, n_families, class_to_family):
        self.n_families = n_families
        self.family_head = nn.Linear(self.d_model, n_families)
        self.register_buffer('class_to_family', torch.tensor(class_to_family, dtype=torch.long))

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape

        # Project embeddings
        h = self.input_proj(emb)
        h = h + self.pos_enc[:, :T, :]

        # Add metadata embeddings
        if site_ids is not None and hours is not None:
            s_emb = self.site_emb(site_ids)
            h_emb = self.hour_emb(hours)
            meta = self.meta_proj(torch.cat([s_emb, h_emb], dim=-1))
            h = h + meta[:, None, :]

        # Bidirectional SSM
        for fwd, bwd, merge, norm in zip(
            self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm
        ):
            residual = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h = merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = norm(h + residual)

        # NEW: Cross-attention for non-local temporal patterns
        if self.use_cross_attn:
            h = self.cross_attn(h)

        h_temporal = h

        # Prototypical cosine similarity + class bias
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        temp = F.softplus(self.proto_temp)
        sim = torch.matmul(h_norm, p_norm.T) * temp + self.class_bias[None, None, :]

        # Gated fusion with Perch logits
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            species_logits = alpha * sim + (1 - alpha) * perch_logits
        else:
            species_logits = sim

        # Taxonomic auxiliary prediction
        family_logits = None
        if self.family_head is not None:
            h_pool = h.mean(dim=1)
            family_logits = self.family_head(h_pool)

        return species_logits, family_logits, h_temporal

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
