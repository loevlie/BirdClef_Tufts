import torch.nn as nn


class TemporalCrossAttention(nn.Module):
    """Multi-head cross-attention between temporal windows.
    Captures non-local patterns (e.g., dawn chorus onset, counter-singing)
    that sequential SSM may miss."""

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, D)
        residual = x
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x
