"""SSL pretraining on Kaggle GPU — masked window prediction.

Run as a Kaggle GPU notebook. Attach:
  - Dataset: dennyloevlie/birdclef2026-ssl-embeddings

Outputs ssl_encoder.pt to /kaggle/working/ for download.
Tries multiple SSL approaches and saves the best.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

N_WINDOWS = 12

# ── Load embeddings ──────────────────────────────────────────────────────
t0 = time.time()

# Find the dataset
import glob
emb_path = None
for candidate in [
    "/kaggle/input/birdclef2026-ssl-embeddings/all_soundscapes_emb.npz",
    *glob.glob("/kaggle/input/*/all_soundscapes_emb.npz"),
    *glob.glob("/kaggle/input/datasets/*/birdclef2026-ssl-embeddings/all_soundscapes_emb.npz"),
]:
    if Path(candidate).exists():
        emb_path = candidate
        break

assert emb_path is not None, "Embeddings not found! Attach dennyloevlie/birdclef2026-ssl-embeddings"
print(f"Loading from {emb_path}...")

arr = np.load(emb_path)
embeddings = arr["embeddings"]
n_files = len(embeddings) // N_WINDOWS
emb_files = torch.tensor(embeddings.reshape(n_files, N_WINDOWS, -1), dtype=torch.float32, device=DEVICE)
D_INPUT = emb_files.shape[2]
print(f"Loaded: {n_files} files, {emb_files.shape}, {time.time()-t0:.1f}s")

# ── SSM Building Blocks ──────────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B_size, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)
        h = torch.zeros(B_size, D, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dA = torch.exp(A[None, :, :] * dt[:, t, :, None])
            dB = dt[:, t, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            y_t = (h * C[:, t, None, :]).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class MaskedSSMEncoder(nn.Module):
    def __init__(self, d_input=1536, d_model=128, d_state=16, n_ssm_layers=2,
                 n_windows=12, dropout=0.15):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
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
        self.reconstruct_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_input),
        )

    def encode(self, emb):
        h = self.input_proj(emb) + self.pos_enc[:, :emb.shape[1], :]
        for fwd, bwd, merge, norm, drop in zip(
            self.ssm_layers_fwd, self.ssm_layers_bwd,
            self.ssm_merges, self.ssm_norms, self.ssm_drops
        ):
            residual = h
            h = norm(merge(torch.cat([fwd(h), bwd(h.flip(1)).flip(1)], dim=-1)) + residual)
            h = drop(h)
        return h

    def forward(self, emb, mask):
        masked_emb = emb.clone()
        masked_emb[mask] = 0.0
        return self.reconstruct_head(self.encode(masked_emb))


# ── Training function ────────────────────────────────────────────────────

def train_ssl(emb_files, d_model=128, d_state=16, n_ssm_layers=2,
              n_epochs=50, lr=1e-3, batch_size=256, mask_ratio=0.25,
              dropout=0.15, label=""):
    n_files = len(emb_files)
    model = MaskedSSMEncoder(
        d_input=D_INPUT, d_model=d_model, d_state=d_state,
        n_ssm_layers=n_ssm_layers, dropout=dropout,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*50}")
    print(f"  SSL Config: {label}")
    print(f"  d_model={d_model}, layers={n_ssm_layers}, params={n_params:,}")
    print(f"  epochs={n_epochs}, lr={lr}, batch={batch_size}, mask={mask_ratio}")
    print(f"{'='*50}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    n_mask = max(1, min(4, int(N_WINDOWS * mask_ratio)))

    best_loss = float("inf")
    best_state = None
    losses = []

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_files, device=DEVICE)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_files, batch_size):
            idx = perm[start:start + batch_size]
            batch = emb_files[idx]

            # Random mask
            mask = torch.zeros(len(idx), N_WINDOWS, dtype=torch.bool, device=DEVICE)
            for i in range(len(idx)):
                mask[i, torch.randperm(N_WINDOWS, device=DEVICE)[:n_mask]] = True

            pred = model(batch, mask)
            loss = F.mse_loss(pred[mask], batch[mask])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.6f} best={best_loss:.6f}")

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model, best_loss, losses


# ── Run multiple SSL configs ─────────────────────────────────────────────

results = []
t_start = time.time()

# Config 1: Match the large v18 ProtoSSM (d_model=320, 4 layers)
m1, loss1, h1 = train_ssl(emb_files, d_model=320, d_state=32, n_ssm_layers=4,
                           n_epochs=50, lr=1e-3, batch_size=256, mask_ratio=0.25,
                           dropout=0.12, label="Large (320, 4L)")
results.append(("ssl_encoder_large.pt", m1, loss1))

# Config 2: Match the small ProtoSSM (d_model=128, 2 layers)
m2, loss2, h2 = train_ssl(emb_files, d_model=128, d_state=16, n_ssm_layers=2,
                           n_epochs=50, lr=1e-3, batch_size=512, mask_ratio=0.25,
                           dropout=0.15, label="Small (128, 2L)")
results.append(("ssl_encoder_small.pt", m2, loss2))

# Config 3: Large with higher mask ratio (harder task)
m3, loss3, h3 = train_ssl(emb_files, d_model=320, d_state=32, n_ssm_layers=4,
                           n_epochs=50, lr=5e-4, batch_size=256, mask_ratio=0.4,
                           dropout=0.12, label="Large hard mask (320, 4L, mask=0.4)")
results.append(("ssl_encoder_large_hard.pt", m3, loss3))

# Config 4: Large with more epochs
m4, loss4, h4 = train_ssl(emb_files, d_model=320, d_state=32, n_ssm_layers=4,
                           n_epochs=100, lr=1e-3, batch_size=256, mask_ratio=0.25,
                           dropout=0.12, label="Large long (320, 4L, 100ep)")
results.append(("ssl_encoder_large_long.pt", m4, loss4))

# ── Save all models ──────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"  SSL TRAINING COMPLETE ({(time.time()-t_start)/60:.1f} min)")
print(f"{'='*50}")

for name, model, loss in results:
    save_path = f"/kaggle/working/{name}"
    torch.save(model.state_dict(), save_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {name}: loss={loss:.6f}, params={n_params:,}")

print(f"\nDownload from /kaggle/working/ and use with:")
print(f"  uv run python scripts/train.py --ssl-weights <path> --data-dir data/competition")
