"""Train EfficientNet-B0 on mel-spectrograms from train_audio.

Run on Kaggle GPU. Attach competition data as input.
Outputs cnn_efficientnet_b0.pt to /kaggle/working/

At submission time: load weights, generate spectrograms from test audio,
get 234-class predictions, ensemble with Perch pipeline.
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Config ──
SR = 32000
DURATION = 5  # seconds
N_SAMPLES = SR * DURATION  # 160000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
N_CLASSES = 234

BATCH_SIZE = 64
N_EPOCHS = 15
LR = 1e-3
MIXUP_ALPHA = 0.3

BASE = Path("/kaggle/input/competitions/birdclef-2026")
TRAIN_AUDIO = BASE / "train_audio"

# ── Load metadata ──
train_csv = pd.read_csv(BASE / "train.csv")
sample_sub = pd.read_csv(BASE / "sample_submission.csv")
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
label_to_idx = {l: i for i, l in enumerate(PRIMARY_LABELS)}

# Filter to species we need to predict
train_csv = train_csv[train_csv["primary_label"].isin(label_to_idx)].reset_index(drop=True)
print(f"Training samples: {len(train_csv)}, Species: {train_csv['primary_label'].nunique()}")

# ── Mel spectrogram on GPU ──
import torchaudio
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
    n_mels=N_MELS, power=2.0,
).to(DEVICE)

def audio_to_melspec(waveform):
    """Convert waveform tensor to log-mel spectrogram on GPU."""
    with torch.no_grad():
        spec = mel_transform(waveform)
        spec = torch.log(spec + 1e-8)
        # Normalize
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)
    return spec


# ── Dataset ──
class BirdAudioDataset(Dataset):
    def __init__(self, df, audio_dir, label_to_idx, n_samples=N_SAMPLES, is_train=True):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.label_to_idx = label_to_idx
        self.n_samples = n_samples
        self.is_train = is_train
        self.n_classes = len(label_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.audio_dir / row["filename"]

        try:
            y, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1)

            # Random crop (train) or center crop (val)
            if len(y) > self.n_samples:
                if self.is_train:
                    start = np.random.randint(0, len(y) - self.n_samples)
                else:
                    start = (len(y) - self.n_samples) // 2
                y = y[start:start + self.n_samples]
            else:
                y = np.pad(y, (0, max(0, self.n_samples - len(y))))
                y = y[:self.n_samples]

            # Train augmentation: random gain
            if self.is_train:
                gain = np.random.uniform(0.8, 1.2)
                y = y * gain

        except Exception:
            y = np.zeros(self.n_samples, dtype=np.float32)

        label = np.zeros(self.n_classes, dtype=np.float32)
        if row["primary_label"] in self.label_to_idx:
            label[self.label_to_idx[row["primary_label"]]] = 1.0

        return torch.tensor(y, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# ── Model ──
class BirdCNNClassifier(nn.Module):
    """Lightweight CNN: mel-spectrogram → EfficientNet-B0 backbone → 234 classes."""

    def __init__(self, n_classes=234, pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained,
                                          in_chans=1, num_classes=0, global_pool="avg")
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, n_classes),
        )

    def forward(self, mel):
        """mel: (B, 1, N_MELS, T)"""
        features = self.backbone(mel)  # (B, 1280)
        return self.head(features)     # (B, N_CLASSES)

    def get_embeddings(self, mel):
        """Get 1280-dim embeddings (before classification head)."""
        return self.backbone(mel)


# ── Training ──
def mixup(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


def train_epoch(model, loader, optimizer, mel_transform):
    model.train()
    total_loss = 0
    n_batches = 0

    for waveforms, labels in loader:
        waveforms = waveforms.to(DEVICE)  # (B, N_SAMPLES)
        labels = labels.to(DEVICE)

        # Generate mel spectrograms on GPU
        with torch.no_grad():
            mels = mel_transform(waveforms)  # (B, N_MELS, T)
            mels = torch.log(mels + 1e-8)
            mels = (mels - mels.mean(dim=(1, 2), keepdim=True)) / (mels.std(dim=(1, 2), keepdim=True) + 1e-8)
            mels = mels.unsqueeze(1)  # (B, 1, N_MELS, T)

        # Mixup
        if MIXUP_ALPHA > 0:
            mels, labels = mixup(mels, labels, MIXUP_ALPHA)

        logits = model(mels)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, mel_transform):
    model.eval()
    total_loss = 0
    n_batches = 0

    for waveforms, labels in loader:
        waveforms = waveforms.to(DEVICE)
        labels = labels.to(DEVICE)

        mels = mel_transform(waveforms)
        mels = torch.log(mels + 1e-8)
        mels = (mels - mels.mean(dim=(1, 2), keepdim=True)) / (mels.std(dim=(1, 2), keepdim=True) + 1e-8)
        mels = mels.unsqueeze(1)

        logits = model(mels)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ── Main ──
t_start = time.time()

# Train/val split (stratified by species)
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train_csv, test_size=0.1, stratify=train_csv["primary_label"], random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}")

train_ds = BirdAudioDataset(train_df, TRAIN_AUDIO, label_to_idx, is_train=True)
val_ds = BirdAudioDataset(val_df, TRAIN_AUDIO, label_to_idx, is_train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

model = BirdCNNClassifier(n_classes=N_CLASSES, pretrained=True).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

best_val_loss = float("inf")
best_state = None

for epoch in range(N_EPOCHS):
    t0 = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, mel_transform)
    val_loss = validate(model, val_loader, mel_transform)
    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    print(f"  Epoch {epoch+1}/{N_EPOCHS}: train={train_loss:.4f} val={val_loss:.4f} best={best_val_loss:.4f} [{elapsed:.0f}s]", flush=True)

# Save best model
model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
save_path = "/kaggle/working/cnn_efficientnet_b0.pt"
torch.save(best_state, save_path)

total_time = time.time() - t_start
print(f"\nTraining complete in {total_time/60:.1f} min")
print(f"Best val loss: {best_val_loss:.4f}")
print(f"Saved: {save_path}")
print(f"\nTo use: add this notebook's output as input to your submission notebook")
