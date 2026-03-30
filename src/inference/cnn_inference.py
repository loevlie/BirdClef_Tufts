"""Lightweight CNN inference on mel-spectrograms for submission."""

import gc
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

SR = 32000
DURATION = 5
N_SAMPLES = SR * DURATION
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512


class BirdCNNClassifier(nn.Module):
    """Must match the training architecture exactly."""

    def __init__(self, n_classes=234):
        super().__init__()
        import timm
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False,
                                          in_chans=1, num_classes=0, global_pool="avg")
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, n_classes),
        )

    def forward(self, mel):
        return self.head(self.backbone(mel))


def load_cnn_model(weights_path, n_classes=234, device="cpu"):
    """Load trained CNN model."""
    model = BirdCNNClassifier(n_classes=n_classes)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def infer_cnn_on_soundscapes(paths, model, n_classes=234, n_windows=12,
                              batch_size=8, device="cpu", verbose=True):
    """Run CNN inference on 60-second soundscapes.

    Splits each file into 12 x 5-second windows, generates mel-spectrograms,
    and runs the CNN classifier.

    Returns
    -------
    cnn_scores : ndarray (n_files * n_windows, n_classes)
    """
    import torchaudio
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, power=2.0,
    ).to(device)

    n_files = len(paths)
    cnn_scores = np.zeros((n_files * n_windows, n_classes), dtype=np.float32)

    iterator = range(0, n_files, batch_size)
    if verbose:
        iterator = tqdm(iterator, total=(n_files + batch_size - 1) // batch_size, desc="CNN batches")

    for start in iterator:
        batch_paths = paths[start:start + batch_size]
        waveforms = []

        for path in batch_paths:
            try:
                y, sr = sf.read(str(path), dtype="float32", always_2d=False)
                if y.ndim == 2:
                    y = y.mean(axis=1)
                if len(y) < SR * 60:
                    y = np.pad(y, (0, SR * 60 - len(y)))
                y = y[:SR * 60]
                # Split into 12 windows
                windows = y.reshape(n_windows, N_SAMPLES)
                waveforms.append(windows)
            except Exception:
                waveforms.append(np.zeros((n_windows, N_SAMPLES), dtype=np.float32))

        # Stack: (batch_files * 12, N_SAMPLES)
        batch_waves = np.concatenate(waveforms, axis=0)
        batch_tensor = torch.tensor(batch_waves, dtype=torch.float32).to(device)

        with torch.no_grad():
            mels = mel_transform(batch_tensor)
            mels = torch.log(mels + 1e-8)
            mels = (mels - mels.mean(dim=(1, 2), keepdim=True)) / (mels.std(dim=(1, 2), keepdim=True) + 1e-8)
            mels = mels.unsqueeze(1)  # (B, 1, N_MELS, T)

            logits = model(mels)
            probs = torch.sigmoid(logits).cpu().numpy()

        row_start = start * n_windows
        row_end = row_start + len(batch_waves)
        cnn_scores[row_start:row_end] = probs

        del batch_tensor, mels, logits
        gc.collect()

    return cnn_scores
