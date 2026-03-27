"""Physical constants used throughout the BirdCLEF pipeline."""

import torch

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC  # 160_000
FILE_SAMPLES = 60 * SR            # 1_920_000
N_WINDOWS = 12

DEVICE = torch.device("cpu")
