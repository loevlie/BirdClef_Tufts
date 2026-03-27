# ProtoSSM v4 Architecture

```
Audio (60s, 32kHz)
    │
    ▼
┌──────────────────────────┐
│  Perch v2 (TF, frozen)   │  Google's pre-trained bird classifier
│  12 × 5-sec windows      │  Output: 1536-dim embeddings + logits
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Input Projection         │  Linear(1536 → d_model) + LayerNorm + GELU
│  + Positional Encoding    │  Learnable (1, 12, d_model) position embedding
│  + Metadata Embedding     │  Site embedding + Hour embedding → project to d_model
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Bidirectional SSM × N    │  Mamba-style selective state space model
│  (SelectiveSSM)           │  Forward + Backward scan → merge via Linear(2d → d)
│                           │  Input-dependent dt, B, C for selective filtering
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Cross-Attention          │  Multi-head self-attention (4 heads)
│  (optional)               │  Captures non-local patterns (dawn chorus, counter-singing)
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Gated Fusion             │  Learnable per-class alpha blends SSM output with Perch logits
│  α · SSM_logits + (1-α) · Perch_logits + class_bias
└──────────┬───────────────┘
           │
           ▼
   Species predictions (B, T, N_CLASSES)
```

## Key Components

### SelectiveSSM (`src/models/ssm.py`)
Mamba-style SSM with input-dependent discretization. For T=12 windows, the sequential scan is efficient on CPU.

### ProtoSSMv2 (`src/models/proto_ssm.py`)
Main model. d_model=128, 2 SSM layers, ~723K parameters.

### ResidualSSM (`src/models/residual_ssm.py`)
Optional second-pass correction. Takes first-pass scores + embeddings, predicts additive corrections.
