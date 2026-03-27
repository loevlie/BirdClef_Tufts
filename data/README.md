# Data Directory

## Download

Join the [BirdCLEF 2026 competition](https://www.kaggle.com/competitions/birdclef-2026), then:

```bash
# Option 1: Kaggle CLI
kaggle competitions download -c birdclef-2026
unzip birdclef-2026.zip -d data/competition

# Option 2: kagglehub
python -c "import kagglehub; kagglehub.competition_download('birdclef-2026')"

# Option 3: MCP tool (if using an AI agent)
# Use the "mcp_kaggle_download_competition_data_files" tool
```

## Expected Structure

```
data/competition/
  taxonomy.csv
  sample_submission.csv
  train_soundscapes_labels.csv
  train_soundscapes/        # ~15GB of .ogg files
  test_soundscapes/         # empty locally, populated on Kaggle
```

## Cache

Pre-computed Perch embeddings go in `cache/`:
```
cache/
  full_perch_meta.parquet
  full_perch_arrays.npz
  full_oof_meta_features.npz
```

Generated automatically by `make train` if not present (requires TensorFlow + Perch model).
