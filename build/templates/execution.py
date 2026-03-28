# === EXECUTION PIPELINE ===
# This runs the full submit-mode pipeline on Kaggle.

import time
_WALL_START = time.time()

# --- Paths ---
from pathlib import Path
import glob
BASE = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")
CACHE_WORK_DIR = Path("/kaggle/working/perch_cache")
CACHE_WORK_DIR.mkdir(parents=True, exist_ok=True)

# --- Verify inputs ---
print("=" * 50)
print("  INPUT VERIFICATION")
print("=" * 50)

# Auto-discover Perch cache
PIPELINE_INPUT = Path("/kaggle/input/birdclef2026-pipeline-inputs")  # combined dataset
CACHE_INPUT_DIR = None
for candidate in [
    PIPELINE_INPUT,
    Path("/kaggle/input/birdclef2026-perch-cache"),
    Path("/kaggle/input/perch-meta"),
    *[Path(p).parent for p in glob.glob("/kaggle/input/*/full_perch_arrays.npz")],
    *[Path(p).parent for p in glob.glob("/kaggle/input/*/perch_cache/full_perch_arrays.npz")],
    *[Path(p).parent for p in glob.glob("/kaggle/input/*/*/perch_cache/full_perch_arrays.npz")],
]:
    if (candidate / "full_perch_arrays.npz").exists() and (candidate / "full_perch_meta.parquet").exists():
        CACHE_INPUT_DIR = candidate
        break

# Find labels.csv (from pipeline inputs or Perch model)
LABELS_CSV = None
for candidate in [
    PIPELINE_INPUT / "labels.csv",
    MODEL_DIR / "assets" / "labels.csv",
    *[Path(p) for p in glob.glob("/kaggle/input/*/labels.csv")],
]:
    if candidate.exists():
        LABELS_CSV = str(candidate)
        break

checks = {
    "Competition data": BASE.exists(),
    "Perch labels.csv": LABELS_CSV is not None,
    "Perch cache": CACHE_INPUT_DIR is not None,
    "ONNX model": ONNX_PATH is not None,
}
for name, ok in checks.items():
    status = "FOUND" if ok else "MISSING"
    print(f"  {name}: {status}")
if CACHE_INPUT_DIR:
    print(f"    -> {CACHE_INPUT_DIR}")
else:
    CACHE_INPUT_DIR = Path("/kaggle/input/perch-meta")
    print("    -> Will compute from scratch (slower)")
print("=" * 50)

# --- Load ONNX for test inference (3x faster than TF) ---
# Auto-discover ONNX model
ONNX_PATH = None
for candidate in [
    str(PIPELINE_INPUT / "perch_v2.onnx"),
    *glob.glob("/kaggle/input/*/perch_v2.onnx"),
    *glob.glob("/kaggle/input/*/*/perch_v2.onnx"),
]:
    if Path(candidate).exists():
        ONNX_PATH = candidate
        break

USE_ONNX = False
onnx_session = None
infer_fn = None

if ONNX_PATH:
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        onnx_session = ort.InferenceSession(ONNX_PATH, opts, providers=["CPUExecutionProvider"])
        USE_ONNX = True
        print(f"Using ONNX Perch ({ONNX_PATH})")
    except Exception as e:
        print(f"ONNX failed ({e}), falling back to TF")

if not USE_ONNX:
    # TF only needed if no ONNX available
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        birdclassifier = tf.saved_model.load(str(MODEL_DIR))
        infer_fn = birdclassifier.signatures["serving_default"]
        print("Using TensorFlow Perch")
    except Exception as e:
        print(f"TF Perch not available ({e}). Cache required for training data.")
        infer_fn = None

timer = WallTimer(budget_seconds=CFG.get("timer", {}).get("budget_seconds", 5400.0))

# --- Load data ---
timer.stage_start("load_data")
taxonomy, sample_sub, soundscape_labels, PRIMARY_LABELS, N_CLASSES = (
    load_competition_data(str(BASE))
)
sc_clean, Y_SC, full_files, full_truth, Y_FULL_TRUTH, label_to_idx = (
    prepare_labels(soundscape_labels, PRIMARY_LABELS)
)
print(f"Classes: {N_CLASSES}, Full files: {len(full_files)}")
timer.stage_end()

# --- Perch mapping ---
timer.stage_start("perch_mapping")
mapping = build_perch_mapping(taxonomy, str(MODEL_DIR), PRIMARY_LABELS, Y_SC, label_to_idx,
                             labels_csv_path=LABELS_CSV)
timer.stage_end()

# --- Load or compute training cache ---
timer.stage_start("cache")
meta_full, scores_full_raw, emb_full = load_or_compute_cache(
    full_files, str(BASE), str(CACHE_WORK_DIR), str(CACHE_INPUT_DIR),
    mapping, CFG.get("pipeline", {}), infer_fn,
)
Y_FULL = align_truth_to_cache(full_truth, Y_SC, meta_full)
print(f"Training data: {meta_full.shape[0]} windows, emb: {emb_full.shape}")
timer.stage_end()

# --- OOF meta-features ---
timer.stage_start("oof_meta")
fuse_kwargs = mapping["fuse_kwargs"]
fusion_cfg = CFG.get("best_fusion", CFG.get("fusion", {}))
fuse_kwargs.update({
    "lambda_event": fusion_cfg.get("lambda_event", 0.4),
    "lambda_texture": fusion_cfg.get("lambda_texture", 1.0),
    "lambda_proxy_texture": fusion_cfg.get("lambda_proxy_texture", 0.8),
    "smooth_texture": fusion_cfg.get("smooth_texture", 0.35),
    "smooth_event": fusion_cfg.get("smooth_event", 0.15),
})
oof_base, oof_prior, oof_fold_id = load_or_compute_oof_meta(
    scores_full_raw, meta_full, sc_clean, Y_SC,
    str(CACHE_WORK_DIR), fuse_kwargs, CFG.get("pipeline", {}),
)
baseline_oof_auc = macro_auc_skip_empty(Y_FULL, oof_base)
print(f"OOF baseline AUC: {baseline_oof_auc:.6f}")
timer.stage_end()

# --- Prior tables + PCA ---
timer.stage_start("priors_pca")
final_prior_tables = fit_prior_tables(sc_clean.reset_index(drop=True), Y_SC)
emb_scaler = StandardScaler()
emb_full_scaled = emb_scaler.fit_transform(emb_full)
probe_cfg = CFG.get("probe", CFG.get("frozen_best_probe", {}))
pca_dim = min(int(probe_cfg.get("pca_dim", 64)), emb_full_scaled.shape[0] - 1, emb_full_scaled.shape[1])
emb_pca = PCA(n_components=pca_dim)
Z_FULL = emb_pca.fit_transform(emb_full_scaled).astype(np.float32)
timer.stage_end()

# --- Reshape to file-level ---
emb_files, file_list = reshape_to_files(emb_full, meta_full)
logits_files, _ = reshape_to_files(scores_full_raw, meta_full)
labels_files, _ = reshape_to_files(Y_FULL, meta_full)

# --- Build metadata ---
ssm_cfg = CFG.get("proto_ssm", {})
n_families, class_to_family, fam_to_idx = build_taxonomy_groups(taxonomy, PRIMARY_LABELS)
site_to_idx, n_sites_mapped = build_site_mapping(meta_full)
n_sites_cfg = ssm_cfg.get("n_sites", 20)
site_ids_all, hours_all = get_file_metadata(meta_full, file_list, site_to_idx, n_sites_cfg)
file_families = np.zeros((len(file_list), n_families), dtype=np.float32)
for fi in range(len(file_list)):
    active = np.where(labels_files[fi].sum(axis=0) > 0)[0]
    for ci in active:
        file_families[fi, class_to_family[ci]] = 1.0

# --- Train ProtoSSM ---
timer.stage_start("proto_ssm_train")
model = ProtoSSMv2(
    d_input=emb_full.shape[1], d_model=ssm_cfg.get("d_model", 128),
    d_state=ssm_cfg.get("d_state", 16), n_ssm_layers=ssm_cfg.get("n_ssm_layers", 2),
    n_classes=N_CLASSES, n_windows=N_WINDOWS, dropout=ssm_cfg.get("dropout", 0.15),
    n_sites=n_sites_cfg, meta_dim=ssm_cfg.get("meta_dim", 16),
    use_cross_attn=ssm_cfg.get("use_cross_attn", True),
    cross_attn_heads=ssm_cfg.get("cross_attn_heads", 4),
)
model.init_prototypes_from_data(
    torch.tensor(emb_full, dtype=torch.float32),
    torch.tensor(Y_FULL, dtype=torch.float32),
)
model.init_family_head(n_families, class_to_family)
train_cfg = CFG.get("proto_ssm_train", {})
model, train_history = train_proto_ssm_single(
    model, emb_files, logits_files, labels_files.astype(np.float32),
    site_ids_train=site_ids_all, hours_train=hours_all,
    file_families_train=file_families, cfg=train_cfg, verbose=True,
)
timer.stage_end()

# --- Train MLP probes ---
timer.stage_start("mlp_probes")
min_pos = int(probe_cfg.get("min_pos", 8))
alpha_probe = float(probe_cfg.get("alpha", 0.40))
mlp_sub = probe_cfg.get("mlp", {})
mlp_params = {
    "hidden_layer_sizes": tuple(mlp_sub.get("hidden_layer_sizes", [128])),
    "activation": mlp_sub.get("activation", "relu"),
    "max_iter": mlp_sub.get("max_iter", 100),
    "early_stopping": mlp_sub.get("early_stopping", True),
    "validation_fraction": mlp_sub.get("validation_fraction", 0.15),
    "n_iter_no_change": mlp_sub.get("n_iter_no_change", 10),
    "random_state": 42,
    "learning_rate_init": mlp_sub.get("learning_rate_init", 0.001),
    "alpha": mlp_sub.get("l2_alpha", 0.01),
}
PROBE_CLASS_IDX = np.where(Y_FULL.sum(axis=0) >= min_pos)[0].astype(np.int32)
probe_models = {}
for cls_idx in PROBE_CLASS_IDX:
    y = Y_FULL[:, cls_idx]
    if y.sum() == 0 or y.sum() == len(y):
        continue
    X_cls = build_class_features(Z_FULL, scores_full_raw[:, cls_idx], oof_prior[:, cls_idx], oof_base[:, cls_idx])
    n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
    if n_pos > 0 and n_neg > n_pos:
        repeat = max(1, n_neg // n_pos)
        pos_idx = np.where(y == 1)[0]
        X_bal = np.vstack([X_cls, np.tile(X_cls[pos_idx], (repeat, 1))])
        y_bal = np.concatenate([y, np.ones(len(pos_idx) * repeat, dtype=y.dtype)])
    else:
        X_bal, y_bal = X_cls, y
    clf = MLPClassifier(**mlp_params)
    clf.fit(X_bal, y_bal)
    probe_models[cls_idx] = clf
print(f"MLP probes: {len(probe_models)}")
timer.stage_end()

# --- Test inference ---
timer.stage_start("test_inference")
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
if len(test_paths) == 0:
    print("No test files. Dry-run on train soundscapes.")
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:CFG.get("pipeline", {}).get("dryrun_n_files", 20)]

if USE_ONNX:
    meta_test, scores_test_raw, emb_test = infer_perch_onnx(
        test_paths, session=onnx_session, n_classes=N_CLASSES,
        mapped_pos=mapping["MAPPED_POS"], mapped_bc_indices=mapping["MAPPED_BC_INDICES"],
        proxy_pos_to_bc=mapping.get("selected_proxy_pos_to_bc"),
        batch_files=CFG.get("pipeline", {}).get("batch_files", 32), verbose=True,
        proxy_reduce=CFG.get("pipeline", {}).get("proxy_reduce", "max"),
    )
else:
    meta_test, scores_test_raw, emb_test = infer_perch_with_embeddings(
        test_paths, infer_fn=infer_fn, n_classes=N_CLASSES,
        mapped_pos=mapping["MAPPED_POS"], mapped_bc_indices=mapping["MAPPED_BC_INDICES"],
        proxy_pos_to_bc=mapping.get("selected_proxy_pos_to_bc"),
        batch_files=CFG.get("pipeline", {}).get("batch_files", 32), verbose=True,
        proxy_reduce=CFG.get("pipeline", {}).get("proxy_reduce", "max"),
    )

# ProtoSSM inference
emb_test_files, test_file_list = reshape_to_files(emb_test, meta_test)
logits_test_files, _ = reshape_to_files(scores_test_raw, meta_test)
test_site_ids, test_hours = get_file_metadata(meta_test, test_file_list, site_to_idx, n_sites_cfg)

model.eval()
with torch.no_grad():
    proto_out, _, _ = model(
        torch.tensor(emb_test_files, dtype=torch.float32),
        torch.tensor(logits_test_files, dtype=torch.float32),
        site_ids=torch.tensor(test_site_ids, dtype=torch.long),
        hours=torch.tensor(test_hours, dtype=torch.long),
    )
proto_scores_flat = proto_out.reshape(-1, N_CLASSES).numpy().astype(np.float32)

# Prior-fused base scores
test_base_scores, test_prior_scores = fuse_scores_with_tables(
    scores_test_raw, sites=meta_test["site"].to_numpy(),
    hours=meta_test["hour_utc"].to_numpy(), tables=final_prior_tables, **fuse_kwargs,
)

# MLP probe scores
emb_test_scaled = emb_scaler.transform(emb_test)
Z_TEST = emb_pca.transform(emb_test_scaled).astype(np.float32)
mlp_scores = test_base_scores.copy()
for cls_idx, clf in probe_models.items():
    X_cls_test = build_class_features(Z_TEST, scores_test_raw[:, cls_idx], test_prior_scores[:, cls_idx], test_base_scores[:, cls_idx])
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X_cls_test)[:, 1].astype(np.float32)
        pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
    else:
        pred = clf.decision_function(X_cls_test).astype(np.float32)
    mlp_scores[:, cls_idx] = (1.0 - alpha_probe) * test_base_scores[:, cls_idx] + alpha_probe * pred

# Ensemble
ENSEMBLE_WEIGHT_PROTO = 0.5
final_test_scores = (ENSEMBLE_WEIGHT_PROTO * proto_scores_flat + (1.0 - ENSEMBLE_WEIGHT_PROTO) * mlp_scores).astype(np.float32)

# Temperature + calibration
CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
class_temperatures = build_class_temperatures(PRIMARY_LABELS, CLASS_NAME_MAP, CFG.get("temperature", {}))
probs = apply_temperature_and_scale(final_test_scores, class_temperatures, n_windows=N_WINDOWS, top_k=CFG.get("file_level_top_k", 2))

# Submission
submission = build_submission(probs, meta_test, PRIMARY_LABELS, test_paths, n_windows=N_WINDOWS)
submission.to_csv("submission.csv", index=False)
timer.stage_end()

print(f"\nSubmission: {submission.shape}")
print(f"Score range: {probs.min():.6f} to {probs.max():.6f}")
timer.print_report()
