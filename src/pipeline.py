"""Shared pipeline setup: data loading, Perch mapping, cache resolution."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import N_WINDOWS
from src.data.parsing import parse_soundscape_labels, parse_soundscape_filename, union_labels
from src.data.taxonomy import TEXTURE_TAXA


# ---------------------------------------------------------------------------
# 1. Load competition data
# ---------------------------------------------------------------------------

def load_competition_data(data_dir):
    """Load taxonomy, sample_submission, and train_soundscapes_labels CSVs.

    Returns
    -------
    taxonomy : pd.DataFrame
    sample_sub : pd.DataFrame
    soundscape_labels : pd.DataFrame
    PRIMARY_LABELS : list[str]
    N_CLASSES : int
    """
    data_dir = Path(data_dir)
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    sample_sub = pd.read_csv(data_dir / "sample_submission.csv")
    soundscape_labels = pd.read_csv(data_dir / "train_soundscapes_labels.csv")

    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    N_CLASSES = len(PRIMARY_LABELS)

    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    soundscape_labels["primary_label"] = soundscape_labels["primary_label"].astype(str)

    return taxonomy, sample_sub, soundscape_labels, PRIMARY_LABELS, N_CLASSES


# ---------------------------------------------------------------------------
# 2. Prepare labels
# ---------------------------------------------------------------------------

def prepare_labels(soundscape_labels, PRIMARY_LABELS):
    """Deduplicate rows, aggregate labels, build multi-hot Y_SC, identify full files.

    Returns
    -------
    sc_clean : pd.DataFrame
    Y_SC : np.ndarray, shape (len(sc_clean), N_CLASSES)
    full_files : list[str]
    full_truth : pd.DataFrame
    Y_FULL_TRUTH : np.ndarray
    label_to_idx : dict[str, int]
    """
    N_CLASSES = len(PRIMARY_LABELS)

    sc_clean = (
        soundscape_labels
        .groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )

    sc_clean["start_sec"] = pd.to_timedelta(sc_clean["start"]).dt.total_seconds().astype(int)
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    sc_clean["row_id"] = (
        sc_clean["filename"].str.replace(".ogg", "", regex=False)
        + "_"
        + sc_clean["end_sec"].astype(str)
    )

    meta = sc_clean["filename"].apply(parse_soundscape_filename).apply(pd.Series)
    sc_clean = pd.concat([sc_clean, meta], axis=1)

    # Fully-labeled files (all 12 windows present)
    windows_per_file = sc_clean.groupby("filename").size()
    full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
    sc_clean["file_fully_labeled"] = sc_clean["filename"].isin(full_files)

    # Multi-hot label matrix aligned with sc_clean row order
    label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
    Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)

    for i, labels in enumerate(sc_clean["label_list"]):
        idxs = [label_to_idx[lbl] for lbl in labels if lbl in label_to_idx]
        if idxs:
            Y_SC[i, idxs] = 1

    full_truth = (
        sc_clean[sc_clean["file_fully_labeled"]]
        .sort_values(["filename", "end_sec"])
        .reset_index(drop=False)
    )

    Y_FULL_TRUTH = Y_SC[full_truth["index"].to_numpy()]

    return sc_clean, Y_SC, full_files, full_truth, Y_FULL_TRUTH, label_to_idx


# ---------------------------------------------------------------------------
# 3. Build Perch mapping
# ---------------------------------------------------------------------------

def build_perch_mapping(taxonomy, model_dir, PRIMARY_LABELS, Y_SC, label_to_idx,
                        labels_csv_path=None):
    """Build species-to-BirdClassifier mapping arrays and index groups.

    Parameters
    ----------
    labels_csv_path : str or Path, optional
        Direct path to Perch labels.csv. If not provided, looks in model_dir/assets/.

    Returns
    -------
    dict with keys: BC_INDICES, MAPPED_MASK, MAPPED_POS, UNMAPPED_POS,
        MAPPED_BC_INDICES, bc_labels, NO_LABEL_INDEX, CLASS_NAME_MAP,
        selected_proxy_pos_to_bc, SELECTED_PROXY_TARGETS,
        idx_active_texture, idx_active_event,
        idx_mapped_active_event, idx_mapped_active_texture,
        idx_selected_proxy_active_texture,
        idx_selected_prioronly_active_event,
        idx_selected_prioronly_active_texture,
        idx_unmapped_inactive,
        fuse_kwargs  (dict ready to pass to fuse_scores_with_tables)
    """
    # Find labels.csv
    if labels_csv_path and Path(labels_csv_path).exists():
        labels_path = Path(labels_csv_path)
    else:
        labels_path = Path(model_dir) / "assets" / "labels.csv"

    bc_labels = (
        pd.read_csv(labels_path)
        .reset_index()
        .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
    )

    NO_LABEL_INDEX = len(bc_labels)

    taxonomy_local = taxonomy.copy()
    taxonomy_local["scientific_name_lookup"] = taxonomy_local["scientific_name"]

    bc_lookup = bc_labels.rename(columns={"scientific_name": "scientific_name_lookup"})

    mapping = taxonomy_local.merge(
        bc_lookup[["scientific_name_lookup", "bc_index"]],
        on="scientific_name_lookup",
        how="left",
    )
    mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL_INDEX).astype(int)

    label_to_bc_index = mapping.set_index("primary_label")["bc_index"]
    BC_INDICES = np.array(
        [int(label_to_bc_index.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32
    )

    MAPPED_MASK = BC_INDICES != NO_LABEL_INDEX
    MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
    UNMAPPED_POS = np.where(~MAPPED_MASK)[0].astype(np.int32)
    MAPPED_BC_INDICES = BC_INDICES[MAPPED_MASK].astype(np.int32)

    CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()

    ACTIVE_CLASSES = [PRIMARY_LABELS[i] for i in np.where(Y_SC.sum(axis=0) > 0)[0]]

    idx_active_texture = np.array(
        [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) in TEXTURE_TAXA],
        dtype=np.int32,
    )
    idx_active_event = np.array(
        [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) not in TEXTURE_TAXA],
        dtype=np.int32,
    )

    idx_mapped_active_texture = idx_active_texture[MAPPED_MASK[idx_active_texture]]
    idx_mapped_active_event = idx_active_event[MAPPED_MASK[idx_active_event]]

    idx_unmapped_active_texture = idx_active_texture[~MAPPED_MASK[idx_active_texture]]
    idx_unmapped_active_event = idx_active_event[~MAPPED_MASK[idx_active_event]]

    idx_unmapped_inactive = np.array(
        [i for i in UNMAPPED_POS if PRIMARY_LABELS[i] not in ACTIVE_CLASSES],
        dtype=np.int32,
    )

    # --- Genus proxy mapping ---
    unmapped_df = mapping[mapping["bc_index"] == NO_LABEL_INDEX].copy()
    unmapped_non_sonotype = unmapped_df[
        ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
    ].copy()

    def _get_genus_hits(scientific_name):
        genus = str(scientific_name).split()[0]
        hits = bc_labels[
            bc_labels["scientific_name"].astype(str).str.match(
                rf"^{re.escape(genus)}\s", na=False
            )
        ]
        return genus, hits

    proxy_map = {}
    for _, row in unmapped_non_sonotype.iterrows():
        target = row["primary_label"]
        sci = row["scientific_name"]
        genus, hits = _get_genus_hits(sci)
        if len(hits) > 0:
            proxy_map[target] = {
                "target_scientific_name": sci,
                "genus": genus,
                "bc_indices": hits["bc_index"].astype(int).tolist(),
                "proxy_scientific_names": hits["scientific_name"].tolist(),
            }

    PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
    SELECTED_PROXY_TARGETS = sorted(
        t for t in proxy_map.keys() if CLASS_NAME_MAP.get(t) in PROXY_TAXA
    )

    selected_proxy_pos = np.array(
        [label_to_idx[c] for c in SELECTED_PROXY_TARGETS], dtype=np.int32
    )

    selected_proxy_pos_to_bc = {
        label_to_idx[target]: np.array(proxy_map[target]["bc_indices"], dtype=np.int32)
        for target in SELECTED_PROXY_TARGETS
    }

    idx_selected_proxy_active_texture = np.intersect1d(selected_proxy_pos, idx_active_texture)
    idx_selected_prioronly_active_texture = np.setdiff1d(
        idx_unmapped_active_texture, selected_proxy_pos
    )
    idx_selected_prioronly_active_event = np.setdiff1d(
        idx_unmapped_active_event, selected_proxy_pos
    )

    # Pre-build the fuse_kwargs dict that fuse_scores_with_tables needs
    fuse_kwargs = dict(
        idx_mapped_active_event=idx_mapped_active_event,
        idx_mapped_active_texture=idx_mapped_active_texture,
        idx_selected_proxy_active_texture=idx_selected_proxy_active_texture,
        idx_selected_prioronly_active_event=idx_selected_prioronly_active_event,
        idx_selected_prioronly_active_texture=idx_selected_prioronly_active_texture,
        idx_unmapped_inactive=idx_unmapped_inactive,
        idx_active_texture=idx_active_texture,
        idx_active_event=idx_active_event,
    )

    return {
        "BC_INDICES": BC_INDICES,
        "MAPPED_MASK": MAPPED_MASK,
        "MAPPED_POS": MAPPED_POS,
        "UNMAPPED_POS": UNMAPPED_POS,
        "MAPPED_BC_INDICES": MAPPED_BC_INDICES,
        "bc_labels": bc_labels,
        "NO_LABEL_INDEX": NO_LABEL_INDEX,
        "CLASS_NAME_MAP": CLASS_NAME_MAP,
        "selected_proxy_pos_to_bc": selected_proxy_pos_to_bc,
        "SELECTED_PROXY_TARGETS": SELECTED_PROXY_TARGETS,
        "idx_active_texture": idx_active_texture,
        "idx_active_event": idx_active_event,
        "idx_mapped_active_event": idx_mapped_active_event,
        "idx_mapped_active_texture": idx_mapped_active_texture,
        "idx_selected_proxy_active_texture": idx_selected_proxy_active_texture,
        "idx_selected_prioronly_active_event": idx_selected_prioronly_active_event,
        "idx_selected_prioronly_active_texture": idx_selected_prioronly_active_texture,
        "idx_unmapped_inactive": idx_unmapped_inactive,
        "fuse_kwargs": fuse_kwargs,
    }


# ---------------------------------------------------------------------------
# 4. Load or compute Perch cache
# ---------------------------------------------------------------------------

def load_or_compute_cache(full_files, data_dir, cache_dir, cache_input_dir,
                          mapping, cfg_dict, infer_fn=None):
    """Load cached Perch outputs or compute from scratch.

    Parameters
    ----------
    full_files : list[str]
        Filenames of fully-labeled soundscapes.
    data_dir : str or Path
        Competition data root.
    cache_dir : str or Path
        Writable cache directory.
    cache_input_dir : str or Path
        Read-only pre-computed cache directory.
    mapping : dict
        Output of ``build_perch_mapping``.
    cfg_dict : dict
        Pipeline config dict with keys proxy_reduce, batch_files, verbose,
        require_full_cache_in_submit, mode.
    infer_fn : callable or None
        TF model ``infer_fn``.  Required if no cache exists.

    Returns
    -------
    meta_full : pd.DataFrame
    scores_full_raw : np.ndarray
    emb_full : np.ndarray
    """
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    cache_input_dir = Path(cache_input_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Search for existing cache
    candidates = [
        (cache_dir / "full_perch_meta.parquet", cache_dir / "full_perch_arrays.npz"),
    ]
    if cache_input_dir.exists():
        candidates.append(
            (cache_input_dir / "full_perch_meta.parquet",
             cache_input_dir / "full_perch_arrays.npz")
        )

    cache_meta_path = cache_npz_path = None
    for meta_p, npz_p in candidates:
        if meta_p.exists() and npz_p.exists():
            cache_meta_path, cache_npz_path = meta_p, npz_p
            break

    if cache_meta_path is not None:
        print(f"Loading cached Perch outputs from: {cache_meta_path.parent}")
        meta_full = pd.read_parquet(cache_meta_path)
        arr = np.load(cache_npz_path)
        scores_full_raw = arr["scores_full_raw"].astype(np.float32)
        emb_full = arr["emb_full"].astype(np.float32)
    else:
        if cfg_dict.get("require_full_cache_in_submit", False) and cfg_dict.get("mode") == "submit":
            raise FileNotFoundError(
                "Submit mode requires cached Perch outputs. "
                "Place full_perch_meta.parquet + full_perch_arrays.npz in cache_dir."
            )
        if infer_fn is None:
            raise RuntimeError(
                "No cache found and no infer_fn provided. "
                "Either provide a Perch cache or install TensorFlow."
            )

        from src.inference.perch import infer_perch_with_embeddings
        print("No cache found. Running Perch on trusted full files...")
        full_paths = [data_dir / "train_soundscapes" / fn for fn in full_files]
        n_classes = len(mapping["MAPPED_POS"]) + len(mapping["UNMAPPED_POS"])
        # n_classes = total classes in competition
        n_classes = mapping["BC_INDICES"].shape[0]

        meta_full, scores_full_raw, emb_full = infer_perch_with_embeddings(
            full_paths,
            infer_fn=infer_fn,
            n_classes=n_classes,
            mapped_pos=mapping["MAPPED_POS"],
            mapped_bc_indices=mapping["MAPPED_BC_INDICES"],
            proxy_pos_to_bc=mapping["selected_proxy_pos_to_bc"],
            batch_files=cfg_dict.get("batch_files", 32),
            verbose=cfg_dict.get("verbose", True),
            proxy_reduce=cfg_dict.get("proxy_reduce", "max"),
        )

        out_meta = cache_dir / "full_perch_meta.parquet"
        out_npz = cache_dir / "full_perch_arrays.npz"
        meta_full.to_parquet(out_meta, index=False)
        np.savez_compressed(out_npz, scores_full_raw=scores_full_raw, emb_full=emb_full)
        print(f"Saved cache to: {cache_dir}")

    return meta_full, scores_full_raw, emb_full


# ---------------------------------------------------------------------------
# 5. Align truth to cached order
# ---------------------------------------------------------------------------

def align_truth_to_cache(full_truth, Y_SC, meta_full):
    """Align ground-truth labels to the cached metadata row order.

    Returns
    -------
    Y_FULL : np.ndarray, same shape as (len(meta_full), N_CLASSES)
    """
    full_truth_aligned = full_truth.set_index("row_id").loc[meta_full["row_id"]].reset_index()
    Y_FULL = Y_SC[full_truth_aligned["index"].to_numpy()]

    assert np.all(full_truth_aligned["filename"].values == meta_full["filename"].values)
    assert np.all(full_truth_aligned["row_id"].values == meta_full["row_id"].values)

    return Y_FULL


# ---------------------------------------------------------------------------
# 6. Load or compute OOF meta-features
# ---------------------------------------------------------------------------

def load_or_compute_oof_meta(scores_full_raw, meta_full, sc_clean, Y_SC,
                             cache_dir, fuse_kwargs, fusion_cfg, n_splits=5,
                             verbose=True):
    """Build or load honest OOF base/prior meta-features.

    Returns
    -------
    oof_base : np.ndarray
    oof_prior : np.ndarray
    oof_fold_id : np.ndarray
    """
    from src.scoring.fusion import build_oof_base_prior

    cache_dir = Path(cache_dir)
    cache_path = cache_dir / "full_oof_meta_features.npz"

    # Merge fusion lambdas into fuse_kwargs for the OOF call
    fuse_kw = dict(fuse_kwargs)
    fuse_kw.setdefault("lambda_event", fusion_cfg.get("lambda_event", 0.4))
    fuse_kw.setdefault("lambda_texture", fusion_cfg.get("lambda_texture", 1.0))
    fuse_kw.setdefault("lambda_proxy_texture", fusion_cfg.get("lambda_proxy_texture", 0.8))
    fuse_kw.setdefault("smooth_texture", fusion_cfg.get("smooth_texture", 0.35))
    fuse_kw.setdefault("smooth_event", fusion_cfg.get("smooth_event", 0.15))

    if cache_path.exists():
        print(f"Loading cached OOF meta-features from: {cache_path}")
        arr = np.load(cache_path)
        oof_base = arr["oof_base"].astype(np.float32)
        oof_prior = arr["oof_prior"].astype(np.float32)
        oof_fold_id = arr["fold_id"].astype(np.int16)
    else:
        print("Building OOF meta-features...")
        oof_base, oof_prior, oof_fold_id = build_oof_base_prior(
            scores_full_raw=scores_full_raw,
            meta_full=meta_full,
            sc_clean=sc_clean,
            Y_SC=Y_SC,
            fuse_kwargs=fuse_kw,
            n_splits=n_splits,
            verbose=verbose,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            oof_base=oof_base,
            oof_prior=oof_prior,
            fold_id=oof_fold_id,
        )
        print(f"Saved OOF meta-features to: {cache_path}")

    return oof_base, oof_prior, oof_fold_id


# ---------------------------------------------------------------------------
# 7. Build per-class temperature vector
# ---------------------------------------------------------------------------

def build_class_temperatures(PRIMARY_LABELS, CLASS_NAME_MAP, temp_cfg):
    """Build per-class temperature array from taxonomy class names.

    Returns
    -------
    np.ndarray, shape (N_CLASSES,)
    """
    T_AVES = temp_cfg.get("aves", 1.10)
    T_TEXTURE = temp_cfg.get("texture", 0.95)

    temps = np.ones(len(PRIMARY_LABELS), dtype=np.float32) * T_AVES
    for ci, label in enumerate(PRIMARY_LABELS):
        cn = CLASS_NAME_MAP.get(label, "Aves")
        if cn in TEXTURE_TAXA:
            temps[ci] = T_TEXTURE

    return temps
