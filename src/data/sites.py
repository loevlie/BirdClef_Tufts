"""Site mapping and file metadata extraction."""

import numpy as np


def build_site_mapping(meta_df):
    sites = meta_df["site"].unique().tolist()
    site_to_idx = {s: i + 1 for i, s in enumerate(sites)}
    n_sites = len(sites) + 1
    return site_to_idx, n_sites


def get_file_metadata(meta_df, file_list, site_to_idx, n_sites_max):
    file_to_row = {}
    filenames = meta_df["filename"].to_numpy()
    sites = meta_df["site"].to_numpy()
    hours = meta_df["hour_utc"].to_numpy()
    for i, f in enumerate(filenames):
        if f not in file_to_row:
            file_to_row[f] = i

    site_ids = np.zeros(len(file_list), dtype=np.int64)
    hour_ids = np.zeros(len(file_list), dtype=np.int64)
    for fi, fname in enumerate(file_list):
        row = file_to_row.get(fname)
        if row is not None:
            sid = site_to_idx.get(sites[row], 0)
            site_ids[fi] = min(sid, n_sites_max - 1)
            hour_ids[fi] = int(hours[row]) % 24
    return site_ids, hour_ids
