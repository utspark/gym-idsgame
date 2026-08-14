from pathlib import Path

import numpy as np
import pandas as pd

from detector_framework.cross_layer import feature_extraction

import detector_framework

from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from typing import Iterable, List, Tuple, Optional
import os


def fix_bad_line(fields: list[str]) -> list[str] | None:
    # Fix lines that have too many/few fields
    expected = 11
    sep = ","

    if len(fields) > expected:
        # keep first expected-1 fields; glue the remainder into the last field
        tail = sep.join(fields[expected - 1:])
        return fields[:expected - 1] + [tail]
    elif len(fields) < expected:
        # (optional) pad short rows
        return fields + [""] * (expected - len(fields))
    return fields


def get_file_df(filepath: Path) -> pd.DataFrame:
    sep = ","



    df = pd.read_csv(
        filepath,
        sep=sep,
        engine="python",  # needed for callable on_bad_lines
        on_bad_lines=fix_bad_line  # normalize bad rows on the fly
    )

    header_cols = [
        "frame.time", "ip.src", "tcp.srcport", "ip.dst", "tcp.dstport", "udp.srcport", "udp.dstport", "frame.len",
        "_ws.col.Protocol", "_ws.col.Info"
    ]

    if header_cols[0] not in df.columns:
        df.drop(columns=df.columns[0], inplace=True)
        old_cols = df.columns.tolist()
        df2 = df.copy()
        df2.columns = header_cols  # set the new header

        data_row = pd.DataFrame([old_cols], columns=header_cols)
        df = pd.concat([data_row, df2], ignore_index=True)

    # df = df[::detector_framework.config.SUBSAMPLE_NETWORK_DATA]

    df.reset_index(drop=True, inplace=True)
    df["tcp.srcport"] = df["tcp.srcport"].fillna(0)
    if df["tcp.srcport"].dtype != str:
        df["tcp.srcport"] = df["tcp.srcport"].astype(str)
    ipv4 = r'\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b'
    df = df[~df['tcp.srcport'].str.contains(ipv4, regex=True, na=False)]

    pat = r'((?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?)'
    df['time'] = df['frame.time'].str.extract(pat)  # first match per row
    df['seconds'] = pd.to_timedelta(df['time']).dt.total_seconds()
    df["seconds"] = df["seconds"] - df["seconds"][0]

    df.drop(columns=["frame.time", "time"], inplace=True)
    col = "seconds"
    df = df[[col] + [c for c in df.columns if c != col]]

    indexes = df[df["ip.src"].str.contains("Unnamed: ", na=False)].index.tolist()
    df = df.drop(indexes)

    df["merged_src"] = df["tcp.srcport"].fillna(df["udp.srcport"])
    df["merged_src"] = df["merged_src"].fillna(0)
    df["merged_src"] = df["merged_src"].astype(float).astype(int)
    df["merged_dst"] = df["tcp.dstport"].fillna(df["udp.dstport"])
    df["merged_dst"] = df["merged_dst"].fillna(0)
    df["merged_dst"] = df["merged_dst"].astype(float).astype(int)
    df = df.drop(columns=["tcp.srcport", "udp.srcport", "tcp.dstport", "udp.dstport", "ip.src", "ip.dst"])
    df = df.drop(columns=["_ws.col.Info"])

    return df


# ---------- worker globals ----------
_G = {}

def _init_worker(frame_len, merged_src, merged_dst, protocols):
    """Runs once per worker process; stash read-only arrays in module globals."""
    global _G
    _G = {
        "frame_len": frame_len,
        "merged_src": merged_src,
        "merged_dst": merged_dst,
        "protocols": protocols,
    }

def _features_one(pair: Tuple[int, int]) -> Optional[List[float]]:
    """Compute features for a single [i:j) window using globals set by _init_worker."""
    i, j = pair
    if j <= i:
        return None
    fl = _G["frame_len"][i:j]
    ms = _G["merged_src"][i:j]
    md = _G["merged_dst"][i:j]
    pr = _G["protocols"][i:j]          # dtype=str

    window_len   = fl.size
    max_len      = float(fl.max())
    mean_len     = float(fl.mean())
    min_len      = float(fl.min())
    std_len      = float(fl.std())
    unique_src   = int(np.unique(ms).size)
    unique_dst   = int(np.unique(md).size)

    # Count protocol occurrences (substring presence, robust even if compound strings)
    ssh = int((pr == "SSH").sum())
    nfs = int((pr == "NFS").sum())
    arp = int((pr == "ARP").sum())
    tcp = int((pr == "TCP").sum())

    denom = ssh + nfs + arp + tcp + 1

    return [
        window_len,
        max_len,
        mean_len,
        min_len,
        std_len,
        unique_src,
        unique_dst,
        ssh / denom,
        nfs / denom,
        arp / denom,
        tcp / denom,
    ]

def _features_batch(pairs: List[Tuple[int, int]]) -> List[List[float]]:
    return feature_extraction.features_batch(pairs, _features_one)


# ---------- parallelized main function ----------
def file_df_feature_extraction_parallel(
    df: pd.DataFrame,
    window_size_time: float,
    window_stride_time: float,
    *,
    n_workers: Optional[int] = None,
    chunksize: int = 512,   # number of windows per task to reduce overhead
    preserve_time: bool = False,
) -> pd.DataFrame:
    # Build windows on the main process
    left_idx, right_idx = feature_extraction.get_time_windows(
        df["seconds"], window_size_time, window_stride_time
    )
    pairs = list(zip(left_idx, right_idx))

    # Extract needed columns as arrays (far cheaper to slice than DataFrame in workers)
    frame_len  = df["frame.len"].to_numpy(dtype=float, copy=False)
    merged_src = df["merged_src"].to_numpy(copy=False)
    merged_dst = df["merged_dst"].to_numpy(copy=False)
    protocols  = df["_ws.col.Protocol"].astype(str).to_numpy(copy=False)

    # Spin up the pool; each worker gets arrays once via initializer
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(frame_len, merged_src, merged_dst, protocols),
    ) as ex:
        # Map in batches to reduce per-task overhead
        results = ex.map(_features_batch, feature_extraction.chunked(pairs, chunksize))
        rows = [row for batch in results for row in batch]

    cols = [
        "window_len",
        "max_frame_len",
        "mean_frame_len",
        "min_frame_len",
        "std_frame_len",
        "unique_src_ports",
        "unique_dst_ports",
        "ssh_count",
        "nfs_count",
        "arp_count",
        "tcp_count",
    ]

    X = pd.DataFrame(rows, columns=cols)

    if preserve_time:
        X["time"] = df["seconds"].iloc[right_idx].reset_index(drop=True)
        X.insert(0, "time", X.pop("time"))

    return X

