from pathlib import Path
import numpy as np
import io

import pandas as pd

from detector_framework.cross_layer import feature_extraction

from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional


def get_file_df(filepath: Path) -> pd.DataFrame:
    with open(filepath, "r", newline="") as f:
        lines = f.readlines()
        arr1 = np.loadtxt(io.StringIO(lines[0]), dtype=int)
        arr2 = np.loadtxt(io.StringIO(lines[1]), dtype=float)

    arr1 = arr1.reshape(-1, 1)
    arr2 = arr2.reshape(-1, 1)
    df = pd.DataFrame(np.column_stack((arr2, arr1)), columns=["seconds", "syscall"])

    df["seconds"] = df["seconds"] - df["seconds"][0]

    return df


# ---------- worker globals ----------
_G = {}

def _init_worker(syscalls):
    """Runs once per worker process; stash read-only arrays in module globals."""
    global _G
    _G = {
        "syscalls": syscalls,
    }

def _features_one(pair: Tuple[int, int]) -> Optional[List[float]]:
    """Compute features for a single [i:j) window using globals set by _init_worker."""
    i, j = pair
    if j <= i:
        return None
    
    values = _G["syscalls"][i:j]

    window_len = len(values)
    syscall_max = float(np.max(values))
    syscall_mean = float(np.mean(values))
    syscall_min = float(np.min(values))
    syscall_std = float(np.std(values))
    syscall_ptp = float(np.ptp(values))

    return [
        window_len,
        syscall_max,
        syscall_mean,
        syscall_min,
        syscall_std,
        syscall_ptp,
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
    chunksize: int = 512,
    preserve_time: bool = False,
) -> pd.DataFrame:
    # Build windows on the main process
    left_idx, right_idx = feature_extraction.get_time_windows(
        df["seconds"], window_size_time, window_stride_time
    )
    pairs = list(zip(left_idx, right_idx))

    # Extract needed columns as arrays
    syscalls = df["syscall"].to_numpy(dtype=int, copy=False)

    # Spin up the pool
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(syscalls,),
    ) as ex:
        results = ex.map(_features_batch, feature_extraction.chunked(pairs, chunksize))
        rows = [row for batch in results for row in batch]

    cols = [
        "window_len",
        "syscall_max",
        "syscall_mean",
        "syscall_min",
        "syscall_std",
        "syscall_ptp",
    ]

    X = pd.DataFrame(rows, columns=cols)

    if preserve_time:
        X["time"] = df["seconds"].iloc[right_idx].reset_index(drop=True)
        X.insert(0, "time", X.pop("time"))

    return X


def file_df_feature_extraction(df: pd.DataFrame, window_size_time, window_stride_time, preserve_time=False) -> pd.DataFrame:
    return file_df_feature_extraction_parallel(
        df, window_size_time, window_stride_time, preserve_time=preserve_time
    )


