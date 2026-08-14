from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import feature_extraction as sklearn_feature_extraction

from detector_framework.cross_layer import feature_extraction

from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from typing import Iterable, List, Tuple, Optional


DEFAULT_COUNTERS = [
    'instructions', 'LLC-load-misses', 'avx_insts.all', 'br_inst_retired.all_branches'
]

FULL_COUNTERS = [
    'instructions', 'LLC-load-misses', 'avx_insts.all', 'block:block_rq_issue',
    'br_inst_retired.all_branches', 'cache-references', 'mem-loads', 'mem-stores',
    'uops_executed_port.port_0', 'uops_executed_port.port_1', 'uops_executed_port.port_2',
    'uops_executed_port.port_3', 'uops_executed_port.port_4', 'uops_executed_port.port_5',
    'uops_executed_port.port_6', 'uops_executed_port.port_7',
]


def get_file_df(filepath: Path, use_full_counters: bool = False, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    cols = FULL_COUNTERS if use_full_counters else DEFAULT_COUNTERS
    cols = ['time'] + cols

    df = df[cols]
    df["time"] = df.time.astype(float)
    df["time"] = df["time"] - df["time"][0]

    df = df.fillna(0)

    return df


_G = {}


def _init_worker(feature_arrays: dict):
    """Runs once per worker process; stash read-only arrays in module globals."""
    global _G
    _G = feature_arrays


def _features_one(pair: Tuple[int, int]) -> Optional[List[float]]:
    """Compute features for a single [i:j) window using globals set by _init_worker."""
    i, j = pair
    if j <= i:
        return None

    # 'instructions' must be first for normalization
    instr_array = _G.get("instructions")
    if instr_array is None:
        return None

    instr_count = np.mean(instr_array[i:j])

    other_means = []
    for name, arr in _G.items():
        if name == "instructions":
            continue
        other_means.append(np.mean(arr[i:j]))

    if instr_count > 0:
        normalized_means = [mean / instr_count for mean in other_means]
    else:
        normalized_means = [0.0] * len(other_means)

    return [instr_count] + normalized_means


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
    use_full_counters: bool = False,
    **kwargs,
) -> pd.DataFrame:
    # Build windows on the main process
    left_idx, right_idx = feature_extraction.get_time_windows(
        df["time"], window_size_time, window_stride_time
    )
    pairs = list(zip(left_idx, right_idx))

    counter_cols = FULL_COUNTERS if use_full_counters else DEFAULT_COUNTERS

    # Map CSV names to internal feature names (replacing hyphens and dots with underscores)
    # This maintains consistency with previous code's column naming in X
    name_mapping = {
        'instructions': 'instructions',
        'LLC-load-misses': 'LLC_load_misses',
        'avx_insts.all': 'avx_insts_all',
        'br_inst_retired.all_branches': 'br_inst_retired',
    }

    feature_arrays = {}
    cols = []
    for col in counter_cols:
        internal_name = name_mapping.get(col, col.replace('-', '_').replace('.', '_').replace(':', '_'))
        feature_arrays[internal_name] = df[col].to_numpy(dtype=float, copy=False)
        cols.append(internal_name)

    # Spin up the pool; each worker gets arrays once via initializer
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(feature_arrays,)
    ) as ex:
        # Map in batches to reduce per-task overhead
        results = ex.map(_features_batch, feature_extraction.chunked(pairs, chunksize))
        rows = [row for batch in results for row in batch]

    X = pd.DataFrame(rows, columns=cols)

    if preserve_time:
        X["time"] = df["time"].iloc[right_idx].reset_index(drop=True)
        X.insert(0, "time", X.pop("time"))

    return X

