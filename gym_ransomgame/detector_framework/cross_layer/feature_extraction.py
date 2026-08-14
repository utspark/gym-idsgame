import pandas as pd
import numpy as np
from itertools import islice
from typing import Iterable, List, Tuple, Optional, Callable


def get_time_windows(t: pd.Series, window_size_time: float, window_stride_time: float):
    t = t.copy().reset_index(drop=True)

    left_edges = np.arange(t.iloc[0], t.iloc[-1] - window_size_time, window_stride_time)
    right_edges = np.arange(t.iloc[0] + window_size_time, t.iloc[-1], window_stride_time)

    if len(left_edges) == 0:
        left_indices = np.array([0])
        right_indices = np.array([len(t)])

    else:
        left_indices = np.searchsorted(t, left_edges, side="left")
        right_indices = np.searchsorted(t, right_edges, side="left")

    return left_indices, right_indices


def chunked(it: Iterable, n: int):
    """Yield lists of up to n items from iterable it."""
    it = iter(it)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


def features_batch(pairs: List[Tuple[int, int]], features_one_func: Callable) -> List[List[float]]:
    """Compute features for a batch of windows using a provided function."""
    out = []
    for p in pairs:
        try:
            row = features_one_func(p)
            if row is not None and row[0] != 0:
                out.append(row)
        except Exception as e:
            # Just to be safe, but we don't want to crash everything if one window fails
            # though it really shouldn't.
            continue
    return out

