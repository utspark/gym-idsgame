import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


@dataclass
class RegressionData:
    """
    A simple data container for four NumPy arrays.
    Each array defaults to an empty 1D float array if not provided.
    """
    benign_windows: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    benign_futures: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    malware_windows: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    malware_futures: np.ndarray = field(default_factory=lambda: np.empty((0,)))


@dataclass
class ModelSettings:
    """
    A simple data container for model preprocessing settings.
    """
    settings_path: Path = None
    problem_formulation: str = "regression"
    preproc_approach: str = None
    window_length: int = 20
    future_length: int = 1
    max_trace_length: int = None
    system_calls: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    model_type: str = None
    model_path: Path = None
    new_model: bool = False
    plot: bool = False

    @property
    def syscalls(self):
        return self.system_calls

    @syscalls.setter
    def syscalls(self, value):
        self.system_calls = value


def first_int(s: str) -> tuple[int, int]:
    m = re.search(r'\d+', s)
    # (0, int) if found; (1, 0) if not — so names without numbers go last
    return (0, int(m.group())) if m else (1, 0)


def concat_short_traces(files: Iterable[str | Path],
                        concat_size: int = 3,
                        out_dir: str | Path = "concatenated",
                        base_name: str = "fscan_group",
                        allow_partial: bool = False) -> list[Path]:
    """
    Concatenate files in order, concat_size at a time.
    - files: iterable of file paths in the exact order to process
    - out_dir: where to write outputs
    - base_name: prefix for output files (group_1.txt, group_2.txt, ...)
    - allow_partial: if True, writes the last group even if it has < 3 files
    Returns list of output Paths.
    """
    paths = [Path(p) for p in files]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths: list[Path] = []
    for i in range(0, len(paths), concat_size):
        group = paths[i:i + concat_size]
        if len(group) < concat_size and not allow_partial:
            break
        out_path = out_dir / f"{base_name}_{(i // concat_size) + 1}.txt"
        with out_path.open("w", encoding="utf-8") as out:
            arr_1 = []
            arr_2 = []
            for j, src in enumerate(group):
                with src.open("r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    # chunk = f.read()

                arr_1.append(np.loadtxt(io.StringIO(lines[0]), dtype=int))
                arr_2.append(np.loadtxt(io.StringIO(lines[1]), dtype=float))

                # out.write(chunk)
                # ensure separation between files (avoid gluing numbers)
                # if not chunk.endswith(" "):
                #     out.write(" ")

            for j in range(1, len(group)):
                arr_2[j] = arr_2[j] - arr_2[j][0] + (arr_2[j][1] - arr_2[j][0])

            for j in range(1, len(group)):
                arr_2[j] += arr_2[j-1][-1]

            arr_1 = np.concatenate(arr_1)
            arr_2 = np.concatenate(arr_2)

            out.write(" ".join(map(str, arr_1)) + "\n")
            out.write(" ".join(map(str, arr_2)))

        out_paths.append(out_path)
    return out_paths


def get_file_arrays(file_path: Path, file_list=None, verbose=False, max_trace_len=0) -> list:
    trace_list = []

    paths = [p for p in file_path.iterdir() if p.is_file()]
    paths.sort()

    if file_list is not None:
        file_set = set(file_list)
        filtered = [path for path in paths if path.name in file_set]
        paths = filtered

    for file_name in paths:
        if verbose:
            print(file_name.name)

        file_path = str(file_name)

        with open(file_path, "r", newline="") as f:
            lines = f.readlines()
            arr1 = np.loadtxt(io.StringIO(lines[0]), dtype=int)
            # arr2 = np.loadtxt(io.StringIO(lines[1]), dtype=float)

            if max_trace_len != 0:
                arr1 = arr1[:max_trace_len]

            trace_list.append(arr1)

    return trace_list


def trace_list_to_windows(model_settings: ModelSettings, trace_list: list):
    def form_windows(array, window_size, window_stride, future_size=0, future_stride=1):
        """
        Returns Numpy array of sliding windows of a Numpy array (timeseries).
        https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
        """

        last_window = len(array) - window_size - future_size * future_stride + 1

        future_idxs = (
                np.expand_dims(np.arange(0, future_size * future_stride, future_stride), 0)
                + np.expand_dims(np.arange(window_size, window_size + last_window, window_stride), 1)
        )
        futures = array[future_idxs]

        window_idxs = (
                np.expand_dims(np.arange(window_size), 0)
                + np.expand_dims(np.arange(0, last_window, window_stride), 1)
        )
        windows = array[window_idxs]

        return windows, futures

    wdw_list = []
    future_list = []
    # print(trace_list)

    for trace in trace_list:
        wdws, futures = form_windows(
            trace,
            model_settings.window_length,
            1,
            model_settings.future_length,
            future_stride=1)

        wdw_list.append(wdws)
        future_list.append(futures)

    if len(wdw_list) > 1:
        wdws = np.concatenate(wdw_list)
        futures = np.concatenate(future_list)

    else:
        wdws = wdw_list[0]
        futures = future_list[0]

    return wdws, futures


def get_windows_and_futures(model_settings: ModelSettings, file_dir: Path, file_list=None):
    trace_list = get_file_arrays(file_dir, file_list, max_trace_len=model_settings.max_trace_length)
    windows, futures = trace_list_to_windows(model_settings, trace_list)

    return windows, futures


def full_trace_samples(model_settings: ModelSettings, file_dir: Path, file_list=None):
    trace_list = get_file_arrays(file_dir, file_list)
    trace_list = [arr[:model_settings.max_trace_length] for arr in trace_list]

    # TODO think of proper padding approach
    padded = [
        np.pad(a,
               pad_width=(0, model_settings.max_trace_length - a.shape[0]),
               mode='constant',
               constant_values=0)
        for a in trace_list
    ]

    padded_matrix = np.vstack(padded)

    return padded_matrix


def preproc_transform(model_settings: ModelSettings, file_dir: Path, file_list=None) -> tuple:
    mode = model_settings.preproc_approach
    VALID_MODES = {"zero-padded_trace", "syscall_frequency", "windowed_features", "windowed"}

    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)!r}, got {mode!r}")

    if mode == "zero-padded_trace":
        transformed = full_trace_samples(model_settings, file_dir, file_list)

    elif mode == "syscall_frequency":
        transformed = full_trace_samples(model_settings, file_dir, file_list)
        transformed = np.sum(transformed[..., None] == model_settings.syscalls, axis=1)

    elif mode == "windowed_features":
        windows, _ = get_windows_and_futures(model_settings, file_dir, file_list)

        feature_array = np.zeros((windows.shape[0], 6))
        feature_array[:, 0] = np.max(windows, axis=1)
        feature_array[:, 1] = np.min(windows, axis=1)
        feature_array[:, 2] = np.median(windows, axis=1)
        feature_array[:, 3] = np.mean(windows, axis=1)
        feature_array[:, 4] = np.ptp(windows, axis=1)
        feature_array[:, 5] = np.std(windows, axis=1)

        transformed = feature_array

    else:  # mode == "windowed"
        windows, futures = get_windows_and_futures(model_settings, file_dir, file_list)
        transformed = (windows, futures)

    return transformed


def get_system_call_map(model_settings: ModelSettings, file_dir: Path, file_list=None) -> np.ndarray:
    transformed = full_trace_samples(model_settings, file_dir, file_list)
    system_calls = np.unique(transformed)

    return np.sort(system_calls)


def form_array_from_files(child_path, file_subsample=-1) -> np.ndarray:
    if file_subsample == -1:
        file_subsample = 40

    files = [p for p in child_path.iterdir() if p.is_file()]
    files.sort()

    trace_list = []
    # TODO use all files
    for file_name in files[0:file_subsample]:
        if verbose := False:
            print(file_name.name)
        file_path = str(file_name)

        with open(file_path, "r", newline="") as f:
            arr = np.loadtxt(f, dtype=int)
            trace_list.append(arr)

    if not trace_list:
        return np.empty((0,))

    max_len = max(a.shape[0] for a in trace_list)

    # TODO think of proper padding approach
    padded = [
        np.pad(a,
               pad_width=(0, max_len - a.shape[0]),
               mode='constant',
               constant_values=0)
        for a in trace_list
    ]

    padded_matrix = np.vstack(padded)

    return padded_matrix


def form_one_hot_encoder(benign_array: np.ndarray) -> OneHotEncoder:
    max_syscall = np.max(benign_array)
    one_hot_array = np.array(range(max_syscall))
    one_hot_array = one_hot_array.reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(one_hot_array)

    return enc


def increase_padding(pad_target: np.ndarray, max_len: int) -> np.ndarray:
    pad_len = max_len - pad_target.shape[1]
    pad_array = np.zeros((pad_target.shape[0], pad_len))
    padded = np.concatenate((pad_target, pad_array), axis=1)

    return padded


def unsupervised_preproc(model_settings: ModelSettings, benign_dir: Path, malware_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesses data for unsupervised learning.
    Returns: X_train, X_test, y_test
    """
    benign = preproc_transform(model_settings, benign_dir)
    malware = preproc_transform(model_settings, malware_dir)

    X = benign
    y = np.zeros(len(X)) + 1

    X_train, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_test = np.concatenate((X_test, malware))

    malware_labels = np.zeros((len(malware))) - 1
    y_test = np.concatenate((y_test, malware_labels))

    return X_train, X_test, y_test
