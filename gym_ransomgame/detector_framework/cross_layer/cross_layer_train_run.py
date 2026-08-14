from copy import deepcopy
from pathlib import Path
from typing import Iterable, Mapping, Tuple, Optional, Dict, Literal, Any, Sequence
from types import ModuleType

import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import os
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from detector_framework import config
import detector_framework
from detector_framework.local_detector import local_detector
from detector_framework.cross_layer import network_signals, syscall_signals, hpc_signals
from detector_framework.local_detector.local_detector_run import DATA_PATH

if os.environ.get("DISPLAY", "") == "":
    print("No display found. Using non-interactive Agg backend.")
    matplotlib.use("Agg")
else:
    try:
        matplotlib.use("Qt5Agg")
    except ImportError:
        print("Qt5Agg not found. Falling back to Agg.")
        matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.ion()


def form_signal_dict(behaviors: dict, signal_modules: dict) -> dict:
    """
    Form signal_dictionary containing raw dfs from each file trace
    Args:
        behaviors:
        signal_modules:

    Returns:

    """

    # signal_df_dict = defaultdict(lambda: defaultdict(list))
    #
    # for action, signals in behaviors.items():
    #     for signal, file_paths in signals.items():
    #         mod = signal_modules[signal]
    #         dfs = [mod.get_file_df(fp) for fp in file_paths]
    #
    #         for df in dfs:
    #             signal_df_dict[signal][action].append(df)

    signal_df_dict = {}

    for action, signals in behaviors.items():
        for signal, file_paths in signals.items():
            mod = signal_modules[signal]

            #  TODO remove
            if os.path.isdir(file_paths[0]):
                dfs = []

            else:
                dfs = [mod.get_file_df(fp) for fp in file_paths]

            # Ensure nested dict and list exist
            signal_df_dict.setdefault(signal, {})
            signal_df_dict[signal].setdefault(action, [])
            signal_df_dict[signal][action].extend(dfs)

    return signal_df_dict


def form_feature_frames(
    feature_dict: dict, train_test_split: float, train: bool
) -> dict:
    feature_frames = {}

    for signal, actions in feature_dict.items():
        for action, X_list in actions.items():
            pruned = [df.drop(columns="time", errors="ignore") for df in X_list]

            feature_frames.setdefault(signal, {})

            if len(pruned) == 0:
                feature_frames[signal][action] = pd.DataFrame()
            else:
                df = pd.concat(pruned, ignore_index=True, sort=False, copy=False)

                n = len(df)
                idx = int(np.floor(train_test_split * n))
                df_train = df.iloc[:idx]
                df_test = df.iloc[idx:]

                feature_frames[signal][action] = df_train if train else df_test

    return feature_frames


def build_cross_layer_X(
    feature_dict: dict[str, dict[str, pd.DataFrame]],
    attack_lens: Iterable[Tuple[str, float]],
    window_size_time: float,
    window_stride_time: float,
    signals: Sequence[str] = ("syscall", "network", "hpc"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def cross_layer_concatenate(
        attack_X: list,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        syscall_X, network_X, hpc_X = zip(*attack_X)

        outer_X = [np.concatenate(X_list) for X_list in (syscall_X, network_X, hpc_X)]

        cross_layer_X = (outer_X[0], outer_X[1], outer_X[2])

        return cross_layer_X

    cross_layer_X = []

    for attack, t in attack_lens:
        # Compute desired window count from duration, then clamp to the shortest signal length
        desired = int((t - window_size_time) / window_stride_time) + 1
        if desired <= 0:
            continue  # nothing to sample for this attack

        # Gather per-signal frames once and compute min length
        sig_frames = [feature_dict[signal].get(attack, None) for signal in signals]

        cleaned_frames = []
        for frame in sig_frames:
            if frame is None or len(frame) == 0:
                tmp = pd.DataFrame()
                tmp["filler"] = pd.Series([-1] * 1000)
                frame = tmp

            cleaned_frames.append(frame)
        sig_frames = cleaned_frames

        sig_lengths = [len(df) for df in sig_frames]

        if not sig_lengths:
            continue
        num_windows = min(desired, min(sig_lengths))
        if num_windows <= 0:
            continue

        # Sample aligned windows for each signal
        X_list = []
        for df, n, width in zip(sig_frames, sig_lengths, [6, 11, 4]):
            start = np.random.choice(range(0, n - num_windows + 1))
            sampled_df = df.iloc[start : start + num_windows]

            # Ensure correct width
            if sampled_df.shape[1] != width:
                sampled_df = pd.DataFrame(np.zeros((sampled_df.shape[0], width)) - 1)

            X_list.append(sampled_df)

        cross_layer_X.append(X_list)

    cross_layer_X = cross_layer_concatenate(cross_layer_X)

    return cross_layer_X


def outer_train_loop(
    parameter_dict: dict, window_size_time, window_stride_time, tts, train: bool
) -> list:
    train_scores = []

    for params in parameter_dict.values():
        if not params["signal_selection"]:
            continue

        data_dir = params["data_dir"]
        malware_dict = params["malware_dictionary"]
        signal_fe = params["feature_extraction_module"]
        save_path = params["save_path"]

        data_paths = [p for p in data_dir.iterdir() if p.is_file()]
        data_paths.sort()

        malware_keys = [item for sublist in malware_dict.values() for item in sublist]
        malware_keys = set(malware_keys)

        filtered = [
            path for path in data_paths if any(key in path.name for key in malware_keys)
        ]
        data_paths = filtered

        # TODO uncomment this
        # subsampled = []
        # for key in malware_keys:
        #     tmp_list = [path for path in syscall_paths if key in str(path)]
        #     subsample = int(len(tmp_list) * 0.6)
        #     subsampled.extend(tmp_list[:subsample])
        # syscall_paths = subsampled

        X, y, xt, yt = local_detector.files_and_labels_to_X_y(
            data_paths,
            signal_fe,
            malware_dict,
            window_size_time,
            window_stride_time,
            train_test_split=tts,
        )

        # print(np.unique(y, return_counts=True))
        # print(np.unique(yt, return_counts=True))

        if train:
            train_score = local_detector.train_and_save_model(X, y, save_path)
            train_scores.append(train_score)
            print(f"Train Score: {train_score}")
        else:
            local_detector.train_and_test_report(X, y)

    return train_scores


def form_feature_frame_joblib(
    cwd: Path,
    feature_frames_path: Path,
    window_size_time,
    window_stride_time,
    tts,
    behaviors,
    signal_modules: dict,
):

    DATA_PATH = Path(__file__).resolve().parents[2] / "data"
    syscall_dir = DATA_PATH / "trace_data/syscall_bucket"
    network_dir = DATA_PATH / "trace_data/network_bucket"
    hpc_dir = DATA_PATH / "trace_data/hpc_bucket"

    for behavior in behaviors:
        for signal_dir, signal in zip(
            [syscall_dir, network_dir, hpc_dir], signal_modules.keys()
        ):
            behaviors[behavior][signal] = [
                signal_dir / file_path for file_path in behaviors[behavior][signal]
            ]

    signal_df_dict = form_signal_dict(behaviors, signal_modules)
    feature_dict = local_detector.build_features(
        signal_df_dict,
        signal_modules,
        window_size_time,
        window_stride_time,
        preserve_time=True,
    )

    #  TODO there should be time alignment of various signals
    #   - because there is not, no use in below functions
    #   - ask Prateek for time alignment; a trace of some action
    #   - should take the same length across all signals
    #   e.g. AES_128 takes 15 seconds in syscalls and hpc
    # correct_feature_vector_times_2(feature_dict)
    # correct_feature_vector_times(feature_dict)

    feature_frames = form_feature_frames(
        feature_dict, train_test_split=tts, train=False
    )

    filepath = Path(feature_frames_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_frames, feature_frames_path)

    return feature_frames


if __name__ == "__main__":
    config.set_seed()

    SYSCALL = True
    NETWORK = True
    HPC = True
    TRAIN = True  # train new local detectors or quickly go through optimization/evaluation loop
    REPROCESS_DATA = True

    cwd = Path.cwd()
    tts = detector_framework.config.TRAIN_TEST_SPLIT

    # window_size_time = 0.1 / 2  # / 2  # 10
    # window_stride_time = window_size_time / 3
    window_size_time = config.WINDOW_SIZE_TIME
    window_stride_time = config.WINDOW_STRIDE_TIME
    # rng = np.random.default_rng(seed=1337)  # optional seed
    # random.seed(1337)

    DATA_PATH = Path(__file__).resolve().parents[2] / "data"
    MODEL_PATH = Path(__file__).resolve().parents[1] / "data/models"

    iteration_dict = {
        "syscall": {
            "signal_selection": SYSCALL,
            "data_dir": DATA_PATH / "trace_data/syscall_bucket",
            "malware_dictionary": detector_framework.config.SYSCALL_BENIGN_MALWARE_DICT,
            "feature_extraction_module": syscall_signals,
            "save_path": MODEL_PATH / "syscall_clf.joblib",
        },
        "network": {
            "signal_selection": NETWORK,
            "data_dir": DATA_PATH / "trace_data/network_bucket",
            "malware_dictionary": detector_framework.config.NETWORK_BENIGN_MALWARE_DICT,
            "feature_extraction_module": network_signals,
            "save_path": MODEL_PATH / "network_clf.joblib",
        },
        "hpc": {
            "signal_selection": HPC,
            "data_dir": DATA_PATH / "trace_data/hpc_bucket",
            "malware_dictionary": detector_framework.config.HPC_BENIGN_MALWARE_DICT,
            "feature_extraction_module": hpc_signals,
            "save_path": MODEL_PATH / "hpc_clf.joblib",
        },
    }

    outer_train_loop(iteration_dict, window_size_time, window_stride_time, tts, TRAIN)

    signal_modules = {
        "syscall": syscall_signals,
        "network": network_signals,
        "hpc": hpc_signals,
    }

    behaviors = deepcopy(detector_framework.config.BEHAVIOR_FILES)

    feature_frames_path = DATA_PATH / "trace_data/feature_frames.joblib"

    if REPROCESS_DATA:
        feature_frames = form_feature_frame_joblib(
            cwd,
            feature_frames_path,
            window_size_time,
            window_stride_time,
            tts,
            behaviors,
            signal_modules,
        )

    else:
        feature_frames = joblib.load(feature_frames_path)
