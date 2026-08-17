from pathlib import Path

import numpy as np

from detector_framework.global_detector import global_detector


def gd_score_bounds(gd: global_detector.LifecycleDetector):
    """
    Calculates the bounds for the GD score and outputs the computed HMM score.

    This function evaluates the upper and lower bounds of GD (Global Detector)
    scores by using maximum HMM score bounds alongside density and propagation
    penalties. It also computes and displays the HMM score based on the input
    lifecycle detector's stage sequence.

    # GD score bounds
    # Max density penalty is 0.08 (DEFAULT_DENSITY_SCALER)
    # Max propagation penalty is longest_increasing_subsequence*DEFAULT_PROPAGATION_SCALER = 4*0.12
    # GD score bounds = [0, 0.96]

    Parameters:
    gd : global_detector.LifecycleDetector
        Instance of LifecycleDetector used to compute the HMM and GD scores.
    """
    # Max HMM score bounds is [0, 0.4]
    stage_seq_filtered = [0]
    hmm_score = np.exp(gd.hmm.score(np.array(stage_seq_filtered).reshape(-1, 1)))
    hmm_score = np.power(hmm_score, 1 / len(stage_seq_filtered))
    print(f"\nhmm_score: {hmm_score:.4f}")


def main():
    """
    Find GD score bounds

    LD score bounds are scaled 0-1
    """
    # Instantiate Global Detector
    MODEL_DIR = Path(__file__).resolve().parents[1] / "detector_framework/data/models"
    model_paths = {
        "syscall_clf_path": MODEL_DIR / "syscall_clf.joblib",
        "network_clf_path": MODEL_DIR / "network_clf.joblib",
        "hpc_clf_path": MODEL_DIR / "hpc_clf.joblib",
    }
    la_components = {
        "density": True,
        "propagation": True,
    }

    gd = global_detector.LifecycleDetector(
        **model_paths, lifecycle_awareness=True, stage_filter=False, **la_components
    )

    gd_score_bounds(gd)


if __name__ == "__main__":
    main()
