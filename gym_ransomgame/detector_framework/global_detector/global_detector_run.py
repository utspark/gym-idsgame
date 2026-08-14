import time
from pathlib import Path

import joblib
import numpy as np

from detector_framework import config, local_detector, global_detector
from detector_framework.local_detector.local_detector_run import DATA_PATH


def get_default_config():
    USE_PRESCORE = False

    # DATA_PATH = Path(__file__).resolve().parents[2] / "data"
    model_dir = DATA_PATH / "models/local_detector_analysis"

    problem_formulation = "multiclass_supervised"
    preproc_approach = "windowed_features"
    model_type = "decision_tree"
    model_stem = f"{problem_formulation}_{preproc_approach}_{model_type}"
    settings_path = model_dir / f"{model_stem}_settings.joblib"
    model_path = model_dir / f"{model_stem}.joblib"

    model_settings = joblib.load(settings_path)
    model_settings.model_path = model_path
    classifier = joblib.load(model_settings.model_path)

    prescored_dir = DATA_PATH / "prescored_windows"
    malware_path = DATA_PATH / "trace_data/syscall_bucket"

    generation_attack_stages = config.GENERATION_ATTACK_STAGES

    model_paths = {
        "syscall_clf_path": model_path,
        "network_clf_path": None,
        "hpc_clf_path": None,
    }

    la_components = {
        "density": True,
        "propagation": True,
        "memory": False,
    }

    return (
        USE_PRESCORE,
        model_settings,
        classifier,
        prescored_dir,
        malware_path,
        generation_attack_stages,
        model_paths,
        la_components,
    )


def run_global_detector(
    USE_PRESCORE,
    model_settings,
    classifier,
    prescored_dir,
    malware_path,
    generation_attack_stages,
    model_paths,
    la_components,
    num_sequences=5,
):
    gd = global_detector.LifecycleDetector(
        **model_paths,
        lifecycle_awareness=True,
        stage_filter=False,
        **la_components,
    )

    results = []
    for i in range(num_sequences):
        stage_keys, stage_windows = (
            global_detector.LifecycleDetector.form_lifecycle_sequence(
                generation_attack_stages, benign=False
            )
        )

        if USE_PRESCORE:
            trace_classes, trace_values = local_detector.get_prescored_predictions(
                stage_keys, stage_windows, prescored_dir
            )

        else:
            trace_classes, trace_values = local_detector.get_live_predictions(
                stage_keys, stage_windows, classifier, model_settings, malware_path
            )

        translation = config.SYSCALL_BENIGN_MALWARE_CLASS_TRANSLATION
        proba = gd.score_single_layer(trace_classes, trace_values, translation)
        print(f"{proba: 6.5f}")
        results.append(proba)

    DATA_PATH = Path(__file__).resolve().parents[1] / "data"
    model_dir = DATA_PATH / "models/local_detector_analysis"

    problem_formulation = "multiclass_supervised"
    preproc_approach = "windowed_features"
    model_type = "decision_tree"
    model_stem = f"{problem_formulation}_{preproc_approach}_{model_type}"
    model_path = model_dir / f"{model_stem}.joblib"

    save_path = DATA_PATH / "models/global_detector.joblib"
    joblib.dump(gd, save_path)

    t_global_start = time.time()
    X = np.random.randint(low=0, high=4, size=10000).reshape(-1, 1)
    proba = gd.hmm.score_samples(X)
    t_global_inf = time.time() - t_global_start
    print(f"Overhead {t_global_inf * 1000:.4f}ms")

    return results


if __name__ == "__main__":
    config.set_seed()
    (
        USE_PRESCORE,
        model_settings,
        classifier,
        prescored_dir,
        malware_path,
        generation_attack_stages,
        model_paths,
        la_components,
    ) = get_default_config()

    run_global_detector(
        USE_PRESCORE,
        model_settings,
        classifier,
        prescored_dir,
        malware_path,
        generation_attack_stages,
        model_paths,
        la_components,
    )
