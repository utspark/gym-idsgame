from pathlib import Path

from detector_framework import config
from detector_framework.data_processing.processing import (
    ModelSettings,
    get_system_call_map,
)
from detector_framework.local_detector.local_detector import (
    regression_analysis,
    binary_supervised_analysis,
    unsupervised_analysis,
    multiclass_analysis,
)

config.set_seed()


def run_local_detector(
    model_settings: ModelSettings,
    benign_path: Path,
    benign_dict: dict,
    malware_path: Path,
    malware_dict: dict,
):
    benign_list = list(benign_dict.values())
    malware_list = list(malware_dict.values())

    formulation = model_settings.problem_formulation

    if formulation == "regression":
        return regression_analysis(
            model_settings, benign_path, benign_list, malware_path, malware_list
        )

    if formulation == "binary_supervised":
        if model_settings.preproc_approach == "syscall_frequency":
            model_settings.system_calls = get_system_call_map(
                model_settings, benign_path, benign_list
            )

        return binary_supervised_analysis(
            model_settings, benign_path, benign_list, malware_path, malware_list
        )

    if formulation == "unsupervised":
        return unsupervised_analysis(
            model_settings, benign_path, benign_list, malware_path, malware_list
        )

    if formulation == "multiclass_supervised":
        return multiclass_analysis(
            model_settings, benign_path, benign_dict, malware_path, malware_dict
        )

    raise ValueError(f"Unknown problem formulation: {formulation}")


DATA_PATH = Path(__file__).resolve().parents[1] / "data"


def get_default_config():
    problem_formulation = "multiclass_supervised"
    preproc_approach = "windowed_features"
    window_len = 40
    future_len = 1
    max_trace_length = 50_000
    model_type = "decision_tree"

    model_dir = DATA_PATH / "models/local_detector_analysis"
    model_stem = f"{problem_formulation}_{preproc_approach}_{model_type}"
    model_path = model_dir / f"{model_stem}.joblib"
    settings_path = model_dir / f"{model_stem}_settings.joblib"

    model_settings = ModelSettings(
        settings_path=settings_path,
        problem_formulation=problem_formulation,
        preproc_approach=preproc_approach,
        window_length=window_len,
        future_length=future_len,
        max_trace_length=max_trace_length,
        model_type=model_type,
        model_path=model_path,
        new_model=True,
        plot=True,
    )

    benign_path = malware_path = DATA_PATH / "trace_data/syscall_bucket"

    malware_dict = config.SYSCALL_MALWARE_DICT
    benign_malware_dict = config.SYSCALL_BENIGN_MALWARE_DICT

    benign_dict = {
        k: benign_malware_dict[k] for k in benign_malware_dict if k not in malware_dict
    }

    malware_dict = {i: val for i, val in enumerate(malware_dict.values())}
    offset = len(malware_dict)
    benign_dict = {i + offset: val for i, val in enumerate(benign_dict.values())}

    return model_settings, benign_path, benign_dict, malware_path, malware_dict


if __name__ == "__main__":
    model_settings, benign_path, benign_dict, malware_path, malware_dict = (
        get_default_config()
    )
    run_local_detector(
        model_settings, benign_path, benign_dict, malware_path, malware_dict
    )
