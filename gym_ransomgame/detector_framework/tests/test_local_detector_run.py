import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import joblib
import os
import sys

from detector_framework.data_processing.processing import ModelSettings
from detector_framework.local_detector.local_detector_run import run_local_detector, get_default_config

from detector_framework import config



def test_multiclass_analysis():
    # Ensure project root is in PYTHONPATH and change to it
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    config.set_seed()

    model_settings, benign_path, benign_dict, malware_path, malware_dict = get_default_config()
    model_settings.plot = False

    loss_ohe = run_local_detector(
        model_settings, benign_path, benign_dict, malware_path, malware_dict
    )

    assert loss_ohe == pytest.approx(0.97927, abs=1e-5)

