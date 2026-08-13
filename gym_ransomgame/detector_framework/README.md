# Detector Framework

Welcome to the **Detector Framework documentation**.

## Prerequisites
- Python 3.11

## Key Features

*   Data Processing
*   Local Detector
*   Global Detector
*   Cross-Layer Detector

## Quick Start

### 1.) Process Dataset and Correct syscall data if necessary
1. To download and setup the dataset execute `setup_data.sh` under `data/` directory.
2. [Optional] Move to `detector_framework/data_processing/` and execute `python strace_processor.py <PATH>` with the path to the file holding raw ftrace files. Do this if you need to process individually collected new syscall traces.
3. Execute `setup_data.sh` to create appropriate `_bucket` folders under `detector_framework/data`.

```Bash
cd data/
./setup_data.sh

```

### 2.) Ensure data is placed in the correct directory

    data/trace_data/hpc_bucket/
    data/trace_data/network_bucket/
    data/trace_data/syscall_bucket/

### 3.) Analyze Local and Global Detectors [Optional]
1. Execute `local_detector_run.py` and `global_detector_run.py` from the top level folder (ransomware-setups).
2. If you face python module errors, import appropriate modules using `pip install` or run the `requirements.txt` file on the top level folder. Use Python 3.11.

```Bash
pip install -r requirements.txt
python -m detector_framework.local_detector.local_detector_run
python -m detector_framework.global_detector.global_detector_run
```

### 4.) Run the Cross-Layer Detector

```Bash
python -m detector_framework.cross_layer.cross_layer_train_run
```


project-name/
├── src/              # Source code
├── tests/            # Unit tests
├── docs/             # Documentation
├── examples/         # Usage examples
├── requirements.txt  # Python dependencies
└── README.md        # This file
