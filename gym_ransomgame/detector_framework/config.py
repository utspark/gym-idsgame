from typing import Final, Optional
import random
import os

SUBSAMPLE_NETWORK_DATA : Final = 10
TRAIN_TEST_SPLIT : Final = 0.70
RANDOM_SEED : Final = 42
WINDOW_SIZE_TIME: Final = 0.7
WINDOW_STRIDE_TIME: Final = 0.1

def set_seed(seed: Optional[int] = RANDOM_SEED):
    """
    Globally set all random seeds for reproducibility.
    """
    if seed is None:
        return

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # For deterministic operations in some TensorFlow versions/operations
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # For deterministic operations in PyTorch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

SYSCALL_BENIGN_MALWARE_DICT: Final = {
    0: [
        # "recon_system_",
        "recon_mount_",
        "recon_net_",
    ],
    1: [
        "compress_gzip_1t_",
    ],
    2: [
        "compress_gzip_8t_",
    ],
    3: [
        "compress_zstd_1t_",
        "compress_zstd_8t_",
    ],
    4: [
        "transfer_aws_1t_",
        "transfer_aws_8t_",
    ],
    5: [
        "transfer_sftp_1t_",
        "transfer_sftp_8t_",
    ],
    6: [
        "symm_AES_128b_",
        "symm_AES_256b_",
        "symm_Salsa20_128b_",
        "symm_Salsa20_256b_",
    ],
    7: [
        "browser_compute_",
    ],
    8: [
        "browser_download_",
        "browser_mix_",
        "browser_generic_",
        "browser_streaming_",
        "filebench_oltp_",
    ],
    9: [
        "filebench_fileserver_",
        "filebench_videoserver_",
    ],
    10: [
        "filebench_randomrw_",
    ],
    11: [
        "filebench_varmail_",
    ],
    12: [
        "mediaserver_browse_",
    ],
    13: [
        "mediaserver_index_",
    ],
    14: [
        "spec_gcc_",
        "spec_leela_",
        "spec_deepsjeng_",
    ],
    15: [
        "idle_",
    ],
}

SYSCALL_MALWARE_DICT: Final = {
    0: [
        # "recon_system_",
        "recon_mount_",
        "recon_net_",
    ],
    1: [
        "compress_gzip_1t_",
    ],
    2: [
        "compress_gzip_8t_",
    ],
    3: [
        "compress_zstd_1t_",
        "compress_zstd_8t_",
    ],
    4: [
        "transfer_aws_1t_",
        "transfer_aws_8t_",
    ],
    5: [
        "transfer_sftp_1t_",
        "transfer_sftp_8t_",
    ],
    6: [
        "symm_AES_128b_",
        "symm_AES_256b_",
        "symm_Salsa20_128b_",
        "symm_Salsa20_256b_",
    ],
}

NETWORK_BENIGN_MALWARE_DICT: Final = {
    0: [
        # "recon_system_",
        "recon_mount_",
    ],
    1: [
        "recon_net_",
    ],
    2: [
        "compress_gzip_1t_",
        "compress_gzip_8t_",
        "compress_zstd_1t_",
        "compress_zstd_8t_"
    ],
    3: [
        "transfer_aws_1t_",
        "transfer_aws_8t_",
        "transfer_sftp_1t_",
        "transfer_sftp_8t_"
    ],
    4: [
        "symm_AES_128b_",
        "symm_AES_256b_",
        "symm_Salsa20_128b_",
        "symm_Salsa20_256b_",
    ],
    5: [
        "browser_netcall_compute_",
        "browser_netcall_streaming_",
        "browser_netcall_generic_",
        "browser_netcall_mix_",
        "browser_netcall_download_",
        "mediaserver_index_",
    ],
    6: [
        "mediaserver_browse_",
    ],
    7: [
        "idle_",
    ],
}

NETWORK_MALWARE_DICT : Final = {
    0: [
        # "recon_system_",
        "recon_mount_",
    ],
    1: [
        "recon_net_",
    ],
    2: [
        "compress_gzip_1t_",
        "compress_gzip_8t_",
        "compress_zstd_1t_",
        "compress_zstd_8t_"
    ],
    3: [
        "transfer_aws_1t_",
        "transfer_aws_8t_",
        "transfer_sftp_1t_",
        "transfer_sftp_8t_"
    ],
    4: [
        "symm_AES_128b_",
        "symm_AES_256b_",
        "symm_Salsa20_128b_",
        "symm_Salsa20_256b_",
    ],
}

HPC_BENIGN_MALWARE_DICT : Final = {
    0: [
        # "recon_system_",
        "recon_net_",
    ],
    1: [
        "recon_mount_",
    ],
    2: [
        "compress_gzip_1t_",
        "compress_gzip_8t_",
    ],
    3: [
        "compress_zstd_1t_",
        "compress_zstd_8t_",
    ],
    4: [
        "transfer_aws_1t_",
        "transfer_aws_8t_",
        "transfer_sftp_1t_",
        "transfer_sftp_8t_"
    ],
    5: [
        "symm_AES_128b_",
        "symm_AES_256b_",
        "symm_Salsa20_128b_",
        "symm_Salsa20_256b_",
    ],
    6: [
        "browser_compute_",
    ],
    7: [
        "browser_download_",
        "browser_generic_",
        "browser_streaming_",
        "browser_mix_",
        "filebench_fileserver_",
        "filebench_oltp_",
        "filebench_randomrw_",
        "filebench_varmail_",
        "filebench_videoserver_",
    ],
    8: [
        "mediaserver_browse_",
    ],
    9: [
        "mediaserver_index_",
    ],
    10: [
        "spec_deepsjeng_",
    ],
    11: [
        "spec_gcc_",
    ],
    12: [
        "spec_leela_",
    ],
    13: [
        "idle_",
    ],
}

HPC_MALWARE_DICT : Final = {
    0: [
        # "recon_system_",
        "recon_net_",
    ],
    1: [
        "recon_mount_",
    ],
    2: [
        "compress_gzip_1t_",
        "compress_gzip_8t_",
    ],
    3: [
        "compress_zstd_1t_",
        "compress_zstd_8t_",
    ],
    4: [
        "transfer_aws_1t_",
        "transfer_aws_8t_",
        "transfer_sftp_1t_",
        "transfer_sftp_8t_"
    ],
    5: [
        "symm_AES_128b_",
        "symm_AES_256b_",
        "symm_Salsa20_128b_",
        "symm_Salsa20_256b_",
    ],
}


SYSCALL_MALWARE_CLASS_TRANSLATION: Final = {
    -1: -1,
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 3,
}

SYSCALL_BENIGN_MALWARE_CLASS_TRANSLATION: Final = {
    -1: -1,
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 3,
    7: -1,
    8: -1,
    9: -1,
    10: -1,
    11: -1,
    12: -1,
    13: -1,
    14: -1,
    15: -1,
}

NETWORK_MALWARE_CLASS_TRANSLATION: Final = {
    -1: -1,
    0: 0,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
}

NETWORK_BENIGN_MALWARE_CLASS_TRANSLATION: Final = {
    -1: -1,
    0: 0,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: -1,
    6: -1,
    7: -1,
}

HPC_MALWARE_CLASS_TRANSLATION: Final = {
    -1: -1,
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 2,
    5: 3,
}

HPC_BENIGN_MALWARE_CLASS_TRANSLATION: Final = {
    -1: -1,
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 2,
    5: 3,
    6: -1,
    7: -1,
    8: -1,
    9: -1,
    10: -1,
    11: -1,
    12: -1,
    13: -1,
}

BEHAVIOR_FILES: Final = {
    "browser_compute": {
        "syscall": [
            "browser_compute_1_ints.txt",
            "browser_compute_2_ints.txt",
            "browser_compute_3_ints.txt",
            "browser_compute_4_ints.txt",
            "browser_compute_5_ints.txt",
        ],
        "network": [
            "browser_compute_1",
            "browser_compute_2",
            "browser_compute_3",
            "browser_compute_4",
            "browser_compute_5",
        ],
        "hpc": [
            "browser_compute_1",
            "browser_compute_2",
            "browser_compute_3",
            "browser_compute_4",
            "browser_compute_5",
        ],
    },
    "browser_download": {
        "syscall": [
            "browser_download_1_ints.txt",
            "browser_download_2_ints.txt",
            "browser_download_3_ints.txt",
            "browser_download_4_ints.txt",
            "browser_download_5_ints.txt",
        ],
        "network": [
            "browser_download_1",
            "browser_download_2",
            "browser_download_3",
            "browser_download_4",
            "browser_download_5",
        ],
        "hpc": [
            "browser_download_1",
            "browser_download_2",
            "browser_download_3",
            "browser_download_4",
            "browser_download_5",
        ],
    },
    "browser_generic": {
        "syscall": [
            "browser_generic_1_ints.txt",
            "browser_generic_2_ints.txt",
            "browser_generic_3_ints.txt",
            "browser_generic_4_ints.txt",
            "browser_generic_5_ints.txt",
        ],
        "network": [
            "browser_generic_1",
            "browser_generic_2",
            "browser_generic_3",
            "browser_generic_4",
            "browser_generic_5",
        ],
        "hpc": [
            "browser_generic_1",
            "browser_generic_2",
            "browser_generic_3",
            "browser_generic_4",
            "browser_generic_5",
        ],
    },
    "browser_mix": {
        "syscall": [
            "browser_mix_1_ints.txt",
            "browser_mix_2_ints.txt",
            "browser_mix_3_ints.txt",
            "browser_mix_4_ints.txt",
            "browser_mix_5_ints.txt",
        ],
        "network": [
            "browser_mix_1",
            "browser_mix_2",
            "browser_mix_3",
            "browser_mix_4",
            "browser_mix_5",
        ],
        "hpc": [
            "browser_mix_1",
            "browser_mix_2",
            "browser_mix_3",
            "browser_mix_4",
            "browser_mix_5",
        ],
    },
    "browser_streaming": {
        "syscall": [
            "browser_streaming_1_ints.txt",
            "browser_streaming_2_ints.txt",
            "browser_streaming_3_ints.txt",
            "browser_streaming_4_ints.txt",
            "browser_streaming_5_ints.txt",
        ],
        "network": [
            "browser_streaming_1",
            "browser_streaming_2",
            "browser_streaming_3",
            "browser_streaming_4",
            "browser_streaming_5",
        ],
        "hpc": [
            "browser_streaming_1",
            "browser_streaming_2",
            "browser_streaming_3",
            "browser_streaming_4",
            "browser_streaming_5",
        ],
    },
    "filebench_fileserver": {
        "syscall": [
            "filebench_fileserver_1_ints.txt",
            "filebench_fileserver_2_ints.txt",
            "filebench_fileserver_3_ints.txt",
        ],
        "network": [
            "",
            "",
        ],
        "hpc": [
            "filebench_fileserver_1",
            "filebench_fileserver_2",
            "filebench_fileserver_3",
        ],
    },
    "filebench_oltp": {
        "syscall": [
            "filebench_oltp_1_ints.txt",
            "filebench_oltp_2_ints.txt",
            "filebench_oltp_3_ints.txt",
        ],
        "network": [
            "",
            "",
        ],
        "hpc": [
            "filebench_oltp_1",
            "filebench_oltp_2",
            "filebench_oltp_3",
        ],
    },
    "filebench_randomrw": {
        "syscall": [
            "filebench_randomrw_1_ints.txt",
            "filebench_randomrw_2_ints.txt",
            "filebench_randomrw_3_ints.txt",
        ],
        "network": [
            "",
            "",
        ],
        "hpc": [
            "filebench_randomrw_1",
            "filebench_randomrw_2",
            "filebench_randomrw_3",
        ],
    },
    "filebench_varmail": {
        "syscall": [
            "filebench_varmail_1_ints.txt",
            "filebench_varmail_2_ints.txt",
            "filebench_varmail_3_ints.txt",
        ],
        "network": [
            "",
            "",
        ],
        "hpc": [
            "filebench_varmail_1",
            "filebench_varmail_2",
            "filebench_varmail_3",
        ],
    },
    "filebench_videoserver": {
        "syscall": [
            "filebench_videoserver_1_ints.txt",
            "filebench_videoserver_2_ints.txt",
            "filebench_videoserver_3_ints.txt",
        ],
        "network": [
            "",
            "",
        ],
        "hpc": [
            "filebench_videoserver_1",
            "filebench_videoserver_2",
            "filebench_videoserver_3",
        ],
    },
    "mediaserver_browse": {
        "syscall": [
            "mediaserver_browse_1_ints.txt",
            "mediaserver_browse_2_ints.txt",
            "mediaserver_browse_3_ints.txt",
            "mediaserver_browse_4_ints.txt",
            "mediaserver_browse_5_ints.txt",
        ],
        "network": [
            "mediaserver_browse_1",
            "mediaserver_browse_2",
            "mediaserver_browse_3",
            "mediaserver_browse_4",
            "mediaserver_browse_5",
        ],
        "hpc": [
            "mediaserver_browse_1",
            "mediaserver_browse_2",
            "mediaserver_browse_3",
            "mediaserver_browse_4",
            "mediaserver_browse_5",
        ],
    },
    "mediaserver_index": {
        "syscall": [
            "mediaserver_index_1_ints.txt",
            "mediaserver_index_2_ints.txt",
            "mediaserver_index_3_ints.txt",
            "mediaserver_index_4_ints.txt",
            "mediaserver_index_5_ints.txt",
        ],
        "network": [
            "mediaserver_index_1",
            "mediaserver_index_2",
            "mediaserver_index_3",
            "mediaserver_index_4",
            "mediaserver_index_5",
        ],
        "hpc": [
            "mediaserver_index_1",
            "mediaserver_index_2",
            "mediaserver_index_3",
            "mediaserver_index_4",
            "mediaserver_index_5",
        ],
    },
    "spec_leela": {
        "syscall": [
            "spec_leela_1_ints.txt",
            "spec_leela_2_ints.txt",
            "spec_leela_3_ints.txt",
        ],
        "network": [
            "",
            "",
        ],
        "hpc": [
            "spec_leela_1",
            "spec_leela_2",
            "spec_leela_3",
        ],
    },
    "spec_gcc": {
        "syscall": [
            "spec_gcc_1_ints.txt",
            "spec_gcc_2_ints.txt",
            "spec_gcc_3_ints.txt",
        ],
        "network": [
            "",
            "",
        ],
        "hpc": [
            "spec_gcc_1",
            "spec_gcc_2",
            "spec_gcc_3",
        ],
    },
    "spec_deepsjeng": {
        "syscall": [
            "spec_deepsjeng_1_ints.txt",
            "spec_deepsjeng_2_ints.txt",
            "spec_deepsjeng_3_ints.txt",
        ],
        "network": [
            "",
            "",
        ],
        "hpc": [
            "spec_deepsjeng_1",
            "spec_deepsjeng_2",
            "spec_deepsjeng_3",
        ],
    },
    "idle": {
            "syscall": [
                "idle_1_ints.txt",
                "idle_2_ints.txt",
                "idle_3_ints.txt",
                "idle_4_ints.txt",
                "idle_5_ints.txt",
            ],
            "network": [
                "idle_1",
                "idle_2",
                "idle_3",
                "idle_4",
                "idle_5",
            ],
            "hpc": [
                "idle_1",
                "idle_2",
                "idle_3",
                "idle_4",
                "idle_5",
            ],
        },

    "recon_mount": {
        "syscall": [
            "recon_mount_1_ints.txt",
            "recon_mount_2_ints.txt",
            "recon_mount_3_ints.txt",
            "recon_mount_4_ints.txt",
            "recon_mount_5_ints.txt",
        ],
        "network": [
            "recon_mount_1",
            "recon_mount_2",
            "recon_mount_3",
            "recon_mount_4",
            "recon_mount_5",
        ],
        "hpc": [
            "recon_mount_1",
            "recon_mount_2",
            "recon_mount_3",
        ],
    },
    "recon_net": {
        "syscall": [
            "recon_net_1_ints.txt",
            "recon_net_2_ints.txt",
            "recon_net_3_ints.txt",
            "recon_net_4_ints.txt",
            "recon_net_5_ints.txt",
        ],
        "network": [
            "recon_net_1",
            "recon_net_2",
            "recon_net_3",
            "recon_net_4",
            "recon_net_5",
        ],
        "hpc": [
            "recon_net_1",
            "recon_net_2",
            "recon_net_3",
        ],
    },
    # "recon_system": {
    #     "syscall": [
    #         "recon_system_1_ints.txt",
    #         "recon_system_5_ints.txt",
    #     ],
    #     "network": [
    #         "recon_system_1",
    #         "recon_system_5",
    #     ],
    #     "hpc": [
    #         "recon_system_1",
    #     ],
    # },

    "compress_gzip_1t": {
        "syscall": [
            "compress_gzip_1t_0_ints.txt",
            "compress_gzip_1t_1_ints.txt",
            "compress_gzip_1t_2_ints.txt",
            "compress_gzip_1t_3_ints.txt",
            "compress_gzip_1t_4_ints.txt",
            "compress_gzip_1t_5_ints.txt",
            "compress_gzip_1t_6_ints.txt",
            "compress_gzip_1t_7_ints.txt",
            "compress_gzip_1t_8_ints.txt",
            "compress_gzip_1t_9_ints.txt",
        ],
        "network": [
            "compress_gzip_1t_0",
            "compress_gzip_1t_1",
            "compress_gzip_1t_2",
            "compress_gzip_1t_3",
            "compress_gzip_1t_4",
            "compress_gzip_1t_5",
            "compress_gzip_1t_6",
            "compress_gzip_1t_7",
            "compress_gzip_1t_8",
            "compress_gzip_1t_9",
        ],
        "hpc": [
            "compress_gzip_1t_0",
            "compress_gzip_1t_1",
            "compress_gzip_1t_2",
            "compress_gzip_1t_3",
            "compress_gzip_1t_4",
            "compress_gzip_1t_5",
        ],
    },
    "compress_gzip_8t": {
        "syscall": [
            "compress_gzip_8t_0_ints.txt",
            "compress_gzip_8t_1_ints.txt",
            "compress_gzip_8t_2_ints.txt",
            "compress_gzip_8t_3_ints.txt",
            "compress_gzip_8t_4_ints.txt",
            "compress_gzip_8t_5_ints.txt",
            "compress_gzip_8t_6_ints.txt",
            "compress_gzip_8t_7_ints.txt",
            "compress_gzip_8t_8_ints.txt",
            "compress_gzip_8t_9_ints.txt",
        ],
        "network": [
            "compress_gzip_8t_0",
            "compress_gzip_8t_1",
            "compress_gzip_8t_2",
            "compress_gzip_8t_3",
            "compress_gzip_8t_4",
            "compress_gzip_8t_5",
            "compress_gzip_8t_6",
            "compress_gzip_8t_7",
            "compress_gzip_8t_8",
            "compress_gzip_8t_9",
        ],
        "hpc": [
            "compress_gzip_8t_0",
            "compress_gzip_8t_1",
            "compress_gzip_8t_2",
            "compress_gzip_8t_3",
            "compress_gzip_8t_4",
            "compress_gzip_8t_5",
        ],
    },
    "compress_zstd_1t": {
        "syscall": [
            "compress_zstd_1t_0_ints.txt",
            "compress_zstd_1t_1_ints.txt",
            "compress_zstd_1t_2_ints.txt",
            "compress_zstd_1t_3_ints.txt",
            "compress_zstd_1t_4_ints.txt",
            "compress_zstd_1t_5_ints.txt",
            "compress_zstd_1t_6_ints.txt",
            "compress_zstd_1t_7_ints.txt",
            "compress_zstd_1t_8_ints.txt",
            "compress_zstd_1t_9_ints.txt",
        ],
        "network": [
            "compress_zstd_1t_0",
            "compress_zstd_1t_1",
            "compress_zstd_1t_2",
            "compress_zstd_1t_3",
            "compress_zstd_1t_4",
            "compress_zstd_1t_5",
            "compress_zstd_1t_6",
            "compress_zstd_1t_7",
            "compress_zstd_1t_8",
            "compress_zstd_1t_9",
        ],
        "hpc": [
            "compress_zstd_1t_0",
            "compress_zstd_1t_1",
            "compress_zstd_1t_2",
            "compress_zstd_1t_3",
            "compress_zstd_1t_4",
            "compress_zstd_1t_5",
        ],
    },
    "compress_zstd_8t": {
        "syscall": [
            "compress_zstd_8t_0_ints.txt",
            "compress_zstd_8t_1_ints.txt",
            "compress_zstd_8t_2_ints.txt",
            "compress_zstd_8t_3_ints.txt",
            "compress_zstd_8t_4_ints.txt",
            "compress_zstd_8t_5_ints.txt",
            "compress_zstd_8t_6_ints.txt",
            "compress_zstd_8t_7_ints.txt",
            "compress_zstd_8t_8_ints.txt",
            "compress_zstd_8t_9_ints.txt",
        ],
        "network": [
            "compress_zstd_8t_0",
            "compress_zstd_8t_1",
            "compress_zstd_8t_2",
            "compress_zstd_8t_3",
            "compress_zstd_8t_4",
            "compress_zstd_8t_5",
            "compress_zstd_8t_6",
            "compress_zstd_8t_7",
            "compress_zstd_8t_8",
            "compress_zstd_8t_9",
        ],
        "hpc": [
            "compress_zstd_8t_0",
            "compress_zstd_8t_1",
            "compress_zstd_8t_2",
            "compress_zstd_8t_3",
            "compress_zstd_8t_4",
            "compress_zstd_8t_5",
        ],
    },
    "transfer_aws_1t": {
        "syscall": [
            "transfer_aws_1t_0_ints.txt",
            "transfer_aws_1t_1_ints.txt",
            "transfer_aws_1t_2_ints.txt",
            "transfer_aws_1t_3_ints.txt",
            "transfer_aws_1t_4_ints.txt",
            "transfer_aws_1t_5_ints.txt",
            "transfer_aws_1t_6_ints.txt",
            "transfer_aws_1t_7_ints.txt",
            "transfer_aws_1t_8_ints.txt",
            "transfer_aws_1t_9_ints.txt",
            "transfer_aws_1t_10_ints.txt",
            "transfer_aws_1t_11_ints.txt",
        ],
        "network": [
            "transfer_aws_1t_0",
            "transfer_aws_1t_1",
            "transfer_aws_1t_2",
            "transfer_aws_1t_3",
            "transfer_aws_1t_4",
            "transfer_aws_1t_5",
            "transfer_aws_1t_6",
            "transfer_aws_1t_7",
            "transfer_aws_1t_8",
            "transfer_aws_1t_9",
            "transfer_aws_1t_10",
            "transfer_aws_1t_11",
            "transfer_aws_1t_12",
            "transfer_aws_1t_13",
            "transfer_aws_1t_14",
        ],
        "hpc": [
            "transfer_aws_1t_0",
            "transfer_aws_1t_1",
            "transfer_aws_1t_2",
            "transfer_aws_1t_3",
            "transfer_aws_1t_4",
            "transfer_aws_1t_5",
        ],
    },
    "transfer_aws_8t": {
        "syscall": [
            "transfer_aws_8t_0_ints.txt",
            "transfer_aws_8t_1_ints.txt",
            "transfer_aws_8t_2_ints.txt",
            "transfer_aws_8t_3_ints.txt",
            "transfer_aws_8t_4_ints.txt",
            "transfer_aws_8t_5_ints.txt",
            "transfer_aws_8t_6_ints.txt",
            "transfer_aws_8t_7_ints.txt",
            "transfer_aws_8t_8_ints.txt",
            "transfer_aws_8t_9_ints.txt",
            "transfer_aws_8t_10_ints.txt",
            "transfer_aws_8t_11_ints.txt",
        ],
        "network": [
            "transfer_aws_8t_0",
            "transfer_aws_8t_1",
            "transfer_aws_8t_2",
            "transfer_aws_8t_3",
            "transfer_aws_8t_4",
            "transfer_aws_8t_5",
            "transfer_aws_8t_6",
            "transfer_aws_8t_7",
            "transfer_aws_8t_8",
            "transfer_aws_8t_9",
            "transfer_aws_8t_10",
            "transfer_aws_8t_11",
            "transfer_aws_8t_12",
            "transfer_aws_8t_13",
            "transfer_aws_8t_14",
        ],
        "hpc": [
            "transfer_aws_8t_0",
            "transfer_aws_8t_1",
            "transfer_aws_8t_2",
            "transfer_aws_8t_3",
            "transfer_aws_8t_4",
            "transfer_aws_8t_5",
        ],
    },
    "transfer_sftp_1t": {
        "syscall": [
            "transfer_sftp_1t_0_ints.txt",
            "transfer_sftp_1t_1_ints.txt",
            "transfer_sftp_1t_2_ints.txt",
            "transfer_sftp_1t_3_ints.txt",
            "transfer_sftp_1t_4_ints.txt",
            "transfer_sftp_1t_5_ints.txt",
            "transfer_sftp_1t_6_ints.txt",
            "transfer_sftp_1t_7_ints.txt",
            "transfer_sftp_1t_8_ints.txt",
            "transfer_sftp_1t_9_ints.txt",
            "transfer_sftp_1t_10_ints.txt",
            "transfer_sftp_1t_11_ints.txt",
        ],
        "network": [
            "transfer_sftp_1t_0",
            "transfer_sftp_1t_1",
            "transfer_sftp_1t_2",
            "transfer_sftp_1t_3",
            "transfer_sftp_1t_4",
            "transfer_sftp_1t_5",
            "transfer_sftp_1t_6",
            "transfer_sftp_1t_7",
            "transfer_sftp_1t_8",
            "transfer_sftp_1t_9",
            "transfer_sftp_1t_10",
            "transfer_sftp_1t_11",
            "transfer_sftp_1t_12",
            "transfer_sftp_1t_13",
            "transfer_sftp_1t_14",
        ],
        "hpc": [
            "transfer_sftp_1t_0",
            "transfer_sftp_1t_1",
            "transfer_sftp_1t_2",
            "transfer_sftp_1t_3",
            "transfer_sftp_1t_4",
            "transfer_sftp_1t_5",
        ],
    },
    "transfer_sftp_8t": {
        "syscall": [
            "transfer_sftp_8t_0_ints.txt",
            "transfer_sftp_8t_1_ints.txt",
            "transfer_sftp_8t_2_ints.txt",
            "transfer_sftp_8t_3_ints.txt",
            "transfer_sftp_8t_4_ints.txt",
            "transfer_sftp_8t_5_ints.txt",
            "transfer_sftp_8t_6_ints.txt",
            "transfer_sftp_8t_7_ints.txt",
            "transfer_sftp_8t_8_ints.txt",
            "transfer_sftp_8t_9_ints.txt",
            "transfer_sftp_8t_10_ints.txt",
            "transfer_sftp_8t_11_ints.txt",
        ],
        "network": [
            "transfer_sftp_8t_0",
            "transfer_sftp_8t_1",
            "transfer_sftp_8t_2",
            "transfer_sftp_8t_3",
            "transfer_sftp_8t_4",
            "transfer_sftp_8t_5",
            "transfer_sftp_8t_6",
            "transfer_sftp_8t_7",
            "transfer_sftp_8t_8",
            "transfer_sftp_8t_9",
            "transfer_sftp_8t_10",
            "transfer_sftp_8t_11",
            "transfer_sftp_8t_12",
            "transfer_sftp_8t_13",
            "transfer_sftp_8t_14",
        ],
        "hpc": [
            "transfer_sftp_8t_0",
            "transfer_sftp_8t_1",
            "transfer_sftp_8t_2",
            "transfer_sftp_8t_3",
            "transfer_sftp_8t_4",
            "transfer_sftp_8t_5",
        ],
    },

    "symm_AES_128b": {
        "syscall": [
            "symm_AES_128b_0_ints.txt",
            "symm_AES_128b_1_ints.txt",
            "symm_AES_128b_2_ints.txt",
            "symm_AES_128b_3_ints.txt",
            "symm_AES_128b_4_ints.txt",
            "symm_AES_128b_5_ints.txt",
            "symm_AES_128b_6_ints.txt",
            "symm_AES_128b_7_ints.txt",
            "symm_AES_128b_8_ints.txt",
            "symm_AES_128b_9_ints.txt",
            "symm_AES_128b_10_ints.txt",
            "symm_AES_128b_11_ints.txt",
        ],
        "network": [
            "symm_AES_128b_0",
            "symm_AES_128b_1",
            "symm_AES_128b_2",
            "symm_AES_128b_3",
        ],
        "hpc": [
            "symm_AES_128b_0",
            "symm_AES_128b_1",
            "symm_AES_128b_2",
            "symm_AES_128b_3",
            "symm_AES_128b_4",
            "symm_AES_128b_5",
            "symm_AES_128b_6",
            "symm_AES_128b_7",
            "symm_AES_128b_8",
            "symm_AES_128b_9",
            "symm_AES_128b_10",
            "symm_AES_128b_11",
        ],
    },
    "symm_AES_256b": {
        "syscall": [
            "symm_AES_256b_0_ints.txt",
            "symm_AES_256b_1_ints.txt",
            "symm_AES_256b_2_ints.txt",
            "symm_AES_256b_3_ints.txt",
            "symm_AES_256b_4_ints.txt",
            "symm_AES_256b_5_ints.txt",
            "symm_AES_256b_6_ints.txt",
            "symm_AES_256b_7_ints.txt",
            "symm_AES_256b_8_ints.txt",
            "symm_AES_256b_9_ints.txt",
            "symm_AES_256b_10_ints.txt",
            "symm_AES_256b_11_ints.txt",
        ],
        "network": [
            "symm_AES_256b_0",
            "symm_AES_256b_1",
            "symm_AES_256b_2",
            "symm_AES_256b_3",
        ],
        "hpc": [
            "symm_AES_256b_0",
            "symm_AES_256b_1",
            "symm_AES_256b_2",
            "symm_AES_256b_3",
            "symm_AES_256b_4",
            "symm_AES_256b_5",
            "symm_AES_256b_6",
            "symm_AES_256b_7",
            "symm_AES_256b_8",
            "symm_AES_256b_9",
            "symm_AES_256b_10",
            "symm_AES_256b_11",
        ],
    },
    "symm_Salsa20_128b": {
        "syscall": [
            "symm_Salsa20_128b_0_ints.txt",
            "symm_Salsa20_128b_1_ints.txt",
            "symm_Salsa20_128b_2_ints.txt",
            "symm_Salsa20_128b_3_ints.txt",
            "symm_Salsa20_128b_4_ints.txt",
            "symm_Salsa20_128b_5_ints.txt",
            "symm_Salsa20_128b_6_ints.txt",
            "symm_Salsa20_128b_7_ints.txt",
            "symm_Salsa20_128b_8_ints.txt",
            "symm_Salsa20_128b_9_ints.txt",
            "symm_Salsa20_128b_10_ints.txt",
            "symm_Salsa20_128b_11_ints.txt",
        ],
        "network": [
            "symm_Salsa20_128b_0",
            "symm_Salsa20_128b_1",
            "symm_Salsa20_128b_2",
            "symm_Salsa20_128b_3",
        ],
        "hpc": [
            "symm_Salsa20_128b_0",
            "symm_Salsa20_128b_1",
            "symm_Salsa20_128b_2",
            "symm_Salsa20_128b_3",
            "symm_Salsa20_128b_4",
            "symm_Salsa20_128b_5",
            "symm_Salsa20_128b_6",
            "symm_Salsa20_128b_7",
            "symm_Salsa20_128b_8",
            "symm_Salsa20_128b_9",
            "symm_Salsa20_128b_10",
            "symm_Salsa20_128b_11",
        ],
    },
    "symm_Salsa20_256b": {
        "syscall": [
            "symm_Salsa20_256b_0_ints.txt",
            "symm_Salsa20_256b_1_ints.txt",
            "symm_Salsa20_256b_2_ints.txt",
            "symm_Salsa20_256b_3_ints.txt",
            "symm_Salsa20_256b_4_ints.txt",
            "symm_Salsa20_256b_5_ints.txt",
            "symm_Salsa20_256b_6_ints.txt",
            "symm_Salsa20_256b_7_ints.txt",
            "symm_Salsa20_256b_8_ints.txt",
            "symm_Salsa20_256b_9_ints.txt",
            "symm_Salsa20_256b_10_ints.txt",
            "symm_Salsa20_256b_11_ints.txt",
        ],
        "network": [
            "symm_Salsa20_256b_0",
            "symm_Salsa20_256b_1",
            "symm_Salsa20_256b_2",
            "symm_Salsa20_256b_3",
        ],
        "hpc": [
            "symm_Salsa20_256b_0",
            "symm_Salsa20_256b_1",
            "symm_Salsa20_256b_2",
            "symm_Salsa20_256b_3",
            "symm_Salsa20_256b_4",
            "symm_Salsa20_256b_5",
            "symm_Salsa20_256b_6",
            "symm_Salsa20_256b_7",
            "symm_Salsa20_256b_8",
            "symm_Salsa20_256b_9",
            "symm_Salsa20_256b_10",
            "symm_Salsa20_256b_11",
        ],
    },
}

GENERATION_ATTACK_STAGES: Final = {
    "recon": [
        "recon_mount",
        "recon_net",
        # "recon_system",
    ],
    "exfil_1": [
        # "fscan_group",

        "compress_gzip_1t",
        "compress_gzip_8t",
        "compress_zstd_1t",
        "compress_zstd_8t",
    ],
    "exfil_2": [
        "transfer_aws_1t",
        "transfer_aws_8t",
        "transfer_sftp_1t",
        "transfer_sftp_8t",
    ],
    # "exec_1": [
    #     "asymm",
    #
    # ],
    "exec_2": [
        "symm_AES_128b",
        "symm_AES_256b",
        "symm_Salsa20_128b",
        "symm_Salsa20_256b",
    ],
}

GENERATION_ATTACK_STAGES_REVIL: Final = {
    "recon": [
        "recon_mount",
        "recon_net",
        # "recon_system",
    ],
    "exfil_2": [
        "transfer_sftp_1t",
        "transfer_sftp_8t",
    ],
    "exec_2": [
        "symm_AES_256b",
        "symm_Salsa20_256b",
    ],
}

GENERATION_ATTACK_STAGES_RANSOMHUB: Final = {
    "recon": [
        "recon_mount",
        "recon_net",
        # "recon_system",
    ],
    "exfil_2": [
        "transfer_aws_1t",
        "transfer_aws_8t",
    ],
    "exec_2": [
        "symm_AES_256b",
    ],
}

GENERATION_ATTACK_STAGES_LOCKBIT: Final = {
    "recon": [
        "recon_mount",
        "recon_net",
        # "recon_system",
    ],
    "exfil_1": [
        "compress_gzip_1t",
        "compress_gzip_8t",
    ],
    "exfil_2": [
        "transfer_aws_1t",
        "transfer_aws_8t",
    ],
    "exec_2": [
        "symm_AES_256b",
    ],
}

GENERATION_BENIGN: Final = [
    "browser_compute",
    "browser_download",
    "browser_generic",
    "browser_mix",
    "browser_streaming",
    "filebench_fileserver",
    "filebench_oltp",
    "filebench_randomrw",
    "filebench_varmail",
    "filebench_videoserver",
    "mediaserver_browse",
    "mediaserver_index",
    "spec_gcc",
    "spec_leela",
    "spec_deepsjeng",
    "idle",
]

GENERATION_BENIGN_ENCRYPTION: Final = [
    "browser_compute",
    "browser_download",
    "browser_generic",
    "browser_mix",
    "browser_streaming",
    "filebench_fileserver",
    "filebench_oltp",
    "filebench_randomrw",
    "filebench_varmail",
    "filebench_videoserver",
    "mediaserver_browse",
    "mediaserver_index",
    "spec_gcc",
    "spec_leela",
    "spec_deepsjeng",
    "idle",
    "symm_AES_128b",
    "symm_AES_256b",
    "symm_Salsa20_128b",
    "symm_Salsa20_256b",
]


GENERATION_BENIGN_ENCRYPTION_ONLY: Final = [
    "symm_AES_128b",
    "symm_AES_256b",
    "symm_Salsa20_128b",
    "symm_Salsa20_256b",
]


