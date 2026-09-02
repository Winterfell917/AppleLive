"""Runtime configuration for the compact MocapPipe live-demo project."""

from pathlib import Path

import torch


class train_hypers:
    """Values retained because the Lightning model classes reference them."""

    batch_size = 256
    num_workers = 8
    num_epochs = 60
    accelerator = "gpu"
    device = 0
    lr = 1e-3


class finetune_hypers(train_hypers):
    batch_size = 32
    num_epochs = 15
    lr = 5e-5


class paths:
    package_dir = Path(__file__).resolve().parent
    data_dir = package_dir / "data"
    checkpoint = data_dir / "checkpoints"
    record_dir = data_dir / "records"
    smpl_file = package_dir / "smpl/basicmodel_m.pkl"
    weights_file = checkpoint / "base_model_12combo.pth"


class model_config:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    n_joints = 5
    n_imu = 12 * n_joints
    n_output_joints = 24
    n_pose_output = n_output_joints * 6
    past_frames = 40
    future_frames = 5
    total_frames = past_frames + future_frames


class amass:
    combos_full = {"lw_rp_h": [0, 3, 4]}
    acc_scale = 30
    vel_scale = 2


class datasets:
    fps = 30
    window_length = 125


class joint_set:
    gravity_velocity = -0.018
    full = list(range(24))
    reduced = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]
    n_full = len(full)
    n_ignored = len(ignored)
    n_reduced = len(reduced)
    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]


class HuaweiDevices:
    device_ids = {
        "Left_Watch": 0,
        "Right_Phone": 3,
        "Head": 4,
    }
    time_offsets = [0] * 7
    BUFFER_SIZE = 50
