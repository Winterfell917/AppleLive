import torch
from pygame.time import Clock

import articulate as art
import os
import copy
from config import *
from articulate.utils.unity import MotionViewer

from utils.model_utils import load_model
import numpy as np
import matplotlib
from argparse import ArgumentParser
from sensor_huawei.sensor import CalibratedHuaweiSensor
import traceback
import datetime
from models.imu_calibrator import ComboTemporalIMUCalibrator, build_imu_input
from models.tic_calibrator import TICOnlineCalibrator, TICOperatorConfig, TICTransformerCalibrator

colors = matplotlib.colormaps['tab10'].colors
CALIBRATOR_COMBO = [0, 3, 4]


def load_combo_calibrator(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint.get("args", {})
    model = ComboTemporalIMUCalibrator(
        combo_size=len(CALIBRATOR_COMBO),
        predict_acc=args.get("predict_acc", False),
        hidden_dim=args.get("hidden_dim", 128),
        dropout=args.get("dropout", 0.1),
        num_layers=args.get("num_layers", 3),
        nhead=args.get("nhead", 4),
        max_seq_len=args.get("window_size", 125),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.reset()
    return model


def load_tic_calibrator(path: str, device: torch.device, buffer_size: int, trigger_t: float):
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint.get("args", {})
    model = TICTransformerCalibrator(
        imu_num=len(CALIBRATOR_COMBO),
        n_input=len(CALIBRATOR_COMBO) * 12,
        stack=args.get("stack", 4),
        multi_head=args.get("nhead", 8),
        d_model=args.get("d_model", 256),
        d_ff=args.get("d_ff", 512),
        dropout=args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    operator = TICOnlineCalibrator(
        model,
        imu_num=len(CALIBRATOR_COMBO),
        config=TICOperatorConfig(
            buffer_size=buffer_size,
            trigger_t=trigger_t,
            data_frame_rate=30,
            ego_idx=len(CALIBRATOR_COMBO) - 1,
        ),
    )
    operator.reset()
    return operator


def make_mocap_input(acc: torch.Tensor, ori: torch.Tensor):
    ori = ori[:model_config.n_joints].view(model_config.n_joints, 3, 3)
    acc = acc[:model_config.n_joints].view(model_config.n_joints, 3)
    acc = acc / amass.acc_scale
    return torch.cat([acc.flatten(), ori.flatten()], dim=0).to(model_config.device)


def apply_combo_calibrator(
    calibrator: ComboTemporalIMUCalibrator,
    acc: torch.Tensor,
    ori: torch.Tensor,
):
    calibrated_ori = ori.clone()
    calibrator_input = build_imu_input(
        acc.unsqueeze(0),
        ori.unsqueeze(0),
    )[0, CALIBRATOR_COMBO]
    _, pred_ori_combo = calibrator.forward_frame_windowed(calibrator_input)
    calibrated_ori[CALIBRATOR_COMBO] = pred_ori_combo
    return calibrated_ori


def apply_tic_calibrator(
    calibrator: TICOnlineCalibrator,
    acc: torch.Tensor,
    ori: torch.Tensor,
):
    calibrated_acc = acc.clone()
    calibrated_ori = ori.clone()
    pred_ori_combo, pred_acc_combo = calibrator.forward_frame(
        ori[CALIBRATOR_COMBO].detach().cpu(),
        acc[CALIBRATOR_COMBO].detach().cpu(),
    )
    calibrated_acc[CALIBRATOR_COMBO] = pred_acc_combo.to(acc.device)
    calibrated_ori[CALIBRATOR_COMBO] = pred_ori_combo.to(ori.device)
    return calibrated_acc, calibrated_ori


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--sub', type=str, default='chaoran_0529')
    parser.add_argument('--mocap', action='store_true', help='use mocap')
    parser.add_argument(
        '--ours-calibrator',
        type=str,
        default='data/checkpoints/combo_imu_calibrator_lw_rp_h_ori_only_jerk_nopose_fulltrain_tb_noncausal/best.pt',
        help='ours combo calibrator checkpoint for lw_rp_h (windowed non-causal version)',
    )
    parser.add_argument(
        '--tic-calibrator',
        type=str,
        default='data/checkpoints/tic_calibrator_amass_full/best.pt',
        help='TIC calibrator checkpoint',
    )
    parser.add_argument(
        '--calibrator-type',
        type=str,
        choices=['ours', 'tic'],
        default='ours',
        help='which calibrator to use for single-view mocap mode',
    )
    parser.add_argument(
        '--tic-buffer-size',
        type=int,
        default=128,
        help='TIC online buffer size',
    )
    parser.add_argument(
        '--tic-trigger-t',
        type=float,
        default=1.0,
        help='TIC dynamic calibration trigger interval in seconds',
    )
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='show live poses with w/o calibrator, ours calibrator, and TIC calibrator side by side',
    )
    args = parser.parse_args()

    device = model_config.device
    clock = Clock()

    if args.mocap:
        ckpt_path = "data/checkpoints/base_model_12combo.pth"
        net = load_model(ckpt_path)
        net.eval()
        if args.compare_all:
            raw_net = copy.deepcopy(net).eval()
            ours_net = copy.deepcopy(net).eval()
            tic_net = copy.deepcopy(net).eval()
            ours_calibrator = load_combo_calibrator(args.ours_calibrator, device)
            tic_calibrator = load_tic_calibrator(args.tic_calibrator, device, args.tic_buffer_size, args.tic_trigger_t)
            print(
                f'Mobileposer model loaded. Ours calibrator: {args.ours_calibrator}. '
                f'TIC calibrator: {args.tic_calibrator}'
            )
        else:
            raw_net = None
            ours_net = None
            tic_net = None
            if args.calibrator_type == 'ours':
                calibrator = load_combo_calibrator(args.ours_calibrator, device)
                print(f'Mobileposer model loaded. Ours calibrator: {args.ours_calibrator}')
            else:
                calibrator = load_tic_calibrator(args.tic_calibrator, device, args.tic_buffer_size, args.tic_trigger_t)
                print(f'Mobileposer model loaded. TIC calibrator: {args.tic_calibrator}')

    sensor = CalibratedHuaweiSensor(HuaweiDevices.device_ids)
    sensor.calibrate("walking_6dof")

    raw_accs, accs, oris, gyros, mags, pressures, ppgs, poses = [], [], [], [], [], [], [], []
    raw_poses, ours_poses, tic_poses = [], [], []
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    ids = sensor.ids

    print(f"Using device IDs: {ids}")

    idx = 0
    I, z = torch.eye(3).to(device), torch.zeros(3).to(device)
    viewer_count = 3 if args.mocap and args.compare_all else 1
    viewer_names = ['NoCalibrator', 'Ours', 'TIC'] if viewer_count == 3 else ['LiveDemo']
    with torch.no_grad(), MotionViewer(viewer_count, overlap=False, names=viewer_names) as viewer:
        while True:
            try:
                clock.tick(30)
                ori = torch.zeros(7, 3, 3).to(device)
                a = torch.zeros(7, 3).to(device)

                # device readings
                t, aS, aI, aM, RIS, RMB, gyro, mag, pressure, ppg = sensor.get()

                ori[ids] = RMB.to(device)
                a[ids] = aM.to(device)

                oris.append(ori.clone())
                accs.append(a.clone())
                raw_accs.append(aS)

                gyros.append(gyro)
                mags.append(mag)
                pressures.append(pressure)
                ppgs.append(ppg)

                if args.mocap:
                    zero_tran = np.array([0, 0, 0])
                    if args.compare_all:
                        raw_input = make_mocap_input(a, ori)
                        raw_pose = raw_net.forward_frame(raw_input)
                        ours_ori = apply_combo_calibrator(ours_calibrator, a, ori)
                        ours_input = make_mocap_input(a, ours_ori)
                        ours_pose = ours_net.forward_frame(ours_input)
                        tic_acc, tic_ori = apply_tic_calibrator(tic_calibrator, a, ori)
                        tic_input = make_mocap_input(tic_acc, tic_ori)
                        tic_pose = tic_net.forward_frame(tic_input)

                        raw_poses.append(raw_pose)
                        ours_poses.append(ours_pose)
                        tic_poses.append(tic_pose)
                        poses.append(ours_pose)
                        viewer.update_all(
                            [raw_pose.cpu().numpy(), ours_pose.cpu().numpy(), tic_pose.cpu().numpy()],
                            [zero_tran, zero_tran, zero_tran],
                            render=False,
                        )
                    else:
                        if args.calibrator_type == 'ours':
                            calibrated_ori = apply_combo_calibrator(calibrator, a, ori)
                            calibrated_input = make_mocap_input(a, calibrated_ori)
                            pose = net.forward_frame(calibrated_input)
                        else:
                            calibrated_acc, calibrated_ori = apply_tic_calibrator(calibrator, a, ori)
                            calibrated_input = make_mocap_input(calibrated_acc, calibrated_ori)
                            pose = net.forward_frame(calibrated_input)
                        poses.append(pose)
                        viewer.update_all([pose.cpu().numpy()], [zero_tran], render=False)
                    viewer.render()

                idx += 1

                print('\r', clock.get_fps(), end='')

            except Exception as e:
                print(f"Error occurred: {e}")
                print(traceback.format_exc())
                os._exit(0)
            except KeyboardInterrupt:
                print("Exiting...")
                os._exit(0)

    accs = torch.stack(accs)
    oris = torch.stack(oris)
    poses = torch.stack(poses) if poses else torch.empty(0)
    raw_poses = torch.stack(raw_poses) if raw_poses else torch.empty(0)
    ours_poses = torch.stack(ours_poses) if ours_poses else torch.empty(0)
    tic_poses = torch.stack(tic_poses) if tic_poses else torch.empty(0)
    raw_accs = torch.tensor(np.array(raw_accs))
    gyros = torch.tensor(np.array(gyros))
    mags = torch.tensor(np.array(mags))
    pressures = torch.tensor(np.array(pressures))
    ppgs = torch.tensor(np.array(ppgs))
    RMI, RSB, acc_bias = sensor.get_cali_matrices()

    print(f"raw_accs: {raw_accs.shape}, accs: {accs.shape}, oris: {oris.shape}, poses: {poses.shape}")
    if args.compare_all:
        print(f"raw poses: {raw_poses.shape}, ours poses: {ours_poses.shape}, tic poses: {tic_poses.shape}")
    print(f"gyros: {gyros.shape}, mags: {mags.shape}, pressures: {pressures.shape}, ppgs: {ppgs.shape}")
    print(f"RMI: {RMI.shape}, RSB: {RSB.shape}, acc_bias: {acc_bias.shape}")
    print('Frames: %d' % accs.shape[0])

    data_name = f"{args.name}_{timestamp}.pt"
    sub_name = args.sub

    save_dir = os.path.join(paths.record_dir, sub_name)
    os.makedirs(save_dir, exist_ok=True)
    record = {
        'raw_acc': raw_accs,
        'acc': accs,
        'ori': oris,
        'gyro': gyros,
        'mag': mags,
        'pressure': pressures,
        'ppg': ppgs,
        'pose': poses,
        'RMI': RMI,
        'RSB': RSB,
        'acc_bias': acc_bias,
    }
    if args.compare_all:
        record['pose_raw'] = raw_poses
        record['pose_ours'] = ours_poses
        record['pose_tic'] = tic_poses
    elif args.mocap:
        record['pose_calibrated'] = poses
    torch.save(record, os.path.join(save_dir, data_name))

    print('\rFinish.')
    os._exit(0)
