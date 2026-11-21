"""Run MobilePoser on a processed new_datasets sequence and render an MP4.

The input must first be manually reviewed and exported by
``prepare_new_datasets.py --export``.  ``orientation`` in labels.pt is R_MS;
this script applies the saved per-device R_SB to obtain R_MB, places both
devices in their ground-truth MobilePoser slots, runs the pose network, and
renders the SMPL mesh with pyrender.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import model_config, paths
from render_eval_pose import choose_device, infer_pose, render_video

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = BASE_DIR / 'data/new_datasets'
SLOT_INDEX = {'lw': 0, 'rw': 1, 'lp': 2, 'rp': 3}


def resolve_sequence(value: Path, root: Path) -> Path:
    if value.is_dir():
        return value.resolve()
    candidate = root / 'processed' / value
    if candidate.is_dir():
        return candidate.resolve()
    raise SystemExit(f'找不到已整理序列：{value}\n请先在 review.json 中批准边界并运行 prepare_new_datasets.py --export')


def proper_rotation(name: str, matrix: torch.Tensor, atol: float = 2e-3) -> None:
    identity = torch.eye(3, dtype=matrix.dtype)
    if not torch.allclose(matrix.transpose(-1, -2) @ matrix, identity.expand_as(matrix), atol=atol):
        raise RuntimeError(f'{name} 不是正交旋转矩阵')
    if bool((torch.det(matrix) < 0.99).any()):
        raise RuntimeError(f'{name} 含反射或退化矩阵')


def mobileposer_signals(payload: dict) -> tuple[torch.Tensor, torch.Tensor, int, int, dict]:
    devices = list(payload.get('devices', []))
    slots = list(payload.get('slots', []))
    if devices != ['apple_watch', 'iphone'] or len(slots) != 2:
        raise RuntimeError(f'设备顺序或槽位无效：devices={devices}, slots={slots}')
    try:
        slot_ids = [SLOT_INDEX[str(code)] for code in slots]
    except KeyError as error:
        raise RuntimeError(f'当前 MobilePoser 五槽模型不支持槽位 {error.args[0]}') from error
    if slot_ids[0] == slot_ids[1]:
        raise RuntimeError('两设备不能占用同一模型槽位')

    rms = payload['orientation'].float()       # [T, device, 3, 3]
    am = payload['acceleration'].float()       # [T, device, 3]
    if rms.ndim != 4 or rms.shape[1:] != (2, 3, 3) or am.shape != rms.shape[:2] + (3,):
        raise RuntimeError(f'输入形状无效：orientation={tuple(rms.shape)}, acceleration={tuple(am.shape)}')
    proper_rotation('R_MS', rms)

    calibration = payload.get('calibration')
    if not isinstance(calibration, dict):
        raise RuntimeError('labels.pt 缺少 calibration')
    rsb = []
    calibration_summary = {}
    for device in devices:
        item = calibration.get(device)
        if not isinstance(item, dict) or 'R_MI' not in item or 'R_SB' not in item:
            raise RuntimeError(f'缺少 {device} 的 R_MI/R_SB')
        if not item.get('passed', False):
            raise RuntimeError(f'{device} 标定质量未通过，拒绝渲染为可靠结果')
        rmi = torch.tensor(item['R_MI'], dtype=torch.float32)
        device_rsb = torch.tensor(item['R_SB'], dtype=torch.float32)
        proper_rotation(f'{device}.R_MI', rmi)
        proper_rotation(f'{device}.R_SB', device_rsb)
        rsb.append(device_rsb)
        calibration_summary[device] = {
            'R_MI': item['R_MI'], 'R_SB': item['R_SB'],
            'flat': item.get('flat'), 'npose': item.get('npose'),
        }
    rsb = torch.stack(rsb)
    rmb = rms @ rsb.unsqueeze(0)
    proper_rotation('R_MB', rmb)

    frame_count = len(rms)
    acceleration = torch.zeros(frame_count, model_config.n_joints, 3)
    orientation = torch.zeros(frame_count, model_config.n_joints, 3, 3)
    for device_index, slot in enumerate(slot_ids):
        acceleration[:, slot] = am[:, device_index]
        orientation[:, slot] = rmb[:, device_index]

    motion_valid = payload.get('motion_valid')
    if motion_valid is not None and not bool(torch.as_tensor(motion_valid).all()):
        bad = int((~torch.as_tensor(motion_valid).bool()).sum())
        raise RuntimeError(f'序列含 {bad} 个无效 IMU 设备帧，请先修复整理结果')
    return acceleration, orientation, slot_ids[0], slot_ids[1], calibration_summary


def render_one(sequence: Path, output: Path, model_path: Path, device: torch.device,
               fps_override: float | None, width: int, height: int, save_pose: bool) -> None:
    labels_path = sequence / 'labels.pt'
    if not labels_path.is_file():
        raise RuntimeError(f'找不到 {labels_path}')
    payload = torch.load(labels_path, map_location='cpu', weights_only=False)
    acceleration, orientation, watch_slot, phone_slot, calibration = mobileposer_signals(payload)
    fps = float(fps_override if fps_override is not None else payload.get('fps', 30.0))
    if fps <= 0:
        raise RuntimeError('fps 必须为正数')
    print(f'序列：{sequence}')
    print(f'帧数：{len(acceleration)}，时长：{len(acceleration) / fps:.2f} s，槽位：watch={watch_slot}, phone={phone_slot}')
    pose = infer_pose(acceleration, orientation, model_path, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    title = '/'.join(sequence.parts[-4:])
    render_video(pose, output, watch_slot, phone_slot, title, fps, width, height)
    if save_pose:
        torch.save({'pose': pose, 'fps': fps, 'watch_slot': watch_slot,
                    'phone_slot': phone_slot, 'calibration': calibration}, output.with_suffix('.pose.pt'))
    output.with_suffix('.json').write_text(json.dumps({
        'sequence': str(sequence), 'labels': str(labels_path), 'model': str(model_path),
        'output': str(output), 'frames': len(pose), 'fps': fps,
        'watch_slot': watch_slot, 'phone_slot': phone_slot,
        'coordinate_chain': 'R_MS = R_MI @ R_IS; R_MB = R_MS @ R_SB',
    }, ensure_ascii=False, indent=2), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='渲染 new_datasets 中 p02 的 MobilePoser 姿态')
    parser.add_argument('--sequence', type=Path, help='完整目录，或如 p02/lw_lp/WalkingStraight/g01 的相对路径')
    parser.add_argument('--all-p02', action='store_true', help='渲染 processed/p02 下全部已整理序列')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, help='单条输出 mp4；批量模式不能使用')
    parser.add_argument('--output-root', type=Path, default=None, help='批量输出根目录，默认 new_datasets/rendered')
    parser.add_argument('--model', type=Path, default=paths.weights_file)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda:0'), default='auto')
    parser.add_argument('--fps', type=float, default=None)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--save-pose', action='store_true', help='同时保存网络预测的24关节旋转')
    args = parser.parse_args()
    if bool(args.sequence) == bool(args.all_p02):
        parser.error('--sequence 与 --all-p02 必须且只能选择一个')
    if args.all_p02 and args.output:
        parser.error('--output 只适用于单条模式')
    if args.width <= 0 or args.height <= 0:
        parser.error('width/height 必须为正数')
    return args


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(f'找不到模型：{model_path}')
    device = choose_device(args.device)
    output_root = (args.output_root or root/'rendered').resolve()
    if args.sequence:
        sequences = [resolve_sequence(args.sequence, root)]
    else:
        sequences = sorted(path.parent for path in (root/'processed/p02').rglob('labels.pt'))
        if not sequences:
            raise SystemExit('processed/p02 中没有 labels.pt；请先完成人工边界确认并运行 --export')
    for sequence in sequences:
        try:
            relative = sequence.relative_to((root/'processed').resolve())
        except ValueError:
            relative = Path(sequence.name)
        output = args.output.resolve() if args.output else output_root/relative/'pose.mp4'
        render_one(sequence, output, model_path, device, args.fps, args.width, args.height, args.save_pose)


if __name__ == '__main__':
    main()
