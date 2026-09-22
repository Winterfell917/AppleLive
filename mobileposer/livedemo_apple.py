"""Real-time MobilePoser demo for the Sensor Read iOS/watchOS app."""

from __future__ import annotations

import argparse
import copy
import contextlib
import json
import pathlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pygame
import torch

from articulate.utils.unity import MotionViewer
from articulate.utils.smpl_viewer import LightweightSMPLViewer
from config import amass, model_config, paths
from models.chi2027_calibrator import build_model as build_chi2027_calibrator
from models.tic_calibrator import TICOnlineCalibrator, TICOperatorConfig, TICTransformerCalibrator
from sensor_apple import AppleIMUPlotter, AppleMocapSensor, MODALITIES
from utils.model_utils import load_model


BASE_DIR = Path(__file__).resolve().parent
CALIBRATOR_COMBO = [0, 3, 4]


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _safe_sequence_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._")
    if not safe:
        raise argparse.ArgumentTypeError("sequence name must contain a letter or number")
    return safe


def save_sequence_package(
    args,
    records,
    sensor,
    measured_fps: float,
    physical_start_timestamp_s: float,
    physical_end_timestamp_s: float,
    end_event: dict,
) -> Path:
    sequence_dir = args.dataset_dir / args.subject / args.sequence_name
    if sequence_dir.exists() and any(sequence_dir.iterdir()):
        raise RuntimeError(f"Sequence directory is not empty: {sequence_dir}")
    sequence_dir.mkdir(parents=True, exist_ok=True)

    payload = {key: torch.stack(value) for key, value in records.items()}
    calibration = sensor.calibration_state()
    payload["calibration"] = calibration
    payload["source_slots"] = dict(sensor.source_slots)
    torch.save(payload, sequence_dir / "mocap.pt")

    timestamps = payload["timestamp"]
    valid_timestamps = timestamps[timestamps > 0]
    mocap_start_timestamp_s = float(valid_timestamps.min().item())
    mocap_end_timestamp_s = float(valid_timestamps.max().item())
    start_timestamp_s = physical_start_timestamp_s
    end_timestamp_s = physical_end_timestamp_s
    session_ids = sorted(set(calibration.get("session_ids", {}).values()))
    manifest = {
        "schema_version": 1,
        "name": args.sequence_name,
        "sequence_name": args.sequence_name,
        "subject": args.subject,
        "action": args.action,
        "trial": args.trial,
        "notes": args.notes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_ids": session_ids,
        "source_session_ids": calibration.get("session_ids", {}),
        "sequence_start_timestamp_s": start_timestamp_s,
        "sequence_end_timestamp_s": end_timestamp_s,
        "save_trigger": end_event.get("event_type", "sensor_read_recording_end"),
        "recording_end_event": dict(end_event),
        "duration_s": end_timestamp_s - start_timestamp_s,
        "mocap_start_timestamp_s": mocap_start_timestamp_s,
        "mocap_end_timestamp_s": mocap_end_timestamp_s,
        "frame_count": int(payload["pose"].shape[0]),
        "target_fps": args.fps,
        "measured_fps": measured_fps,
        "calibration_method": args.calibration,
        "calibrator": (
            "ours+nocalibration"
            if args.compare_ours_nocalibration
            else args.calibrator
        ),
        "compare_ours_nocalibration": args.compare_ours_nocalibration,
        "model": str(args.model.resolve()),
        "source_slots": dict(sensor.source_slots),
        "coordinate_frames": {
            "M": "model/world",
            "I": "right-handed inertial; Apple input reflected on Z",
            "S": "right-handed sensor; Apple input reflected on Z",
            "B": "SMPL bone",
            "rotation_convention": "RXY maps vectors from Y to X",
        },
        "artifacts": {
            "mocap": "mocap.pt",
            "calibration": "calibration.json",
            "raw_sequence": None,
            "alignment": None,
            "quality_report": None,
        },
    }
    (sequence_dir / "calibration.json").write_text(
        json.dumps(_json_value(calibration), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (sequence_dir / "manifest.json").write_text(
        json.dumps(_json_value(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sequence_dir


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    # This model's small packed LSTMs are substantially faster on Apple CPU
    # than on the PyTorch 2.1 MPS backend used by the project.
    return torch.device("cpu")


class OnlineCHI2027Calibrator:
    """Causal rolling-window adapter for a direct-rotation calibrator."""

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.max_seq_len = model.max_seq_len
        self.buffer = None

    def reset(self) -> None:
        self.buffer = None

    @torch.inference_mode()
    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        frame = frame.to(self.device)
        self.buffer = (
            frame.unsqueeze(0)
            if self.buffer is None
            else torch.cat((self.buffer, frame.unsqueeze(0)), dim=0)
        )
        self.buffer = self.buffer[-self.max_seq_len :]
        sequence = self.buffer.unsqueeze(0)
        mask = torch.ones(
            1, len(self.buffer), dtype=torch.bool, device=self.device
        )
        prediction, _ = self.model(sequence, mask)
        return prediction[0, -1]


def _load_portable_checkpoint(path: Path):
    """Load checkpoints containing Linux PosixPath metadata on Windows."""
    try:
        return torch.load(path, map_location="cpu")
    except NotImplementedError as error:
        if "PosixPath" not in str(error):
            raise
        original_posix_path = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            return torch.load(path, map_location="cpu")
        finally:
            pathlib.PosixPath = original_posix_path


def _flatten_recurrent_parameters(module: torch.nn.Module) -> None:
    """Restore cuDNN-friendly contiguous RNN weights after deepcopy."""
    for child in module.modules():
        flatten = getattr(child, "flatten_parameters", None)
        if callable(flatten):
            flatten()


def load_combo_calibrator(path: Path, device: torch.device):
    checkpoint = _load_portable_checkpoint(path)
    required = {"model_type", "model_kwargs", "model_state_dict"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            f"Not a CHI 2027 calibrator checkpoint; missing {sorted(missing)}"
        )
    model = build_chi2027_calibrator(
        checkpoint["model_type"], **checkpoint["model_kwargs"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    return OnlineCHI2027Calibrator(model, device)


def load_tic_calibrator(path: Path, device: torch.device, buffer_size: int, trigger_t: float):
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
    return TICOnlineCalibrator(
        model,
        imu_num=len(CALIBRATOR_COMBO),
        config=TICOperatorConfig(
            buffer_size=buffer_size,
            trigger_t=trigger_t,
            data_frame_rate=30,
            ego_idx=len(CALIBRATOR_COMBO) - 1,
        ),
    )


def make_mocap_input(aM: torch.Tensor, RMB: torch.Tensor, device: torch.device):
    count = model_config.n_joints
    return torch.cat(
        ((aM[:count] / amass.acc_scale).flatten(), RMB[:count].flatten())
    ).to(device)


def arm_down_angles(model, pose: torch.Tensor):
    """Return shoulder-to-wrist angles from model-down (0=N-pose, 90=T-pose)."""
    _, joints = model.bodymodel.forward_kinematics(pose.unsqueeze(0), calc_mesh=False)
    joints = joints[0]
    down = joints.new_tensor([0.0, -1.0, 0.0])
    angles = {}
    for side, shoulder, wrist in (("left", 16, 20), ("right", 17, 21)):
        arm = joints[wrist] - joints[shoulder]
        cosine = torch.dot(arm / arm.norm().clamp_min(1e-8), down).clamp(-1.0, 1.0)
        angles[side] = float(torch.rad2deg(torch.acos(cosine)).item())
    return angles


def apply_combo_calibrator(calibrator, aM, RMB):
    calibrated_RMB = RMB.clone()
    frame = torch.cat(
        (
            aM[CALIBRATOR_COMBO] / amass.acc_scale,
            RMB[CALIBRATOR_COMBO].flatten(-2),
        ),
        dim=-1,
    )
    predicted_RMB = calibrator.forward_frame(frame)
    calibrated_RMB[CALIBRATOR_COMBO] = predicted_RMB
    return aM, calibrated_RMB


def apply_tic_calibrator(calibrator, aM, RMB):
    calibrated_aM = aM.clone()
    calibrated_RMB = RMB.clone()
    predicted_RMB, predicted_aM = calibrator.forward_frame(
        RMB[CALIBRATOR_COMBO].detach().cpu(),
        aM[CALIBRATOR_COMBO].detach().cpu(),
    )
    calibrated_aM[CALIBRATOR_COMBO] = predicted_aM.to(aM.device)
    calibrated_RMB[CALIBRATOR_COMBO] = predicted_RMB.to(RMB.device)
    return calibrated_aM, calibrated_RMB


def parse_source_slots(values):
    mapping = {}
    for value in values:
        source, separator, raw_slot = value.partition(":")
        if not separator:
            raise argparse.ArgumentTypeError(f"Invalid source mapping {value!r}; use source:slot")
        mapping[source] = int(raw_slot)
    return mapping


def parse_args():
    parser = argparse.ArgumentParser(description="Sensor Read Apple IMU → MobilePoser live demo")
    parser.add_argument("--host", default="0.0.0.0", help="UDP listen address")
    parser.add_argument("--port", type=int, default=9000, help="Sensor Read UDP port")
    parser.add_argument(
        "--source-slot",
        action="append",
        default=None,
        metavar="SOURCE:SLOT",
        help="override source mapping; repeat for each source",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda:0"), default="auto")
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="live inference rate; model timing remains calibrated at 30 Hz",
    )
    parser.add_argument("--stale-after", type=float, default=1.0)
    parser.add_argument(
        "--calibration-wait-timeout",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="extra time to wait for delayed native frames after a calibration window",
    )
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("--calibration", choices=("npose", "walking_6dof", "none"), default="walking_6dof")
    parser.add_argument(
        "--calibrator",
        choices=("none", "nocalibration", "plain", "ours", "tic"),
        default="none",
        help=(
            "rotation calibrator; none/nocalibration bypass calibration, "
            "ours and plain use the CHI 2027 checkpoints"
        ),
    )
    parser.add_argument(
        "--compare-ours-nocalibration",
        action="store_true",
        help="run independent ours and no-calibration MobilePoser branches together",
    )
    parser.add_argument(
        "--visualize-sensor",
        choices=("apple_watch", "iphone", "airpods"),
        default=None,
        help="show a live plot for this Apple sensor",
    )
    parser.add_argument(
        "--visualize-modality",
        choices=MODALITIES,
        default=None,
        help="modality to plot: acceleration (aS), aI, angular_velocity, or orientation",
    )
    parser.add_argument(
        "--visualize-window",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="rolling time span shown in the sensor plot",
    )
    parser.add_argument(
        "--viewer",
        choices=("skeleton", "lightweight", "unity", "none"),
        default="skeleton",
        help="pose viewer; skeleton is the default non-blocking stick-figure preview",
    )
    parser.add_argument(
        "--viewer-fps",
        type=float,
        default=30.0,
        help="maximum skeleton viewer refresh rate",
    )
    parser.add_argument(
        "--viewer-width",
        type=int,
        default=640,
        metavar="PIXELS",
        help="skeleton viewer width (default: 640)",
    )
    parser.add_argument(
        "--viewer-height",
        type=int,
        default=640,
        metavar="PIXELS",
        help="skeleton viewer height (default: 640)",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="deprecated alias for --viewer none",
    )
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument(
        "--debug-calibration",
        action="store_true",
        help="print N-pose input errors and inferred arm-down angles once per second",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop ordinary live mode automatically after N seconds",
    )
    sequence = parser.add_mutually_exclusive_group()
    sequence.add_argument(
        "--name",
        type=_safe_sequence_name,
        default=None,
        help="sequence name saved under data/datasets/apple/<subject>/<name>",
    )
    sequence.add_argument(
        "--sequence-name",
        type=_safe_sequence_name,
        default=None,
        help="deprecated alias for --name",
    )
    parser.add_argument(
        "--subject", type=_safe_sequence_name, default="yatong_0904",
        help="dataset subject folder; required with --name/--sequence-name",
    )
    parser.add_argument("--action", default=None, help="optional action label")
    parser.add_argument("--trial", default=None, help="optional trial identifier")
    parser.add_argument("--notes", default=None, help="optional free-form sequence notes")
    parser.add_argument("--model", type=Path, default=BASE_DIR / "data/checkpoints/base_model_12combo.pth")
    parser.add_argument(
        "--ours-calibrator",
        type=Path,
        default=BASE_DIR / "data/checkpoints/chi2027_calibrator_ours/best.pt",
        help="CHI 2027 cross-device calibrator checkpoint",
    )
    parser.add_argument(
        "--plain-calibrator",
        type=Path,
        default=BASE_DIR / "data/checkpoints/chi2027_calibrator_plain/best.pt",
        help="CHI 2027 causal Plain Transformer checkpoint",
    )
    parser.add_argument(
        "--tic-calibrator",
        type=Path,
        default=BASE_DIR / "data/checkpoints/tic_calibrator_amass_full/best.pt",
    )
    parser.add_argument("--tic-buffer-size", type=int, default=128)
    parser.add_argument("--tic-trigger-t", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data/records/apple")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=BASE_DIR / "data/datasets/apple",
        help="root directory for curated Apple sequence packages",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.sequence_name = args.name or args.sequence_name
    if args.calibrator == "nocalibration":
        args.calibrator = "none"
    if args.compare_ours_nocalibration and args.calibrator not in {"none", "ours"}:
        raise SystemExit(
            "--compare-ours-nocalibration cannot be combined with plain or tic"
        )
    if args.viewer == "lightweight":
        args.viewer = "skeleton"
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.viewer_fps <= 0:
        raise SystemExit("--viewer-fps must be positive")
    if args.viewer_width <= 0 or args.viewer_height <= 0:
        raise SystemExit("--viewer-width and --viewer-height must be positive")
    if args.calibration_wait_timeout <= 0:
        raise SystemExit("--calibration-wait-timeout must be positive")
    if args.no_viewer:
        args.viewer = "none"
    if bool(args.visualize_sensor) != bool(args.visualize_modality):
        raise SystemExit("--visualize-sensor and --visualize-modality must be used together")
    if args.visualize_window <= 0:
        raise SystemExit("--visualize-window must be positive")
    if args.sequence_name and not args.subject:
        raise SystemExit("--subject is required when recording a named sequence")
    if args.sequence_name and args.no_record:
        raise SystemExit("--name cannot be combined with --no-record")
    if args.sequence_name and args.duration is not None:
        raise SystemExit(
            "--name is stopped by Sensor Read End Collection "
            "and cannot use --duration"
        )
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration must be positive")
    source_slots = parse_source_slots(args.source_slot) if args.source_slot else None
    device = choose_device(args.device)
    model_config.device = device
    paths.smpl_file = BASE_DIR / "smpl/basicmodel_m.pkl"
    print(f"Inference device: {device}")

    model = load_model(str(args.model)).to(device).eval()
    nocalibration_model = None
    calibrator = None
    if args.compare_ours_nocalibration:
        nocalibration_model = copy.deepcopy(model).eval()
        _flatten_recurrent_parameters(model)
        _flatten_recurrent_parameters(nocalibration_model)
        calibrator = load_combo_calibrator(args.ours_calibrator, device)
        print(
            "Comparison mode: CHI 2027 ours + nocalibration; "
            f"checkpoint={args.ours_calibrator}"
        )
    elif args.calibrator == "plain":
        calibrator = load_combo_calibrator(args.plain_calibrator, device)
        print(
            "Loaded CHI 2027 Plain Transformer calibrator: "
            f"{args.plain_calibrator}"
        )
    elif args.calibrator == "ours":
        calibrator = load_combo_calibrator(args.ours_calibrator, device)
        print(f"Loaded CHI 2027 ours calibrator: {args.ours_calibrator}")
    elif args.calibrator == "tic":
        calibrator = load_tic_calibrator(
            args.tic_calibrator, device, args.tic_buffer_size, args.tic_trigger_t
        )

    records = {"timestamp": [], "acc": [], "ori": [], "gyro": [], "pose": [], "valid": []}
    if args.compare_ours_nocalibration:
        records.update(
            {
                "ori_nocalibration": [],
                "pose_nocalibration": [],
                "pose_ours": [],
            }
        )
    if args.viewer == "skeleton":
        viewer_context = LightweightSMPLViewer(
            model_path=paths.smpl_file,
            fps=args.viewer_fps,
            width=args.viewer_width,
            height=args.viewer_height,
            title=(
                "Apple Mocap — Calibration (comparison)"
                if args.compare_ours_nocalibration
                else "Apple Mocap — Skeleton Preview"
            ),
        )
    elif args.viewer == "unity":
        if args.compare_ours_nocalibration:
            viewer_context = MotionViewer(
                2, overlap=False, names=["NoCalibration", "Calibration"]
            )
        else:
            viewer_context = MotionViewer(1, overlap=False, names=["Apple IMU"])
    else:
        viewer_context = contextlib.nullcontext(None)

    with AppleMocapSensor(
        host=args.host,
        port=args.port,
        source_slots=source_slots,
        stale_after_s=args.stale_after,
        calibration_wait_timeout_s=args.calibration_wait_timeout,
    ) as sensor:
        print(f"Listening for Sensor Read at udp://{args.host}:{args.port}")
        print(f"Source mapping: {sensor.source_slots}")
        print("Live input: latest clock-aligned sample from each device")
        sensor.wait_until_ready(timeout_s=args.ready_timeout)
        if args.visualize_sensor and args.visualize_sensor not in sensor.source_slots:
            raise RuntimeError(
                f"Visualization source {args.visualize_sensor!r} is not configured; "
                f"available sources: {', '.join(sensor.required_sources)}"
            )
        if (
            (args.calibrator != "none" or args.compare_ours_nocalibration)
            and set(CALIBRATOR_COMBO) - set(sensor.source_slots.values())
        ):
            calibrator_name = (
                "ours" if args.compare_ours_nocalibration else args.calibrator
            )
            raise RuntimeError(
                f"{calibrator_name} calibrator requires MobilePoser slots "
                f"{CALIBRATOR_COMBO}"
            )
        if args.calibration == "npose":
            sensor.calibrate_npose()
        elif args.calibration == "walking_6dof":
            sensor.calibrate_walking_6dof()
        frozen_offsets = sensor.freeze_clock_offsets()
        print(f"Frozen clock offsets: {frozen_offsets}", flush=True)

        plotter_context = (
            AppleIMUPlotter(
                source=args.visualize_sensor,
                modality=args.visualize_modality,
                window_s=args.visualize_window,
            )
            if args.visualize_sensor
            else contextlib.nullcontext(None)
        )

        sequence_stop_state = {}
        sequence_completed = False
        recording_end_received = False
        interrupted = False
        ordinary_completed = False
        active_session_ids = {
            frame.session_id for frame in sensor.latest_frames().values()
        }
        try:
            sequence_start_timestamp_s = None
            with (
                torch.inference_mode(),
                plotter_context as plotter,
                contextlib.ExitStack() as viewer_stack,
            ):
                if args.sequence_name:
                    sequence_start_timestamp_s = time.time()
                    print(
                        "Calibration complete; sequence recording starts now. "
                        "Tap End Collection in "
                        "Sensor Read to stop and save; Ctrl-C discards it.",
                        flush=True,
                    )
                # Entering this context creates and raises the local window.
                viewer = viewer_stack.enter_context(viewer_context)
                model.reset()
                if nocalibration_model is not None:
                    nocalibration_model.reset()
                if isinstance(calibrator, OnlineCHI2027Calibrator):
                    calibrator.reset()
                clock = pygame.time.Clock()
                started = time.monotonic()
                next_debug_time = started
                if not args.sequence_name:
                    print(
                        "Streaming started. End collection in Sensor Read to save; "
                        "Ctrl-C discards the recording."
                    )
                while True:
                    recording_end = sensor.recording_end_state()
                    if (
                        recording_end is not None
                        and recording_end["session_id"] in active_session_ids
                    ):
                        recording_end_received = True
                        sequence_stop_state = recording_end
                        sequence_completed = bool(args.sequence_name)
                        ordinary_completed = not args.sequence_name
                        print(
                            "\nAccepted Sensor Read stop signal received "
                            f"({recording_end.get('event_type')}); saving is authorized.",
                            flush=True,
                        )
                        break
                    if (
                        not args.sequence_name
                        and args.duration is not None
                        and time.monotonic() - started >= args.duration
                    ):
                        ordinary_completed = True
                        break
                    clock.tick(args.fps)

                    timestamp, aM, RMB, gyroS, valid = sensor.get()
                    required_slots = sorted(sensor.source_slots.values())
                    required_valid = valid[required_slots]
                    missing = [
                        required_slots[i]
                        for i, value in enumerate(required_valid)
                        if not value
                    ]
                    # if missing:
                    #     print(
                    #         f"\rStale Apple IMU slot(s): {missing}; using latest samples",
                    #         end="",
                    #         flush=True,
                    #     )

                    # Use the oldest latest-sample time as the initial physical
                    # boundary; live inference itself does not require a shared
                    # per-frame watermark after this point.
                    frame_timestamp_s = float(timestamp[required_slots].min().item())
                    # One device's latest sample may still precede the start
                    # keypress while its next native batch is arriving.
                    if (
                        args.sequence_name
                        and frame_timestamp_s < sequence_start_timestamp_s
                    ):
                        continue

                    if plotter is not None:
                        frame = sensor.latest_frames().get(args.visualize_sensor)
                        if frame is not None:
                            plotter.update(frame)

                    aM = aM.to(device)
                    RMB = RMB.to(device)
                    pose_nocalibration = None
                    RMB_nocalibration = None
                    if args.compare_ours_nocalibration:
                        RMB_nocalibration = RMB.clone()
                        nocalibration_input = make_mocap_input(aM, RMB, device)
                        pose_nocalibration = nocalibration_model.forward_frame(
                            nocalibration_input
                        ).view(24, 3, 3)
                        aM, RMB = apply_combo_calibrator(calibrator, aM, RMB)
                    elif args.calibrator in {"plain", "ours"}:
                        aM, RMB = apply_combo_calibrator(calibrator, aM, RMB)
                    elif args.calibrator == "tic":
                        aM, RMB = apply_tic_calibrator(calibrator, aM, RMB)

                    mocap_input = make_mocap_input(aM, RMB, device)
                    pose = model.forward_frame(mocap_input).view(24, 3, 3)
                    pose_cpu = pose.detach().cpu()
                    if (
                        args.debug_calibration
                        and time.monotonic() >= next_debug_time
                    ):
                        input_errors = sensor.npose_orientation_errors(RMB.detach().cpu(), valid)
                        arm_angles = arm_down_angles(model, pose)
                        formatted_errors = ", ".join(
                            f"{source}={error:.1f}deg" if error is not None else f"{source}=stale"
                            for source, error in input_errors.items()
                        )
                        print(
                            "N-pose debug | input rotation error: "
                            f"{formatted_errors} | inferred arm-down angle: "
                            f"left={arm_angles['left']:.1f}deg, right={arm_angles['right']:.1f}deg",
                            flush=True,
                        )
                        next_debug_time = time.monotonic() + 1.0
                    if viewer is not None:
                        if args.viewer == "skeleton":
                            viewer.update(pose_cpu)
                        elif args.compare_ours_nocalibration:
                            viewer.update_all(
                                [
                                    pose_nocalibration.detach().cpu().numpy(),
                                    pose_cpu.numpy(),
                                ],
                                [np.zeros(3), np.zeros(3)],
                                render=True,
                            )
                        else:
                            viewer.update_all([pose_cpu.numpy()], [np.zeros(3)], render=True)

                    if not args.no_record:
                        records["timestamp"].append(timestamp.clone())
                        records["acc"].append(aM.detach().cpu())
                        records["ori"].append(RMB.detach().cpu())
                        records["gyro"].append(gyroS.clone())
                        records["pose"].append(pose_cpu)
                        records["valid"].append(valid.clone())
                        if args.compare_ours_nocalibration:
                            records["ori_nocalibration"].append(
                                RMB_nocalibration.detach().cpu()
                            )
                            records["pose_nocalibration"].append(
                                pose_nocalibration.detach().cpu()
                            )
                            records["pose_ours"].append(pose_cpu.clone())
                    print(f"\rFPS: {clock.get_fps():.2f}", end="", flush=True)
                if args.sequence_name and sequence_completed:
                    print("\nSequence recording complete.", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print("\nCtrl-C received; this recording will not be saved.")

        if args.sequence_name:
            if sequence_completed and recording_end_received and records["pose"]:
                physical_duration_s = (
                    sequence_stop_state["timestamp_s"] - sequence_start_timestamp_s
                )
                measured_fps = len(records["pose"]) / max(physical_duration_s, 1e-6)
                sequence_dir = save_sequence_package(
                    args,
                    records,
                    sensor,
                    measured_fps,
                    sequence_start_timestamp_s,
                    sequence_stop_state["timestamp_s"],
                    sequence_stop_state,
                )
                print(f"Saved curated sequence package: {sequence_dir}")
            else:
                print("No matching Sensor Read End Collection event; nothing was saved.")
        elif not args.no_record and records["pose"] and ordinary_completed and not interrupted:
            measured_fps = len(records["pose"]) / max(time.monotonic() - started, 1e-6)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output = args.output_dir / f"apple_mocap_{datetime.now():%Y%m%d_%H%M%S}.pt"
            payload = {key: torch.stack(value) for key, value in records.items()}
            payload["calibration"] = sensor.calibration_state()
            payload["source_slots"] = dict(sensor.source_slots)
            torch.save(payload, output)
            print(f"Saved recording: {output}")
        elif not args.no_record and records["pose"]:
            print("Recording discarded; nothing was saved.")


if __name__ == "__main__":
    main()
