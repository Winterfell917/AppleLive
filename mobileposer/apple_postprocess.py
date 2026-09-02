"""Apple IMU dataset postprocessing: sync, resample, live-style infer, and render."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import matplotlib
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apple_sequence_dataset import (
    DEFAULT_BUNDLE_ID,
    DEFAULT_DEVICE,
    canonical_source,
    load_json,
    pull_recording,
    write_json,
)
from config import amass, model_config, paths
from sensor_apple.sensor import APPLE_LH_TO_RH, GRAVITY, NUM_MODEL_SLOTS
from utils.model_utils import load_model


BASE_DIR = Path(__file__).resolve().parent
EXPECTED_STREAMS = {
    "apple_watch": "device_motion",
    "iphone": "device_motion",
    "airpods": "head_motion",
}


@dataclass
class NativeMotion:
    source: str
    timestamp_s: np.ndarray
    sequence_number: np.ndarray
    aS_left_g: np.ndarray
    gyroS_left: np.ndarray
    RIS_left: np.ndarray


def _event_time_s(event: dict) -> float:
    return int(event["timestampUnixNs"]) / 1_000_000_000.0


def _iter_session_events(raw_path: Path, session_ids: set) -> Iterable[dict]:
    with raw_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                _event_time_s(event)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"invalid NDJSON event at line {line_number}: {error}") from error
            if session_ids and str(event.get("sessionID", "")) not in session_ids:
                continue
            yield event


def load_session_events(raw_path: Path, session_ids: set) -> List[dict]:
    events = list(_iter_session_events(raw_path, session_ids))
    if not events:
        raise RuntimeError("raw file contains no event matching the manifest sessionID")
    return events


def save_raw_session(events: List[dict], output: Path) -> None:
    with output.open("w", encoding="utf-8") as target:
        for event in events:
            target.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def _rotation_matrix(values: dict) -> np.ndarray:
    keys = [f"rotation_m{row}{column}" for row in range(1, 4) for column in range(1, 4)]
    if not all(key in values for key in keys):
        raise ValueError("incomplete rotation matrix")
    # CMAttitude stores reference-to-sensor; downstream expects sensor-to-I.
    return np.asarray([float(values[key]) for key in keys], dtype=np.float64).reshape(3, 3).T


def extract_native_motion(events: List[dict]) -> Dict[str, NativeMotion]:
    grouped: Dict[str, List[dict]] = {source: [] for source in EXPECTED_STREAMS}
    for event in events:
        source = canonical_source(str(event.get("source", "")))
        if source in EXPECTED_STREAMS and event.get("sensor") == EXPECTED_STREAMS[source]:
            grouped[source].append(event)

    result = {}
    for source, items in grouped.items():
        if not items:
            raise RuntimeError(f"raw session has no {source}/{EXPECTED_STREAMS[source]} events")
        items.sort(key=_event_time_s)
        timestamps, sequences, accelerations, gyros, rotations = [], [], [], [], []
        for event in items:
            values = event.get("values", {})
            try:
                rotation = _rotation_matrix(values)
                acceleration = [
                    float(values[f"user_accel_{axis}_g"]) for axis in ("x", "y", "z")
                ]
                gyro = [
                    float(values[f"rotation_{axis}_rad_s"]) for axis in ("x", "y", "z")
                ]
            except (KeyError, TypeError, ValueError):
                continue
            timestamps.append(_event_time_s(event))
            sequences.append(int(event.get("sequenceNumber", -1)))
            accelerations.append(acceleration)
            gyros.append(gyro)
            rotations.append(rotation)
        if len(timestamps) < 2:
            raise RuntimeError(f"{source} has fewer than two complete motion samples")

        # Keep the final sample if duplicate timestamps are present.
        timestamp_array = np.asarray(timestamps, dtype=np.float64)
        keep = np.r_[np.diff(timestamp_array) > 0, True]
        result[source] = NativeMotion(
            source=source,
            timestamp_s=timestamp_array[keep],
            sequence_number=np.asarray(sequences, dtype=np.int64)[keep],
            aS_left_g=np.asarray(accelerations, dtype=np.float64)[keep],
            gyroS_left=np.asarray(gyros, dtype=np.float64)[keep],
            RIS_left=np.asarray(rotations, dtype=np.float64)[keep],
        )
    return result


def _clock_offsets(calibration: dict) -> Dict[str, float]:
    return {
        canonical_source(source): float(value or 0.0)
        for source, value in calibration.get("clock_offset_s", {}).items()
    }


def collection_start_time(events: List[dict], offsets: Dict[str, float], native) -> Tuple[float, str]:
    accepted_starts = []
    for event in events:
        source = canonical_source(str(event.get("source", "")))
        values = event.get("values", {})
        if (
            source == "apple_watch"
            and event.get("sensor") == "control_event"
            and values.get("action_start")
            and values.get("accepted")
        ):
            accepted_starts.append(_event_time_s(event) + offsets.get(source, 0.0))
    if accepted_starts:
        return min(accepted_starts), "watch_control_start"
    return max(
        motion.timestamp_s[0] + offsets.get(source, 0.0)
        for source, motion in native.items()
    ), "latest_first_motion_sample"


def _smoothed_peak(times: np.ndarray, acceleration_g: np.ndarray, start_s: float, end_s: float):
    mask = (times >= start_s) & (times <= end_s)
    if int(mask.sum()) < 5:
        raise RuntimeError("fewer than five samples in jump synchronization window")
    selected_times = times[mask]
    magnitude = np.linalg.norm(acceleration_g[mask], axis=1)
    median_dt = float(np.median(np.diff(selected_times)))
    smooth_count = max(1, round(0.04 / median_dt)) if median_dt > 0 else 1
    kernel = np.ones(smooth_count, dtype=np.float64) / smooth_count
    smoothed = np.convolve(magnitude, kernel, mode="same")
    index = int(np.argmax(smoothed))
    baseline = float(np.median(smoothed))
    return {
        "timestamp_s": float(selected_times[index]),
        "magnitude_g": float(smoothed[index]),
        "prominence_g": float(smoothed[index] - baseline),
        "times": selected_times,
        "magnitude": magnitude,
        "smoothed": smoothed,
    }


def estimate_jump_shifts(
    native: Dict[str, NativeMotion],
    events: List[dict],
    offsets: Dict[str, float],
    reference_source: str,
    search_offset_s: float,
    search_duration_s: float,
    min_prominence_g: float,
):
    start_s, start_source = collection_start_time(events, offsets, native)
    search_start_s = start_s + search_offset_s
    search_end_s = search_start_s + search_duration_s
    peaks = {}
    for source, motion in native.items():
        corrected = motion.timestamp_s + offsets.get(source, 0.0)
        peak = _smoothed_peak(corrected, motion.aS_left_g, search_start_s, search_end_s)
        if peak["prominence_g"] < min_prominence_g:
            raise RuntimeError(
                f"weak jump peak for {source}: prominence={peak['prominence_g']:.3f} g; "
                "repeat collection with one clear upward jump or lower --min-jump-prominence"
            )
        peaks[source] = peak
    reference_time_s = peaks[reference_source]["timestamp_s"]
    shifts = {
        source: reference_time_s - peak["timestamp_s"] for source, peak in peaks.items()
    }
    report = {
        "mode": "jump_acceleration_magnitude",
        "collection_start_timestamp_s": start_s,
        "collection_start_source": start_source,
        "search_start_timestamp_s": search_start_s,
        "search_end_timestamp_s": search_end_s,
        "reference_source": reference_source,
        "sources": {
            source: {
                "peak_timestamp_before_shift_s": peak["timestamp_s"],
                "peak_magnitude_g": peak["magnitude_g"],
                "peak_prominence_g": peak["prominence_g"],
                "residual_shift_s": shifts[source],
                "peak_timestamp_after_shift_s": peak["timestamp_s"] + shifts[source],
            }
            for source, peak in peaks.items()
        },
    }
    return shifts, peaks, report


def plot_jump_alignment(peaks: dict, shifts: dict, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.8))
    reference = min(peak["timestamp_s"] + shifts[source] for source, peak in peaks.items())
    for source, peak in peaks.items():
        aligned_times = peak["times"] + shifts[source] - reference
        axis.plot(aligned_times, peak["smoothed"], label=source)
        peak_time = peak["timestamp_s"] + shifts[source] - reference
        axis.axvline(peak_time, linestyle="--", alpha=0.35)
    axis.set(title="Jump-based device synchronization", xlabel="Aligned time (s)", ylabel="|userAcceleration| (g)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _linear_resample(times: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(targets, times, values[:, axis]) for axis in range(values.shape[1])], axis=1)


def _rotation_resample(times: np.ndarray, matrices: np.ndarray, targets: np.ndarray) -> np.ndarray:
    origin = times[0]
    return Slerp(times - origin, Rotation.from_matrix(matrices))(targets - origin).as_matrix()


def _stream_report(motion: NativeMotion, aligned_times: np.ndarray, targets: np.ndarray) -> dict:
    gaps = np.diff(aligned_times)
    sequence_diff = np.diff(motion.sequence_number)
    right = np.searchsorted(aligned_times, targets, side="left")
    right = np.clip(right, 1, len(aligned_times) - 1)
    interpolation_gaps = aligned_times[right] - aligned_times[right - 1]
    span = aligned_times[-1] - aligned_times[0]
    return {
        "native_sample_count": int(len(aligned_times)),
        "estimated_rate_hz": float((len(aligned_times) - 1) / span),
        "max_native_gap_s": float(gaps.max()),
        "max_resampling_bracket_s": float(interpolation_gaps.max()),
        "sequence_gap_count": int((sequence_diff > 1).sum()),
        "missing_sequence_count": int(np.maximum(sequence_diff - 1, 0).sum()),
        "coverage_start_s": float(aligned_times[0]),
        "coverage_end_s": float(aligned_times[-1]),
    }


def build_processed_imu(
    native: Dict[str, NativeMotion],
    calibration: dict,
    manifest: dict,
    shifts: Dict[str, float],
    fps: float,
    max_gap_s: float,
    gap_policy: str,
):
    offsets = _clock_offsets(calibration)
    aligned_times = {
        source: motion.timestamp_s + offsets.get(source, 0.0) + shifts.get(source, 0.0)
        for source, motion in native.items()
    }
    requested_start = float(manifest["sequence_start_timestamp_s"])
    requested_end = float(manifest["sequence_end_timestamp_s"])
    common_start = max(requested_start, *(times[0] for times in aligned_times.values()))
    common_end = min(requested_end, *(times[-1] for times in aligned_times.values()))
    grid_start = math.ceil(common_start * fps - 1e-7) / fps
    grid_end = math.floor(common_end * fps + 1e-7) / fps
    frame_count = int(math.floor((grid_end - grid_start) * fps + 1e-6)) + 1
    if frame_count < 2:
        raise RuntimeError("three devices have no usable common 30 FPS interval")
    targets = grid_start + np.arange(frame_count, dtype=np.float64) / fps

    RMI = torch.tensor(calibration["RMI"], dtype=torch.float32)
    RSB = torch.tensor(calibration["RSB"], dtype=torch.float32)
    source_slots = {
        canonical_source(source): int(slot)
        for source, slot in calibration["source_slots"].items()
    }
    aS = torch.zeros(frame_count, NUM_MODEL_SLOTS, 3)
    aI = torch.zeros_like(aS)
    aM = torch.zeros_like(aS)
    gyroS = torch.zeros_like(aS)
    RIS = torch.zeros(frame_count, NUM_MODEL_SLOTS, 3, 3)
    RMB = torch.zeros_like(RIS)
    valid = torch.zeros(frame_count, NUM_MODEL_SLOTS, dtype=torch.bool)
    source_sample_timestamp = torch.zeros(frame_count, NUM_MODEL_SLOTS, dtype=torch.float64)
    C = APPLE_LH_TO_RH.numpy().astype(np.float64)
    reports = {}

    for source, slot in source_slots.items():
        motion = native[source]
        times = aligned_times[source]
        report = _stream_report(motion, times, targets)
        right = np.searchsorted(times, targets, side="left")
        right = np.clip(right, 1, len(times) - 1)
        interpolation_gaps = times[right] - times[right - 1]
        frame_valid_np = interpolation_gaps <= max_gap_s
        report["invalid_output_frame_count"] = int((~frame_valid_np).sum())
        report["invalid_output_fraction"] = float((~frame_valid_np).mean())
        if gap_policy == "error" and not bool(frame_valid_np.all()):
            raise RuntimeError(
                f"{source} has a {report['max_resampling_bracket_s']:.3f}s gap inside the "
                f"output interval (limit {max_gap_s:.3f}s)"
            )
        reports[source] = report

        aS_right = (motion.aS_left_g @ C.T) * GRAVITY
        gyro_right = motion.gyroS_left @ (-C).T
        RIS_right = np.einsum("ij,njk,kl->nil", C, motion.RIS_left, C)
        aS_slot = torch.from_numpy(_linear_resample(times, aS_right, targets)).float()
        gyro_slot = torch.from_numpy(_linear_resample(times, gyro_right, targets)).float()
        RIS_slot = torch.from_numpy(_rotation_resample(times, RIS_right, targets)).float()

        right = np.searchsorted(times, targets, side="left")
        right = np.clip(right, 0, len(times) - 1)
        left = np.maximum(right - 1, 0)
        choose_left = np.abs(times[left] - targets) <= np.abs(times[right] - targets)
        nearest = np.where(choose_left, left, right)
        frame_valid = torch.from_numpy(frame_valid_np)
        aI_slot = torch.matmul(RIS_slot, aS_slot.unsqueeze(-1)).squeeze(-1)
        aM_slot = torch.matmul(RMI[slot], aI_slot.unsqueeze(-1)).squeeze(-1)
        RMB_slot = RMI[slot] @ RIS_slot @ RSB[slot]

        # A long native-data hole is represented exactly like a missing IMU in
        # training: zero acceleration/rotation and valid=False.  This preserves
        # a common timeline without inventing measurements across the hole.
        aS[frame_valid, slot] = aS_slot[frame_valid]
        aI[frame_valid, slot] = aI_slot[frame_valid]
        aM[frame_valid, slot] = aM_slot[frame_valid]
        gyroS[frame_valid, slot] = gyro_slot[frame_valid]
        RIS[frame_valid, slot] = RIS_slot[frame_valid]
        RMB[frame_valid, slot] = RMB_slot[frame_valid]
        valid[frame_valid, slot] = True
        source_sample_timestamp[frame_valid, slot] = torch.from_numpy(times[nearest])[frame_valid]

    result = {
        "schema_version": 1,
        "fps": float(fps),
        "timestamp_s": torch.from_numpy(targets),
        "source_sample_timestamp_s": source_sample_timestamp,
        "aS": aS,
        "aI": aI,
        "aM": aM,
        "gyroS": gyroS,
        "RIS": RIS,
        "RMB": RMB,
        "valid": valid,
        "RMI": RMI,
        "RSB": RSB,
        "source_slots": source_slots,
        "clock_offset_s": offsets,
        "jump_residual_shift_s": shifts,
    }
    grid_report = {
        "requested_start_s": requested_start,
        "requested_end_s": requested_end,
        "output_start_s": float(targets[0]),
        "output_end_s": float(targets[-1]),
        "fps": float(fps),
        "frame_count": frame_count,
        "sources": reports,
    }
    return result, grid_report


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def infer_pose(processed: dict, model_path: Path, device: torch.device) -> dict:
    model_config.device = device
    paths.smpl_file = BASE_DIR / "smpl/basicmodel_m.pkl"
    model = load_model(str(model_path)).to(device).eval()
    model.reset()
    aM = processed["aM"][:, :model_config.n_joints]
    RMB = processed["RMB"][:, :model_config.n_joints]
    model_input = torch.cat(
        ((aM / amass.acc_scale).flatten(1), RMB.flatten(1)), dim=1
    ).to(device)
    # forward_frame appends the new sample at the end of a 45-frame window,
    # then returns window index 40. Consequently its output corresponds to the
    # input four samples earlier. Feed four copies of the final sample to flush
    # that look-ahead, then discard the four startup outputs so pose[i] and the
    # original model_input[i] refer to the same target time.
    output_index = int(model.num_past_frames)
    lookahead_frames = int(model.num_total_frames) - output_index - 1
    inference_input = torch.cat(
        (model_input, model_input[-1:].repeat(lookahead_frames, 1)), dim=0
    )
    raw_poses = []
    with torch.inference_mode():
        for index, frame in enumerate(inference_input):
            # Deliberately identical to livedemo_apple.py: one stateful online
            # step per input frame, with the model reset once per sequence.
            raw_poses.append(model.forward_frame(frame).view(24, 3, 3).cpu())
            if (index + 1) % 100 == 0 or index + 1 == len(inference_input):
                print(
                    f"\rModel inference: {index + 1}/{len(inference_input)}",
                    end="", flush=True,
                )
    print()
    poses = torch.stack(raw_poses)[lookahead_frames:lookahead_frames + len(model_input)]
    return {
        "model_input": model_input.cpu(),
        "pose": poses,
        "pose_lookahead_frames": lookahead_frames,
        "pose_tail_padding": "repeat_last_frame",
    }


def render_pose_video(
    pose: torch.Tensor,
    output: Path,
    fps: float,
    width: int = 720,
    height: int = 720,
    batch_size: int = 64,
) -> None:
    import open3d as o3d
    from articulate.model import ParametricModel

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    bodymodel = ParametricModel(BASE_DIR / "smpl/basicmodel_m.pkl", device=torch.device("cpu"))
    _, zero_vertices = bodymodel.get_zero_pose_joint_and_vertex()
    center = zero_vertices.mean(dim=0).numpy()
    scale = float(zero_vertices.std(dim=0).max()) * 6.0
    normalized_zero = (zero_vertices.numpy() - center) / scale
    faces = np.asarray(bodymodel.face, dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(normalized_zero),
        triangles=o3d.utility.Vector3iVector(faces),
    )
    mesh.paint_uniform_color([0.28, 0.62, 0.92])
    visualizer = o3d.visualization.Visualizer()
    if not visualizer.create_window(width=width, height=height, visible=False):
        raise RuntimeError("Open3D could not create an offscreen video renderer")
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        visualizer.destroy_window()
        raise RuntimeError(f"could not open MP4 writer: {output}")
    try:
        visualizer.add_geometry(mesh)
        render = visualizer.get_render_option()
        render.background_color = np.asarray([0.96, 0.97, 0.99])
        render.mesh_show_back_face = True
        camera = visualizer.get_view_control()
        parameters = camera.convert_to_pinhole_camera_parameters()
        parameters.extrinsic = np.asarray(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.2], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        camera.convert_from_pinhole_camera_parameters(parameters)

        frame_index = 0
        for start in range(0, len(pose), batch_size):
            with torch.inference_mode():
                vertices = bodymodel.forward_kinematics(
                    pose[start:start + batch_size], calc_mesh=True
                )[2].numpy()
            vertices = (vertices - center) / scale
            for frame_vertices in vertices:
                mesh.vertices = o3d.utility.Vector3dVector(frame_vertices)
                mesh.compute_vertex_normals()
                visualizer.update_geometry(mesh)
                visualizer.poll_events()
                visualizer.update_renderer()
                rgb = (np.asarray(visualizer.capture_screen_float_buffer()) * 255).astype(np.uint8)
                cv2.putText(
                    rgb,
                    f"Frame {frame_index + 1}/{len(pose)}  {frame_index / fps:.2f}s",
                    (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 35), 2, cv2.LINE_AA,
                )
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                frame_index += 1
                if frame_index % 100 == 0 or frame_index == len(pose):
                    print(f"\rVideo rendering: {frame_index}/{len(pose)}", end="", flush=True)
    finally:
        writer.release()
        visualizer.destroy_window()
    print()


def validate_lengths(processed: dict) -> Dict[str, int]:
    keys = (
        "timestamp_s", "source_sample_timestamp_s", "aS", "aI", "aM",
        "gyroS", "RIS", "RMB", "valid", "model_input", "pose",
    )
    lengths = {key: int(len(processed[key])) for key in keys}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"processed modality lengths do not match: {lengths}")
    for key in (
        "aS", "aI", "aM", "gyroS", "RIS", "RMB", "model_input", "pose",
    ):
        if not bool(torch.isfinite(processed[key]).all()):
            raise RuntimeError(f"processed modality contains non-finite values: {key}")
    return lengths


def resolve_model_path(args, manifest: dict) -> Path:
    if args.model is not None:
        model_path = args.model.resolve()
    else:
        manifest_model = Path(str(manifest.get("model", "")))
        model_path = manifest_model if manifest_model.is_file() else BASE_DIR / "data/checkpoints/base_model_12combo.pth"
    if not model_path.is_file():
        raise RuntimeError(f"model checkpoint does not exist: {model_path}")
    return model_path


def process_sequence(args, raw_path: Path) -> None:
    sequence_dir = args.sequence_dir.resolve()
    manifest_path = sequence_dir / "manifest.json"
    calibration_path = sequence_dir / "calibration.json"
    manifest = load_json(manifest_path)
    calibration = load_json(calibration_path)
    session_ids = set(str(value) for value in manifest.get("session_ids", []))
    events = load_session_events(raw_path, session_ids)
    save_raw_session(events, sequence_dir / "raw.ndjson")
    native = extract_native_motion(events)
    offsets = _clock_offsets(calibration)

    if args.sync_mode == "jump":
        shifts, peaks, sync_report = estimate_jump_shifts(
            native, events, offsets, args.reference_source,
            args.jump_search_offset, args.jump_search_seconds,
            args.min_jump_prominence,
        )
        plot_jump_alignment(peaks, shifts, sequence_dir / "sync_jump.png")
    else:
        shifts = {source: 0.0 for source in native}
        sync_report = {"mode": "calibration_clock_offsets_only", "sources": {
            source: {"residual_shift_s": 0.0} for source in native
        }}

    processed, grid_report = build_processed_imu(
        native, calibration, manifest, shifts, args.fps, args.max_gap, args.gap_policy
    )
    model_path = resolve_model_path(args, manifest)
    device = choose_device(args.device)
    started = time.monotonic()
    processed.update(infer_pose(processed, model_path, device))
    processed["inference_mode"] = "online_forward_frame_livedemo_delay_compensated"
    processed["model"] = str(model_path)
    processed["inference_device"] = str(device)
    lengths = validate_lengths(processed)
    torch.save(processed, sequence_dir / "processed_30fps.pt")

    video_name = None
    if not args.no_video:
        video_name = "preview_30fps.mp4"
        render_pose_video(processed["pose"], sequence_dir / video_name, args.fps)

    quality = {
        "schema_version": 1,
        "raw_source_file": raw_path.name,
        "selected_session_event_count": len(events),
        "settings": {
            "fps": args.fps,
            "sync_mode": args.sync_mode,
            "max_allowed_resampling_gap_s": args.max_gap,
            "gap_policy": args.gap_policy,
        },
        "synchronization": sync_report,
        "output_grid": grid_report,
        "modality_lengths": lengths,
        "model": str(model_path),
        "inference_device": str(device),
        "inference_mode": processed["inference_mode"],
        "pose_lookahead_frames": processed["pose_lookahead_frames"],
        "pose_tail_padding": processed["pose_tail_padding"],
        "pipeline_elapsed_s": time.monotonic() - started,
        "artifacts": {
            "raw": "raw.ndjson",
            "processed": "processed_30fps.pt",
            "sync_plot": "sync_jump.png" if args.sync_mode == "jump" else None,
            "preview_video": video_name,
        },
    }
    write_json(sequence_dir / "postprocess_quality.json", quality)
    manifest.setdefault("artifacts", {}).update({
        "raw_sequence": "raw.ndjson",
        "processed_30fps": "processed_30fps.pt",
        "postprocess_quality": "postprocess_quality.json",
        "sync_plot": "sync_jump.png" if args.sync_mode == "jump" else None,
        "preview_video": video_name,
    })
    manifest["postprocess"] = {
        "fps": args.fps,
        "frame_count": lengths["pose"],
        "sync_mode": args.sync_mode,
        "max_allowed_resampling_gap_s": args.max_gap,
        "gap_policy": args.gap_policy,
        "model": str(model_path),
        "inference_mode": processed["inference_mode"],
        "pose_lookahead_frames": processed["pose_lookahead_frames"],
        "pose_tail_padding": processed["pose_tail_padding"],
    }
    write_json(manifest_path, manifest)
    print(f"Processed dataset sequence: {sequence_dir}")
    print(f"Frames: {lengths['pose']} @ {args.fps:g} FPS; all modality lengths match")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess one Apple IMU sequence at 30 FPS")
    parser.add_argument("--sequence-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pull", action="store_true", help="pull full matching NDJSON from iPhone")
    source.add_argument("--raw-file", type=Path, help="use a manually exported full-session NDJSON")
    parser.add_argument("--device-name", default=DEFAULT_DEVICE, help="devicectl device name or identifier")
    parser.add_argument("--bundle-id", default=DEFAULT_BUNDLE_ID)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--sync-mode", choices=("jump", "clock"), default="jump")
    parser.add_argument("--reference-source", choices=tuple(EXPECTED_STREAMS), default="iphone")
    parser.add_argument("--jump-search-offset", type=float, default=0.2)
    parser.add_argument("--jump-search-seconds", type=float, default=8.0)
    parser.add_argument("--min-jump-prominence", type=float, default=0.35, metavar="G")
    parser.add_argument("--max-gap", type=float, default=0.12, metavar="SECONDS")
    parser.add_argument(
        "--gap-policy", choices=("error", "mask"), default="error",
        help="fail on long native gaps, or zero/mask the affected device frames",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda:0"), default="auto")
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.sequence_dir = args.sequence_dir.resolve()
    for required in ("manifest.json", "calibration.json"):
        if not (args.sequence_dir / required).is_file():
            raise SystemExit(f"missing sequence artifact: {args.sequence_dir / required}")
    if args.fps <= 0 or args.jump_search_seconds <= 0 or args.max_gap <= 0:
        raise SystemExit("--fps, --jump-search-seconds, and --max-gap must be positive")
    if args.jump_search_offset < 0 or args.min_jump_prominence < 0:
        raise SystemExit("jump search offset/prominence must be non-negative")

    if args.raw_file is not None:
        process_sequence(args, args.raw_file.resolve())
    elif args.pull:
        with tempfile.TemporaryDirectory(prefix="apple-postprocess-") as temporary:
            raw_path = pull_recording(
                args.sequence_dir, args.device_name, args.bundle_id, Path(temporary)
            )
            process_sequence(args, raw_path)
    else:
        raw_path = args.sequence_dir / "raw.ndjson"
        if not raw_path.is_file():
            raise SystemExit("use --pull or --raw-file, or provide sequence-dir/raw.ndjson")
        process_sequence(args, raw_path)


if __name__ == "__main__":
    main()
