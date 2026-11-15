"""Audit calibration and coordinate consistency in controlled retest recordings.

This is deliberately an input/coordinate audit, not a model-training script.
It does not alter any label or raw-data artifact.  It reports what can be
reproduced from the saved calibration samples separately from what can be
established about physical, cross-recording direction.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import collect_train_sequence as train
from sensor_apple.sensor import _project_rotation


BASE = Path(__file__).resolve().parent
SUBJECTS = ("p01", "p02")
OUT = BASE / "data" / "datasets" / "orientation_retest_audit"
YAW_VALUES = (0, 90, 180, 270)
WINDOW_SHIFTS_S = (-0.2, -0.1, 0.0, 0.1, 0.2)
REFERENCE_FRAME_STATUS = {
    "status": "unconfirmed",
    "reason": (
        "The iPhone/Watch application source that selects Core Motion's "
        "attitude reference frame is not present in this repository."
    ),
    "cross_record_R_MI_interpretation": "descriptive_only",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: object) -> object:
    """Serialize NumPy/Torch values retained for audit traceability."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def rotation_angle_deg(relative: np.ndarray) -> np.ndarray:
    trace = np.trace(relative, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def mean_rotation_angle_deg(rotations: np.ndarray) -> float:
    if len(rotations) < 2:
        return 0.0
    relative = np.matmul(rotations, np.swapaxes(rotations[0:1], -1, -2))
    return float(np.median(rotation_angle_deg(relative)))


def finite_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(array).all())


def matrix_difference_deg(first: object, second: object) -> float | None:
    try:
        a = np.asarray(first, dtype=float)
        b = np.asarray(second, dtype=float)
        if a.shape != (3, 3) or b.shape != (3, 3):
            return None
        return float(rotation_angle_deg(np.matmul(a, b.T)))
    except (TypeError, ValueError):
        return None


def robust_start_rotation(orientation: np.ndarray, device: int, frames: int = 15) -> np.ndarray:
    """Project the mean of the first 0.5 s of formal-action R_MS samples."""
    return _project_rotation(
        torch.tensor(orientation[:frames, device], dtype=torch.float32).mean(dim=0)
    ).numpy()


def project_mean_rotation(rotations: list[torch.Tensor]) -> np.ndarray | None:
    """Return a proper rotation matrix for a short, mostly-static pose."""
    if not rotations:
        return None
    return _project_rotation(torch.stack(rotations).mean(dim=0)).numpy()


def model_pose(samples: list[dict], r_mi: np.ndarray) -> np.ndarray | None:
    """Mean R_MS = R_MI @ R_IS for a selected raw sample window."""
    matrix = torch.tensor(r_mi, dtype=torch.float32)
    return project_mean_rotation([matrix @ sample["RIS"] for sample in samples])


def integrate_walk_trace(samples: list[dict], span_s: float) -> dict | None:
    """Mirror the calibration integration and retain traces for plotting."""
    ordered = sorted(samples, key=lambda sample: sample["timestamp_s"])
    if len(ordered) < 2:
        return None
    begin = ordered[0]["timestamp_s"]
    previous = begin
    position = torch.zeros(3)
    velocity = torch.zeros(3)
    times, acceleration_h, velocity_h, position_h = [], [], [], []
    for sample in ordered[1:]:
        timestamp = sample["timestamp_s"]
        if timestamp <= previous:
            continue
        if timestamp - begin > span_s:
            break
        dt = timestamp - previous
        previous = timestamp
        acceleration = sample["aI"]
        position = position + dt * velocity + 0.5 * dt * dt * acceleration
        velocity = velocity + dt * acceleration
        horizontal_acceleration = acceleration - torch.dot(
            acceleration, train.APPLE_I_UP_RH
        ) * train.APPLE_I_UP_RH
        horizontal_velocity = velocity - torch.dot(
            velocity, train.APPLE_I_UP_RH
        ) * train.APPLE_I_UP_RH
        horizontal_position = position - torch.dot(
            position, train.APPLE_I_UP_RH
        ) * train.APPLE_I_UP_RH
        times.append(timestamp - begin)
        acceleration_h.append(float(horizontal_acceleration.norm()))
        velocity_h.append(float(horizontal_velocity.norm()))
        position_h.append(float(horizontal_position.norm()))
    duration = previous - begin
    if duration < span_s * 0.8:
        return None
    corrected = position - 0.5 * duration * velocity
    return {
        "position": position,
        "velocity": velocity,
        "duration_s": duration,
        "corrected": corrected,
        "time_s": np.asarray(times),
        "horizontal_acceleration_m_s2": np.asarray(acceleration_h),
        "horizontal_velocity_m_s": np.asarray(velocity_h),
        "horizontal_position_m": np.asarray(position_h),
    }


def rmi_from_walk_samples(samples: list[dict], span_s: float) -> tuple[np.ndarray | None, float | None]:
    trace = integrate_walk_trace(samples, span_s)
    if trace is None:
        return None, None
    built = train.rmi_from_displacement(trace["corrected"])
    if built is None:
        return None, 0.0
    return built[0].numpy(), float(built[1])


def load_raw_motion(capture: dict) -> tuple[dict[str, list[dict]] | None, str | None]:
    raw = capture.get("raw_files", {})
    if not raw.get("iphone") or not raw.get("apple_watch"):
        return None, "raw file paths missing"
    try:
        events, _ = train.merge_device_events(Path(raw["iphone"]), Path(raw["apple_watch"]))
        offsets = capture.get("time_offsets_s") or {
            source: 0.0 for source in train.DEVICES
        }
        motion, _, _ = train.extract_offline_streams(events, offsets)
        return motion, None
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        return None, str(error)


def selected_phone_calibration(
    directory: Path, capture: dict, motion: dict[str, list[dict]]
) -> tuple[list[dict], list[dict], str, str | None]:
    """Use exact live event identities when this older take preserved them."""
    npose_start, npose_end = train.window_bounds(capture, "npose")
    walk_start, walk_end = train.window_bounds(capture, "forward_walk")
    artifact = capture.get("artifacts", {}).get("live_calibration")
    if artifact and (directory / artifact).is_file():
        try:
            live_trace = train.load_live_calibration_samples(directory / artifact)
            npose, _ = train.match_trace_samples(motion["iphone"], live_trace["npose"], "N-pose")
            walk, _ = train.match_trace_samples(
                motion["iphone"], live_trace["forward_walk"], "forward-walk"
            )
            return npose, walk, "explicit_event_identity", None
        except (OSError, KeyError, RuntimeError, ValueError) as error:
            return [], [], "explicit_event_identity_failed", str(error)
    return (
        train.samples_between(motion["iphone"], npose_start, npose_end),
        train.samples_between(motion["iphone"], walk_start, walk_end),
        "pc_unix_window_fallback",
        None,
    )


def window_sensitivity(motion: list[dict], capture: dict) -> dict:
    """Rebuild R_MI after independently shifting walk start/end by ±0.1/0.2 s."""
    start, end = train.window_bounds(capture, "forward_walk")
    variants = []
    for start_shift in WINDOW_SHIFTS_S:
        for end_shift in WINDOW_SHIFTS_S:
            shifted_start, shifted_end = start + start_shift, end + end_shift
            if shifted_end - shifted_start < 0.8:
                continue
            samples = train.samples_between(motion, shifted_start, shifted_end)
            r_mi, horizontal_m = rmi_from_walk_samples(samples, shifted_end - shifted_start)
            variants.append(
                {
                    "start_shift_s": start_shift,
                    "end_shift_s": end_shift,
                    "sample_count": len(samples),
                    "horizontal_m": horizontal_m,
                    "R_MI": r_mi,
                }
            )
    valid = [item for item in variants if item["R_MI"] is not None]
    angles = [
        matrix_difference_deg(first["R_MI"], second["R_MI"])
        for index, first in enumerate(valid)
        for second in valid[index + 1 :]
    ]
    distances = [item["horizontal_m"] for item in valid if item["horizontal_m"] is not None]
    return {
        "method": "PC Unix forward-walk window, independent start/end shifts",
        "shift_values_s": list(WINDOW_SHIFTS_S),
        "variant_count": len(variants),
        "valid_variant_count": len(valid),
        "R_MI_pairwise_max_deg": float(max(angles)) if angles else None,
        "R_MI_pairwise_median_deg": float(np.median(angles)) if angles else None,
        "horizontal_distance_min_m": float(min(distances)) if distances else None,
        "horizontal_distance_max_m": float(max(distances)) if distances else None,
        "variants": variants,
    }


def compare_rmi_and_start_rms() -> list[dict]:
    rows = []
    root_base = BASE / "data" / "datasets" / "subjects"
    for subject in SUBJECTS:
        for action in ("Boxing", "Waving", "Walking"):
            entries = []
            for yaw in (0, 90, 180, 270):
                directory = (
                    root_base
                    / subject
                    / "orientation_retest_labels"
                    / f"yaw_{yaw:03d}"
                    / action
                    / "lw_lp"
                    / "g01_screen_in"
                )
                labels = torch.load(
                    directory / "labels.pt", map_location="cpu", weights_only=False
                )
                meta = read_json(directory / "meta.json")
                entries.append(
                    {
                        "yaw": yaw,
                        "rmi": np.asarray(meta["calibration"]["iphone"]["R_MI"], dtype=float),
                        "orientation": labels["orientation"].numpy(),
                    }
                )
            for metric, device, getter in (
                ("R_MI", None, lambda item: item["rmi"]),
                ("R_MS_start_phone", 1, lambda item: robust_start_rotation(item["orientation"], 1)),
                ("R_MS_start_watch", 0, lambda item: robust_start_rotation(item["orientation"], 0)),
            ):
                for left in range(4):
                    for right in range(left + 1, 4):
                        first = getter(entries[left])
                        second = getter(entries[right])
                        rows.append(
                            {
                                "subject": subject,
                                "action": action,
                                "metric": metric,
                                "left_yaw_deg": entries[left]["yaw"],
                                "right_yaw_deg": entries[right]["yaw"],
                                "angle_deg": matrix_difference_deg(first, second),
                                "start_window": "first_15_formal_action_frames_0.5s"
                                if metric != "R_MI"
                                else "saved_meta_calibration_iphone_R_MI",
                            }
                        )
    return rows


def raw_calibration_summary(capture: dict, meta: dict, motion: dict[str, list[dict]] | None) -> dict:
    """Recompute diagnostics from raw samples without changing saved labels."""
    if motion is None:
        return {"raw_calibration_available": False, "raw_calibration_reason": "raw motion unavailable"}
    result = {"raw_calibration_available": True}
    for source in train.DEVICES:
        source_name = "iphone" if source == "iphone" else "watch"
        npose_start, npose_end = train.window_bounds(capture, "npose")
        walk_start, walk_end = train.window_bounds(capture, "forward_walk")
        npose = train.samples_between(motion[source], npose_start, npose_end)
        walk = train.samples_between(motion[source], walk_start, walk_end)
        result[f"raw_{source_name}_npose_count"] = len(npose)
        result[f"raw_{source_name}_walk_count"] = len(walk)
        if npose:
            ris = train.average_ris(npose)
            quality = train.npose_quality(npose, ris)
            result[f"raw_{source_name}_npose_max_orientation_deviation_deg"] = quality.get(
                "max_orientation_deviation_deg"
            )
            result[f"raw_{source_name}_npose_acceleration_rms_m_s2"] = quality.get(
                "acceleration_rms_m_s2"
            )
        try:
            integrated = train.integrate_walk(walk, walk_end - walk_start)
        except (TypeError, ValueError):
            integrated = None
        if integrated is None:
            result[f"raw_{source_name}_walk_horizontal_distance_m"] = None
            result[f"raw_{source_name}_rmi_vs_saved_deg"] = None
            continue
        position, velocity, duration_s = integrated
        corrected = position - 0.5 * duration_s * velocity
        built = train.rmi_from_displacement(corrected)
        horizontal = corrected - torch.dot(corrected, train.APPLE_I_UP_RH) * train.APPLE_I_UP_RH
        result[f"raw_{source_name}_walk_horizontal_distance_m"] = float(horizontal.norm())
        saved = meta.get("calibration", {}).get(source, {}).get("R_MI")
        result[f"raw_{source_name}_rmi_vs_saved_deg"] = (
            matrix_difference_deg(built[0].numpy(), saved) if built is not None else None
        )
    return result


def audit_take(directory: Path, subject: str) -> dict:
    capture = read_json(directory / "capture.json")
    meta = read_json(directory / "meta.json")
    quality = read_json(directory / "quality.json")
    labels = torch.load(directory / "labels.pt", map_location="cpu", weights_only=False)
    experiment = meta.get("experiment", capture.get("experiment", {}))
    orientation = labels["orientation"].numpy()
    acceleration = labels["acceleration"].numpy()
    motion_valid = labels["motion_valid"].numpy().astype(bool)
    uwb_valid = labels["uwb_valid"].numpy().astype(bool)
    motion, raw_error = load_raw_motion(capture)
    rows = {
        "subject": subject,
        "sequence": directory.relative_to(BASE / "data" / "datasets" / "subjects" / subject).as_posix(),
        "action": labels.get("action"),
        "action_room_yaw_deg": experiment.get(
            "action_room_yaw_deg", experiment.get("room_body_yaw_deg")
        ),
        "calibration_room_yaw_deg": experiment.get("calibration_room_yaw_deg"),
        "protocol_id": experiment.get("protocol_id"),
        "protocol_condition": (
            "calibration_and_action_yaw_move_together"
            if experiment.get("protocol_id") == "orientation_sensitivity_retest_v1"
            else "calibration_fixed_at_room_0_action_yaw_varies"
            if experiment.get("protocol_id")
            == "orientation_sensitivity_retest_v2_fixed_calibration_reference"
            else "unknown"
        ),
        "frame_count": int(orientation.shape[0]) if orientation.ndim else 0,
        "orientation_shape": str(tuple(orientation.shape)),
        "acceleration_shape": str(tuple(acceleration.shape)),
        "all_imu_valid": bool(motion_valid.all()),
        "imu_valid_ratio": float(motion_valid.mean()),
        "uwb_valid_ratio": float(uwb_valid.mean()),
        "output_frame_count_quality": quality.get("output_frame_count"),
        "capture_status": capture.get("status"),
        "time_offsets_s": capture.get("time_offsets_s"),
        "time_offset_resolution": capture.get("time_offset_resolution"),
        "npose_window": capture.get("windows", {}).get("npose"),
        "forward_walk_window": capture.get("windows", {}).get("forward_walk"),
        "action_window": capture.get("windows", {}).get("action"),
        "action_window_contamination_verifiable": False,
        "action_window_note": (
            "The saved manifest has only one formal action window; no event-level "
            "marker distinguishes preparation, turn-back, or turn segments."
        ),
        "raw_files_present": all(
            bool(capture.get("raw_files", {}).get(source))
            for source in ("iphone", "apple_watch")
        ),
        "calibration_window_reference_present": bool(
            capture.get("calibration_window_reference")
        ),
        "attitude_reference_frame_status": REFERENCE_FRAME_STATUS["status"],
        "attitude_reference_frame_reason": REFERENCE_FRAME_STATUS["reason"],
        "cross_record_R_MI_interpretation": REFERENCE_FRAME_STATUS[
            "cross_record_R_MI_interpretation"
        ],
        "online_offline_R_MI_note": (
            "Agreement only establishes reproducibility of the saved calibration "
            "calculation; it does not establish a common physical heading across recordings."
        ),
    }
    calibration = meta.get("calibration", {})
    phone = calibration.get("iphone", {})
    watch = calibration.get("apple_watch", {})
    phone_rmi = np.asarray(phone.get("R_MI"), dtype=float)
    watch_rmi = np.asarray(watch.get("R_MI"), dtype=float)
    rows.update(
        {
            "phone_R_MI_present": finite_matrix(phone.get("R_MI")),
            "watch_R_MI_present": finite_matrix(watch.get("R_MI")),
            "phone_watch_R_MI_equal": bool(
                phone_rmi.shape == watch_rmi.shape
                and np.allclose(phone_rmi, watch_rmi, atol=1e-6, rtol=1e-6)
            ),
            "watch_independent_R_MI_status": watch.get(
                "independent_R_MI_status", "not_recorded"
            ),
            "phone_npose_sample_count": phone.get("npose_sample_count"),
            "phone_walk_sample_count": phone.get("walk_sample_count"),
            "watch_npose_sample_count": watch.get("independent_R_MI_details", {})
            .get("window_selection", {})
            .get("npose", {})
            .get("sample_count"),
            "watch_walk_sample_count": watch.get("independent_R_MI_details", {})
            .get("window_selection", {})
            .get("forward_walk", {})
            .get("sample_count"),
        }
    )
    rows.update(raw_calibration_summary(capture, meta, motion))
    if motion is not None:
        npose_start, npose_end = train.window_bounds(capture, "npose")
        walk_start, walk_end = train.window_bounds(capture, "forward_walk")
        phone_npose, phone_walk, selection, selection_error = selected_phone_calibration(
            directory, capture, motion
        )
        watch_npose = train.samples_between(motion["apple_watch"], npose_start, npose_end)
        watch_walk = train.samples_between(motion["apple_watch"], walk_start, walk_end)
        rows["phone_calibration_sample_selection"] = selection
        rows["phone_calibration_sample_selection_error"] = selection_error
        shared_rmi = np.asarray(phone.get("R_MI"), dtype=float)
        phone_npose_pose = model_pose(phone_npose, shared_rmi)
        watch_npose_shared_pose = model_pose(watch_npose, shared_rmi)
        phone_start_pose = robust_start_rotation(orientation, 1)
        watch_start_pose = robust_start_rotation(orientation, 0)
        rows["phone_npose_R_MS_shared"] = phone_npose_pose
        rows["watch_npose_R_MS_shared"] = watch_npose_shared_pose
        rows["phone_Npose_to_action_start_R_MS_deg"] = matrix_difference_deg(
            phone_npose_pose, phone_start_pose
        )
        rows["watch_Npose_to_action_start_shared_R_MS_deg"] = matrix_difference_deg(
            watch_npose_shared_pose, watch_start_pose
        )
        yaw = int(rows["action_room_yaw_deg"] or 0) % 360
        if rows["protocol_condition"] == "calibration_and_action_yaw_move_together":
            expected_phone_turn_deg = 0.0
        elif rows["protocol_condition"] == "calibration_fixed_at_room_0_action_yaw_varies":
            expected_phone_turn_deg = float(min(yaw, 360 - yaw))
        else:
            expected_phone_turn_deg = None
        observed_phone_turn_deg = rows["phone_Npose_to_action_start_R_MS_deg"]
        rows["expected_phone_Npose_to_action_turn_magnitude_deg"] = expected_phone_turn_deg
        rows["phone_Npose_to_action_turn_magnitude_error_deg"] = (
            abs(observed_phone_turn_deg - expected_phone_turn_deg)
            if observed_phone_turn_deg is not None and expected_phone_turn_deg is not None
            else None
        )
        rows["watch_independent_R_MI_status_recomputed"] = "failed"
        try:
            watch_calibration = train.compute_device_calibration(
                watch_npose,
                watch_walk,
                walk_end - walk_start,
                "raw PC Unix calibration window",
                "apple_watch independent diagnostic",
                source="apple_watch",
            )
        except train.CalibrationError as error:
            rows["watch_independent_R_MI_failure_reason_recomputed"] = str(error)
            rows["watch_npose_R_MS_independent"] = None
        else:
            independent_rmi = watch_calibration["R_MI"].numpy()
            rows["watch_independent_R_MI_status_recomputed"] = "passed"
            rows["watch_independent_R_MI_recomputed"] = independent_rmi
            rows["watch_independent_vs_shared_R_MI_deg"] = matrix_difference_deg(
                independent_rmi, shared_rmi
            )
            watch_npose_independent_pose = model_pose(watch_npose, independent_rmi)
            rows["watch_npose_R_MS_independent"] = watch_npose_independent_pose
            rows["watch_Npose_to_action_start_independent_R_MS_deg"] = matrix_difference_deg(
                watch_npose_independent_pose, watch_start_pose
            )
        sensitivity = window_sensitivity(motion["iphone"], capture)
        rows["phone_walk_window_sensitivity"] = sensitivity
        rows["phone_walk_window_sensitivity_max_R_MI_deg"] = sensitivity[
            "R_MI_pairwise_max_deg"
        ]
        rows["phone_walk_window_sensitivity_median_R_MI_deg"] = sensitivity[
            "R_MI_pairwise_median_deg"
        ]
        rows["phone_walk_window_sensitivity_distance_range_m"] = [
            sensitivity["horizontal_distance_min_m"],
            sensitivity["horizontal_distance_max_m"],
        ]
        requested_walk_s = float(capture["windows"]["forward_walk"].get(
            "requested_duration_s", walk_end - walk_start
        ))
        current_quality = train.forward_walk_quality(phone_walk, requested_walk_s)
        rows["phone_forward_walk_quality_recomputed"] = current_quality
        rows["phone_forward_walk_quality_passes_current_rules"] = current_quality["ok"]
        rows["phone_forward_walk_quality_errors"] = current_quality.get("errors", [])
    else:
        rows["raw_motion_error"] = raw_error
        rows["phone_npose_R_MS_shared"] = None
        rows["watch_npose_R_MS_shared"] = None
        rows["watch_npose_R_MS_independent"] = None
    if orientation.ndim == 4 and orientation.shape[1] == 2:
        relative = np.matmul(orientation[:, 1], np.swapaxes(orientation[:, 0], -1, -2))
        rows["watch_phone_relative_angle_median_deg"] = float(
            np.median(rotation_angle_deg(relative))
        )
        rows["watch_phone_relative_angle_p95_deg"] = float(
            np.percentile(rotation_angle_deg(relative), 95)
        )
        for device, name in enumerate(("watch", "iphone")):
            rows[f"{name}_action_rotation_from_start_median_deg"] = mean_rotation_angle_deg(
                orientation[:, device]
            )
            rows[f"{name}_accel_rms_m_s2"] = float(
                np.sqrt(np.mean(np.square(acceleration[:, device])))
            )
    else:
        rows["watch_phone_relative_angle_median_deg"] = None
        rows["watch_phone_relative_angle_p95_deg"] = None
    rows["prediction_artifacts_found"] = False
    return rows


def collect_rows() -> list[dict]:
    rows = []
    for subject in SUBJECTS:
        root = BASE / "data" / "datasets" / "subjects" / subject / "orientation_retest_labels"
        for labels_path in sorted(root.glob("yaw_*/*/lw_lp/g01_screen_in/labels.pt")):
            rows.append(audit_take(labels_path.parent, subject))
    return rows


def compare_npose_and_start_rms(rows: list[dict]) -> list[dict]:
    """Compare calibrated still pose and formal-action onset in model coordinates."""
    comparisons = []
    def labels_path(row: dict) -> Path:
        return (
            BASE / "data" / "datasets" / "subjects" / row["subject"]
            / row["sequence"] / "labels.pt"
        )
    for subject in SUBJECTS:
        for action in ("Boxing", "Waving", "Walking"):
            entries = sorted(
                [row for row in rows if row["subject"] == subject and row["action"] == action],
                key=lambda row: row["action_room_yaw_deg"],
            )
            for metric, key in (
                ("N-pose_R_MS_shared_phone", "phone_npose_R_MS_shared"),
                ("N-pose_R_MS_shared_watch", "watch_npose_R_MS_shared"),
                ("N-pose_R_MS_independent_watch", "watch_npose_R_MS_independent"),
            ):
                for left, first in enumerate(entries):
                    for second in entries[left + 1 :]:
                        angle = matrix_difference_deg(first.get(key), second.get(key))
                        if angle is not None:
                            comparisons.append(
                                {
                                    "subject": subject,
                                    "action": action,
                                    "metric": metric,
                                    "left_yaw_deg": first["action_room_yaw_deg"],
                                    "right_yaw_deg": second["action_room_yaw_deg"],
                                    "angle_deg": angle,
                                    "window": "raw N-pose calibration samples",
                                }
                            )
            for metric, device in (("Action-start_R_MS_phone", 1), ("Action-start_R_MS_watch", 0)):
                for left, first in enumerate(entries):
                    first_labels = torch.load(
                        labels_path(first),
                        map_location="cpu",
                        weights_only=False,
                    )
                    first_pose = robust_start_rotation(first_labels["orientation"].numpy(), device)
                    for second in entries[left + 1 :]:
                        second_labels = torch.load(
                            labels_path(second),
                            map_location="cpu",
                            weights_only=False,
                        )
                        second_pose = robust_start_rotation(second_labels["orientation"].numpy(), device)
                        comparisons.append(
                            {
                                "subject": subject,
                                "action": action,
                                "metric": metric,
                                "left_yaw_deg": first["action_room_yaw_deg"],
                                "right_yaw_deg": second["action_room_yaw_deg"],
                                "angle_deg": matrix_difference_deg(first_pose, second_pose),
                                "window": "first 15 formal-action frames (0.5 s)",
                            }
                        )
    return comparisons


def series_for_boxing(subject: str, yaw: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directory = (
        BASE
        / "data"
        / "datasets"
        / "subjects"
        / subject
        / "orientation_retest_labels"
        / f"yaw_{yaw:03d}"
        / "Boxing"
        / "lw_lp"
        / "g01_screen_in"
    )
    labels = torch.load(directory / "labels.pt", map_location="cpu", weights_only=False)
    return (
        labels["orientation"].numpy(),
        labels["acceleration"].numpy(),
        labels["uwb_distance_m"].numpy(),
        labels["uwb_valid"].numpy().astype(bool),
    )


def plot_boxing_comparison() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 2, figsize=(13, 10), sharex="col")
    time = np.arange(600) / 30.0
    for column, yaw in enumerate((90, 180)):
        orientation, acceleration, uwb, valid = series_for_boxing("p02", yaw)
        for device, name in enumerate(("Watch", "iPhone")):
            relative = np.matmul(
                orientation[:, device], np.swapaxes(orientation[0:1, device], -1, -2)
            )
            axes[0, column].plot(
                time,
                rotation_angle_deg(relative),
                label=f"{name} relative rotation",
            )
        axes[0, column].set_title(f"p02 Boxing, room yaw {yaw}°")
        axes[0, column].set_ylabel("rotation from action start (deg)")
        axes[0, column].legend(fontsize=8)
        axes[1, column].plot(time, np.linalg.norm(acceleration[:, 0], axis=1), label="Watch")
        axes[1, column].plot(time, np.linalg.norm(acceleration[:, 1], axis=1), label="iPhone")
        axes[1, column].set_ylabel("acceleration norm (m/s²)")
        axes[1, column].legend(fontsize=8)
        for device, name in enumerate(("Watch", "iPhone")):
            values = np.where(valid[:, device], uwb[:, device], np.nan)
            axes[2, column].plot(time, values, label=name)
        axes[2, column].set_ylabel("UWB distance (m)")
        axes[2, column].set_xlabel("formal action time (s)")
        axes[2, column].legend(fontsize=8)
    figure.suptitle(
        "p02 Boxing input audit: 90° versus 180°\n"
        "No prediction probabilities were found in this project; this figure is input-only."
    )
    figure.tight_layout()
    path = OUT / "p02_boxing_yaw090_vs_yaw180_inputs.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return str(path)


def boxing_walk_trace(subject: str, yaw: int) -> tuple[dict | None, dict | None]:
    directory = (
        BASE / "data" / "datasets" / "subjects" / subject / "orientation_retest_labels"
        / f"yaw_{yaw:03d}" / "Boxing" / "lw_lp" / "g01_screen_in"
    )
    capture = read_json(directory / "capture.json")
    motion, _ = load_raw_motion(capture)
    if motion is None:
        return None, None
    start, end = train.window_bounds(capture, "forward_walk")
    span = end - start
    return (
        integrate_walk_trace(train.samples_between(motion["iphone"], start, end), span),
        integrate_walk_trace(train.samples_between(motion["apple_watch"], start, end), span),
    )


def plot_boxing_calibration_diagnostics() -> str:
    """Plot raw forward-walk IMU integration for the requested diagnostic cases."""
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [("p01", yaw) for yaw in YAW_VALUES] + [("p02", 90), ("p02", 180)]
    figure, axes = plt.subplots(3, len(cases), figsize=(22, 9), sharex="col")
    for column, (subject, yaw) in enumerate(cases):
        phone, watch = boxing_walk_trace(subject, yaw)
        axes[0, column].set_title(f"{subject} Boxing {yaw}°")
        for trace, name in ((phone, "iPhone"), (watch, "Watch")):
            if trace is None:
                continue
            axes[0, column].plot(trace["time_s"], trace["horizontal_acceleration_m_s2"], label=name)
            axes[1, column].plot(trace["time_s"], trace["horizontal_velocity_m_s"], label=name)
            axes[2, column].plot(trace["time_s"], trace["horizontal_position_m"], label=name)
            corrected = trace["corrected"]
            horizontal = corrected - torch.dot(corrected, train.APPLE_I_UP_RH) * train.APPLE_I_UP_RH
            axes[2, column].axhline(float(horizontal.norm()), linestyle="--", linewidth=0.8)
        axes[0, column].legend(fontsize=7)
        axes[2, column].set_xlabel("forward-walk time (s)")
    axes[0, 0].set_ylabel("horizontal accel. (m/s²)")
    axes[1, 0].set_ylabel("horizontal speed (m/s)")
    axes[2, 0].set_ylabel("uncorrected horizontal position (m)")
    figure.suptitle(
        "Forward-walk calibration diagnostics: solid traces are raw integration; "
        "dashed lines are terminal drift-corrected horizontal displacement"
    )
    figure.tight_layout()
    path = OUT / "boxing_forward_walk_calibration_diagnostics_v2.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return str(path)


def protocol_aware_summary(rows: list[dict]) -> dict:
    """Summarize only comparisons that are meaningful under each protocol."""
    result = {}
    for subject in SUBJECTS:
        selected = [row for row in rows if row["subject"] == subject]
        protocol = selected[0]["protocol_condition"] if selected else "unknown"
        turn_errors = [
            row["phone_Npose_to_action_turn_magnitude_error_deg"]
            for row in selected
            if row.get("phone_Npose_to_action_turn_magnitude_error_deg") is not None
        ]
        passes = [
            row.get("phone_forward_walk_quality_passes_current_rules")
            for row in selected
            if row.get("phone_forward_walk_quality_passes_current_rules") is not None
        ]
        error_counts = {}
        for row in selected:
            for error in row.get("phone_forward_walk_quality_errors", []):
                error_counts[error] = error_counts.get(error, 0) + 1
        result[subject] = {
            "protocol_condition": protocol,
            "record_count": len(selected),
            "phone_R_MS_turn_check": {
                "meaning": (
                    "expected near 0° because calibration and action direction move together"
                    if protocol == "calibration_and_action_yaw_move_together"
                    else "expected magnitude is 0°, 90°, 180°, 90° after turning from fixed room-0 calibration"
                ),
                "median_absolute_error_deg": float(np.median(turn_errors)) if turn_errors else None,
                "max_absolute_error_deg": float(max(turn_errors)) if turn_errors else None,
                "within_25_deg_count": sum(error <= 25.0 for error in turn_errors),
                "evaluated_count": len(turn_errors),
            },
            "phone_forward_walk_fragility": {
                "passes_current_live_rules": sum(bool(value) for value in passes),
                "fails_current_live_rules": sum(not bool(value) for value in passes),
                "evaluated_count": len(passes),
                "failure_reasons": error_counts,
            },
        }
    return result


def main() -> None:
    rows = collect_rows()
    comparison_rows = compare_rmi_and_start_rms()
    pose_comparison_rows = compare_npose_and_start_rms(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "orientation_retest_audit.json"
    csv_path = OUT / "orientation_retest_audit.csv"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plot_path = plot_boxing_comparison()
    calibration_plot_path = plot_boxing_calibration_diagnostics()
    summary = {
        "record_count": len(rows),
        "by_subject": {subject: sum(row["subject"] == subject for row in rows) for subject in SUBJECTS},
        "prediction_artifacts_found": False,
        "prediction_note": "Only checkpoints were found; no p02 Boxing prediction probability/time-series files were found.",
        "outputs": {
            "json": str(json_path),
            "csv": str(csv_path),
            "boxing_input_plot": plot_path,
            "boxing_calibration_plot": calibration_plot_path,
            "rmi_rms_comparison_json": str(OUT / "rmi_rms_comparison.json"),
            "rmi_rms_comparison_csv": str(OUT / "rmi_rms_comparison.csv"),
            "pose_comparison_json": str(OUT / "npose_action_start_comparison.json"),
            "pose_comparison_csv": str(OUT / "npose_action_start_comparison.csv"),
        },
        "attitude_reference_frame": REFERENCE_FRAME_STATUS,
    }
    (OUT / "rmi_rms_comparison.json").write_text(
        json.dumps(comparison_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    comparison_fields = sorted({key for row in comparison_rows for key in row})
    with (OUT / "rmi_rms_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparison_rows)
    (OUT / "npose_action_start_comparison.json").write_text(
        json.dumps(pose_comparison_rows, ensure_ascii=False, indent=2, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    pose_fields = sorted({key for row in pose_comparison_rows for key in row})
    with (OUT / "npose_action_start_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=pose_fields)
        writer.writeheader()
        writer.writerows(pose_comparison_rows)
    comparison_summary = {}
    for row in comparison_rows:
        key = (row["subject"], row["action"], row["metric"])
        comparison_summary.setdefault(key, []).append(row["angle_deg"])
    summary["rmi_rms_pairwise_summary"] = {
        f"{subject}/{action}/{metric}": {
            "median_deg": float(np.median(values)),
            "max_deg": float(max(values)),
            "min_deg": float(min(values)),
        }
        for (subject, action, metric), values in comparison_summary.items()
    }
    pose_summary = {}
    for row in pose_comparison_rows:
        key = (row["subject"], row["action"], row["metric"])
        pose_summary.setdefault(key, []).append(row["angle_deg"])
    summary["npose_action_start_pairwise_summary"] = {
        f"{subject}/{action}/{metric}": {
            "median_deg": float(np.median(values)),
            "max_deg": float(max(values)),
            "min_deg": float(min(values)),
        }
        for (subject, action, metric), values in pose_summary.items()
    }
    summary["protocol_aware_summary"] = protocol_aware_summary(rows)
    focus = {
        str(row["action_room_yaw_deg"]): row
        for row in rows
        if row["subject"] == "p02" and row["action"] == "Boxing"
    }
    summary["p02_boxing_focus"] = {
        yaw: {
            "raw_iphone_npose_max_orientation_deviation_deg": row.get(
                "raw_iphone_npose_max_orientation_deviation_deg"
            ),
            "raw_watch_npose_max_orientation_deviation_deg": row.get(
                "raw_watch_npose_max_orientation_deviation_deg"
            ),
            "raw_iphone_walk_horizontal_distance_m": row.get(
                "raw_iphone_walk_horizontal_distance_m"
            ),
            "raw_watch_walk_horizontal_distance_m": row.get(
                "raw_watch_walk_horizontal_distance_m"
            ),
            "raw_iphone_rmi_vs_saved_deg": row.get("raw_iphone_rmi_vs_saved_deg"),
            "raw_watch_rmi_vs_saved_deg": row.get("raw_watch_rmi_vs_saved_deg"),
            "watch_phone_relative_angle_median_deg": row.get(
                "watch_phone_relative_angle_median_deg"
            ),
        }
        for yaw, row in focus.items()
        if yaw in {"90", "180"}
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    focus_lines = []
    for yaw in ("90", "180"):
        item = summary["p02_boxing_focus"].get(yaw, {})
        focus_lines.append(
            f"p02 Boxing {yaw}°: phone walk horizontal "
            f"{item.get('raw_iphone_walk_horizontal_distance_m')} m; "
            f"Watch walk horizontal {item.get('raw_watch_walk_horizontal_distance_m')} m; "
            f"phone saved-R_MI difference {item.get('raw_iphone_rmi_vs_saved_deg')}°; "
            f"Watch-vs-shared difference {item.get('raw_watch_rmi_vs_saved_deg')}°."
        )
    (OUT / "README.txt").write_text(
        "这是 p01/p02 朝向复测的标定、坐标和输入数据审计结果。\n"
        "本次未训练模型，也未将准确率反推为标定结论。\n"
        "p02 Boxing 图只展示 90° 与 180° 的两设备姿态、加速度和 UWB 输入；项目内未找到对应预测概率文件。\n"
        "本仓库未包含手机/手表端选择姿态参考系的源代码，因此跨录制 R_MI 差异只作描述，不能单独判定为标定失败。\n"
        "N-pose 与正式动作起始姿态的成对比较见 npose_action_start_comparison.json；前向走窗口 ±0.1/±0.2 秒扰动结果写在每条审计记录中。\n"
        "正式动作窗没有事件级准备/转身标记，因此无法仅凭文件证明是否混入转身。\n"
        + "\n".join(focus_lines)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
