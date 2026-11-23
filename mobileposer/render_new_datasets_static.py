"""Render legacy new_datasets recordings with separate 0rmi and 0rsb sessions.

This is the compatibility runner for the collection protocol used before every
action recording began with an N-pose.  It uses the same static RMI, N-pose
RSB, timestamp alignment, MobilePoser inference, and Open3D mesh renderer as
apple_static_rmi.py and apple_static_mocap.py.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from apple_postprocess import infer_pose
from apple_static_mocap import transform_model_inputs
from config import amass, datasets, model_config
from sensor_apple.alignment import build_aligned, clock_models, load_auxiliary
from sensor_apple.mesh_video import render_mesh
from sensor_apple.offline import NPOSE, CalibrationError, input_paths, load_motion
from sensor_apple.placements import resolve_placements
from sensor_apple.static_calibration import (
    choose_npose,
    fit_rsb,
    read_saved_rmi,
    save_json,
    sha256,
    window_metrics,
)


BASE = Path(__file__).resolve().parent
DEFAULT_ROOT = BASE / "data/new_datasets"
DEFAULT_CONFIG = BASE / "configs/apple_static_lw_lp.json"


def crop_time_series(arrays: dict, keep: np.ndarray) -> dict:
    count = len(keep)
    return {
        key: value[keep] if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == count else value
        for key, value in arrays.items()
    }


def fit_separate_rsb(rmi_path: Path, session: Path, slots: dict, config: dict) -> tuple[dict, dict]:
    files = input_paths([session])
    motions, load_report = load_motion(files, slots)
    session_id = next(iter(motions.values())).session_id
    rmi, saved_rmi, reasons = read_saved_rmi(rmi_path, slots, session_id, allow_unverified=True)
    origin, bounds, quality = choose_npose(motions, config)
    calibration = {
        "schema": "new_datasets_separate_static_rmi_rsb_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "exploratory_cross_session_rmi_reuse",
        "rmi_source": {"path": str(rmi_path.resolve()), "sha256": sha256(rmi_path),
                       "session_id": saved_rmi["sessionID"], "status": saved_rmi["status"]},
        "rsb_session": {"path": str(session.resolve()), "session_id": session_id,
                        "origin_unix_ns": str(origin), "npose_bounds_s": bounds,
                        "load_report": load_report},
        "source_slots": slots,
        "source_calibration": {},
        "rmi_override_reasons": reasons,
        "equations": {"RSB": "(RMI @ mean_RIS_N).T @ target_RMB_N",
                      "RMB": "RMI @ RIS @ RSB", "aM": "RMI @ RIS @ aS"},
    }
    for source, motion in motions.items():
        _, keep = window_metrics(motion, origin, bounds, config["npose_thresholds"])
        rsb, mean = fit_rsb(rmi[source], motion.RIS[keep], NPOSE[slots[source]])
        residual = Rotation.from_matrix(
            rmi[source] @ motion.RIS[keep] @ rsb @ NPOSE[slots[source]].T
        ).magnitude()
        calibration["source_calibration"][source] = {
            "slot": slots[source], "RMI": rmi[source].tolist(), "RSB": rsb.tolist(),
            "mean_RIS_N": mean.tolist(), "target_RMB_N": NPOSE[slots[source]].tolist(),
            "quality": quality[source], "npose_fit_residual_p95_deg": float(np.rad2deg(np.percentile(residual, 95))),
        }
    return calibration, saved_rmi


def render_one(entry: dict, calibration: dict, config: dict, output: Path, render: bool) -> dict:
    raw = Path(entry["raw_folder"])
    files = input_paths([raw])
    motions, load_report = load_motion(files, calibration["source_slots"])
    clocks, clock_report = clock_models(motions)
    streams, auxiliary_report = load_auxiliary(files, motions)
    arrays, alignment_report = build_aligned(
        motions, streams, calibration["source_slots"], clocks, fps=config["fps"],
        max_gap=config["max_interpolation_gap_s"], anti_alias=config["anti_alias_vectors"],
    )
    origin = min(int(motion.unix_ns[0]) for motion in motions.values())
    time_s = (arrays["timestamp_unix_ns"] - origin) / 1e9
    keep = (time_s >= float(entry["start_s"])) & (time_s < float(entry["end_s"]))
    if keep.sum() < 20:
        raise CalibrationError("Action boundary contains fewer than 20 aligned frames")
    if not arrays["joint_motion_valid"][keep].all():
        raise CalibrationError("Action interval contains motion gaps")
    arrays = crop_time_series(arrays, keep)
    time_s = time_s[keep] - float(entry["start_s"])
    processed = transform_model_inputs(arrays, calibration)
    torch.set_num_threads(2)
    amass.combos_full = {config["model_combo"]: config["active_model_slots"]}
    datasets.fps = config["fps"]
    model_config.device = torch.device("cpu")
    checkpoint = (BASE / config["checkpoint"]).resolve()
    result = infer_pose(processed, checkpoint, torch.device("cpu"))
    processed.update(result)
    if result["model_input"].shape != (len(time_s), 60) or not torch.isfinite(result["pose"]).all():
        raise RuntimeError("Invalid MobilePoser input or prediction")
    output.mkdir(parents=True, exist_ok=False)
    effective_calibration = dict(calibration, action_session_id=next(iter(motions.values())).session_id)
    processed.update(
        timestamp_unix_ns=torch.from_numpy(arrays["timestamp_unix_ns"]),
        time_relative_s=torch.from_numpy(time_s), source_names=list(calibration["source_slots"]),
        source_slots=calibration["source_slots"], calibration=effective_calibration,
        experiment_config=config,
    )
    torch.save(processed, output / "mocap_result.pt")
    np.savez_compressed(
        output / "model_io.npz", timestamp_unix_ns=arrays["timestamp_unix_ns"],
        time_from_action_start_s=time_s, input_60d=result["model_input"].numpy(),
        pose_local_rotation=result["pose"].numpy(), acc_M_slots=processed["aM"].numpy(),
        ori_MB_slots=processed["RMB"].numpy(), active_slots=processed["active_model_slots"].numpy(),
    )
    np.savez_compressed(output / "aligned_sensors.npz", **arrays)
    save_json(output / "calibration.json", effective_calibration)
    save_json(output / "effective_config.json", config)
    report = {
        "schema": "new_datasets_static_render_v1", "entry": entry,
        "action_session_id": next(iter(motions.values())).session_id,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "rmi_validation": {"physical_accuracy_verified": False,
                           "cross_session_reuse": True,
                           "override_reasons": calibration["rmi_override_reasons"]},
        "rsb_source": calibration["rsb_session"], "native": load_report,
        "clocks": clock_report, "auxiliary": auxiliary_report, "alignment": alignment_report,
        "action_boundary": {"start_s": entry["start_s"], "end_s": entry["end_s"],
                            "approved": bool(entry.get("approved", False))},
        "model": {"input_shape": list(result["model_input"].shape),
                  "pose_shape": list(result["pose"].shape), "device": "cpu"},
        "warnings": [
            "RMI and RSB come from separate calibration sessions under the legacy new_datasets protocol.",
            "The current action session did not include an opening N-pose. Future recordings should use apple_static_mocap.py directly.",
            "Action boundaries marked approved=false still use the existing proposed review interval.",
        ],
    }
    if render:
        report["video"] = render_mesh(
            result["pose"], output / "mocap_mesh.mp4", time_s, [-1.0, -1.0], config["fps"], config["render"],
            title=f"{entry['subject']} {entry['action']} {entry['group']} | left wrist + left pocket",
            subtitle="Separate 0rmi + 0rsb calibration | cross-session reuse",
        )
    save_json(output / "report.json", report)
    return report


def render_saved(entry: dict, output: Path, config: dict) -> None:
    result = torch.load(output / "mocap_result.pt", map_location="cpu")
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    report["video"] = render_mesh(
        result["pose"], output / "mocap_mesh.mp4", result["time_relative_s"].numpy(), [-1.0, -1.0],
        config["fps"], config["render"],
        title=f"{entry['subject']} {entry['action']} {entry['group']} | left wrist + left pocket",
        subtitle="Separate 0rmi + 0rsb calibration | cross-session reuse",
    )
    save_json(report_path, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subject", action="append", choices=("p01", "p02"))
    parser.add_argument("--only", help="One review key, for example p02/lw_lp/WalkingStraight/g01")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--render", action="store_true", help="Create Open3D MP4 previews as well as model results")
    args = parser.parse_args()
    root, config = args.root.resolve(), json.loads(args.config.read_text())
    slots, placement = resolve_placements(watch=config["watch_placement"], phone=config["phone_placement"])
    if slots != {"apple_watch": 0, "iphone": 2}:
        raise CalibrationError("This compatibility runner is restricted to the collected lw_lp recordings")
    reviews = json.loads((root / "review.json").read_text())
    subjects = args.subject or ["p01", "p02"]
    summary = []
    for subject in subjects:
        wear_root = root / "raw" / subject / "lw_lp"
        rmi_path = root / "static_rmi" / subject / "lw_lp" / "calibration.json"
        if not rmi_path.is_file():
            raise CalibrationError(f"Missing static RMI: {rmi_path}")
        calibration, _ = fit_separate_rsb(rmi_path, wear_root / "0rsb", slots, config)
        calibration["placement"] = placement
        rsb_dir = root / "static_rsb" / subject / "lw_lp"
        if not rsb_dir.exists():
            rsb_dir.mkdir(parents=True)
            save_json(rsb_dir / "calibration.json", calibration)
        entries = [value for key, value in reviews.items() if key.startswith(subject + "/lw_lp/")]
        if args.only:
            entries = [value for value in entries if value["key"] == args.only]
            if not entries:
                raise CalibrationError(f"No matching entry for --only: {args.only}")
        for entry in sorted(entries, key=lambda value: (value["action"], value["group"])):
            output = root / "static_rendered" / subject / "lw_lp" / entry["action"] / f"g{entry['group']:02d}"
            if output.exists():
                if args.render and not (output / "mocap_mesh.mp4").is_file():
                    render_saved(entry, output, config)
                    summary.append({"key": entry["key"], "status": "rendered_existing"})
                    continue
                summary.append({"key": entry["key"], "status": "already_exists"})
                continue
            try:
                render_one(entry, calibration, config, output, args.render)
                summary.append({"key": entry["key"], "status": "rendered" if args.render else "processed"})
            except Exception as error:
                summary.append({"key": entry["key"], "status": "failed", "error": str(error)})
                print(f"FAILED {entry['key']}: {error}", flush=True)
    save_json(root / "static_render_report.json", summary)
    counts = {status: sum(row["status"] == status for row in summary) for status in {row["status"] for row in summary}}
    print(json.dumps(counts, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
