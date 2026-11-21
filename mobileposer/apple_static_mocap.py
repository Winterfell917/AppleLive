"""Offline diagnostic: saved static RMI + opening N-pose RSB -> MobilePoser.

RXY maps Y to X. This runner preserves the current Apple receiver convention.
It never changes the original RMI artifact or bypasses its normal loader.
An explicit --allow-unverified-rmi is required for candidate/cross-session use.
"""
from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from config import amass, datasets, model_config
from sensor_apple.offline import GRAVITY, NPOSE, CalibrationError, input_paths, load_motion
from sensor_apple.alignment import build_aligned, clock_models, load_auxiliary, save_native_archive
from sensor_apple.placements import resolve_placements
from sensor_apple.static_calibration import (
    choose_npose, fit_rsb, read_saved_rmi, save_json, sha256, window_metrics,
)
from sensor_apple.mesh_video import render_mesh

BASE = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE / "configs/apple_static_lw_rp.json"


def transform_model_inputs(arrays, calibration, slot_count=5):
    n = len(arrays["timestamp_unix_ns"])
    sources = arrays["source_names"].tolist()
    a_m = np.zeros((n, slot_count, 3), np.float32)
    r_mb = np.zeros((n, slot_count, 3, 3), np.float32)
    active = np.zeros(slot_count, bool)
    a_i, a_b = [], []
    for col, source in enumerate(sources):
        entry = calibration["source_calibration"][source]
        slot = entry["slot"]
        if not 0 <= slot < slot_count or active[slot]:
            raise CalibrationError("Model slots must be unique and within the five trained input slots")
        if slot != int(arrays["source_slots"][col]):
            raise CalibrationError("Calibration and aligned source slot disagree")
        rmi, rsb = np.asarray(entry["RMI"]), np.asarray(entry["RSB"])
        ris, a_s = arrays["RIS"][:, col], arrays["user_acc_S_m_s2"][:, col]
        ai = np.einsum("nij,nj->ni", ris, a_s)
        a_m[:, slot] = np.einsum("ij,nj->ni", rmi, ai)
        r_mb[:, slot] = rmi @ ris @ rsb
        a_i.append(ai)
        a_b.append(np.einsum("ji,nj->ni", rsb, a_s))
        active[slot] = True
    if not np.isfinite(a_m).all() or not np.isfinite(r_mb).all():
        raise CalibrationError("Non-finite aligned input; no gap filling is allowed for this evaluation")
    return {"aM": torch.from_numpy(a_m), "RMB": torch.from_numpy(r_mb),
            "aI_source": torch.from_numpy(np.stack(a_i, axis=1).astype(np.float32)),
            "aB_source": torch.from_numpy(np.stack(a_b, axis=1).astype(np.float32)),
            "active_model_slots": torch.from_numpy(active)}


def plot_diagnostics(motions, origin, bounds, calibration, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex="col", layout="constrained")
    for col, (source, m) in enumerate(motions.items()):
        t = (m.unix_ns - origin) / 1e9
        mean = np.asarray(calibration["source_calibration"][source]["mean_RIS_N"])
        angles = np.rad2deg(Rotation.from_matrix(m.RIS @ mean.T).magnitude())
        rows = [np.linalg.norm(m.aS, axis=-1) / GRAVITY, np.linalg.norm(m.gyroS, axis=-1), angles]
        for ax, y in zip(axes[:, col], rows):
            ax.plot(t, y, color=("#2563a6" if source == "iphone" else "#00977f"), lw=1)
            ax.axvspan(*bounds, color="#e6af2e", alpha=.25)
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0, col].set_title(source + " | opening N-pose highlighted")
        axes[2, col].set_xlabel("Seconds from earliest device-motion sample")
    for ax, label in zip(axes[:, 0], ["User acceleration norm (g)", "Angular speed (rad/s)", "Rotation from N-pose (deg)"]):
        ax.set_ylabel(label)
    fig.suptitle("Calibration-window inspection (stillness does not identify a body pose)")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(args):
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    config = json.loads(args.config.read_text())
    slots, placement = resolve_placements(watch=config["watch_placement"], phone=config["phone_placement"])
    if sorted(slots.values()) != config["active_model_slots"] or config["model_slot_count"] != 5:
        raise CalibrationError("Experiment config contradicts placement or checkpoint input shape")
    if config["fps"] != 30:
        raise CalibrationError("This checkpoint experiment is configured and tested at 30 Hz")
    checkpoint = (BASE / config["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise CalibrationError(f"Missing checkpoint: {checkpoint}")
    files = input_paths([args.session])
    motions, load_report = load_motion(files, slots)
    session_id = next(iter(motions.values())).session_id
    rmi, original_rmi, override_reasons = read_saved_rmi(args.rmi, slots, session_id, args.allow_unverified_rmi)
    origin, bounds, quality = choose_npose(motions, config, args.npose)
    calibration = {
        "schema": "apple_static_rmi_npose_experiment_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "exploratory_not_physically_validated", "session_id": session_id,
        "source_slots": slots, "placement": placement, "source_calibration": {},
        "rmi_source": {"path": str(args.rmi.resolve()), "sha256": sha256(args.rmi),
                       "session_id": original_rmi["sessionID"], "status": original_rmi["status"]},
        "experimental_override": bool(override_reasons), "override_reasons": override_reasons,
        "origin_unix_ns": str(origin), "npose_bounds_s": bounds,
        "npose_window_unix_ns": [str(origin + round(t * 1e9)) for t in bounds],
        "npose_selection": "manual_checked" if args.npose else "quietest_valid_window_in_opening_search_interval",
        "npose_assumption": "User-declared N-pose. Stillness checks cannot verify body pose or facing direction.",
        "equations": {"RIS": "C @ stored_rotation_matrix.T @ C", "aS": "C @ user_accel_g * GRAVITY",
                      "RSB": "(RMI @ mean_RIS_N).T @ target_RMB_N",
                      "RMB": "RMI @ RIS @ RSB", "aM": "RMI @ RIS @ aS", "aB": "RSB.T @ aS"},
    }
    for s, m in motions.items():
        _, keep = window_metrics(m, origin, bounds, config["npose_thresholds"])
        rsb, mean = fit_rsb(rmi[s], m.RIS[keep], NPOSE[slots[s]])
        residual = Rotation.from_matrix(rmi[s] @ m.RIS[keep] @ rsb @ NPOSE[slots[s]].T).magnitude()
        calibration["source_calibration"][s] = {
            "slot": slots[s], "RMI": rmi[s].tolist(), "RSB": rsb.tolist(), "mean_RIS_N": mean.tolist(),
            "target_RMB_N": NPOSE[slots[s]].tolist(), "quality": quality[s],
            "npose_fit_residual_p95_deg": float(np.rad2deg(np.percentile(residual, 95))),
        }
    clocks, clock_report = clock_models(motions)
    streams, aux_report = load_auxiliary(files, motions)
    arrays, alignment_report = build_aligned(motions, streams, slots, clocks, fps=config["fps"],
        max_gap=config["max_interpolation_gap_s"], anti_alias=config["anti_alias_vectors"])
    if not arrays["joint_motion_valid"].all():
        raise CalibrationError("Missing aligned motion frames. Refusing to hide gaps with zeros or interpolation")
    processed = transform_model_inputs(arrays, calibration)
    times = (arrays["timestamp_unix_ns"] - origin) / 1e9
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "effective_config.json", config)
    save_json(out / "rmi_source_snapshot.json", original_rmi)
    save_json(out / "calibration.json", calibration)
    save_native_archive(out / "native_sensor_arrays.npz", motions, streams, placement)
    np.savez_compressed(out / "aligned_sensors.npz", **arrays)
    plot_diagnostics(motions, origin, bounds, calibration, out / "npose_diagnostics.png")
    print(f"N-pose window {bounds}; session {session_id}; {len(times)} common frames", flush=True)
    torch.set_num_threads(2)
    amass.combos_full = {config["model_combo"]: config["active_model_slots"]}
    datasets.fps = config["fps"]
    model_config.device = torch.device("cpu")
    from apple_postprocess import infer_pose
    result = infer_pose(processed, checkpoint, torch.device("cpu"))
    processed.update(result)
    if result["model_input"].shape != (len(times), 60) or not torch.isfinite(result["pose"]).all():
        raise RuntimeError("Invalid model input/output shape or values")
    processed.update(timestamp_unix_ns=torch.from_numpy(arrays["timestamp_unix_ns"]),
                     time_relative_s=torch.from_numpy(times), source_names=list(slots),
                     source_slots=slots, calibration=calibration, experiment_config=config)
    torch.save(processed, out / "mocap_result.pt")
    np.savez_compressed(out / "model_io.npz", timestamp_unix_ns=arrays["timestamp_unix_ns"],
        time_from_recording_origin_s=times, input_60d=result["model_input"].numpy(),
        pose_local_rotation=result["pose"].numpy(), acc_M_slots=processed["aM"].numpy(),
        ori_MB_slots=processed["RMB"].numpy(), acc_B_sources=processed["aB_source"].numpy(),
        active_slots=processed["active_model_slots"].numpy())
    code_files = [Path(__file__), BASE / "sensor_apple/static_calibration.py", BASE / "sensor_apple/mesh_video.py",
                  BASE / "sensor_apple/offline.py", BASE / "sensor_apple/alignment.py", BASE / "apple_postprocess.py"]
    report = {
        "session_id": session_id, "recording_path": str(args.session.resolve()), "config": config,
        "files": [{"path": str(p), "sha256": sha256(p)} for p in files],
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "implementation": {str(p.relative_to(BASE)): sha256(p) for p in code_files},
        "runtime": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "scipy", "torch", "open3d", "opencv-python")}},
        "rmi_validation": {"source_status": original_rmi["status"], "override_reasons": override_reasons,
                           "physical_accuracy_verified": False},
        "native": load_report, "auxiliary": aux_report, "clocks": clock_report, "alignment": alignment_report,
        "npose_bounds_s": bounds, "npose_quality": quality,
        "model": {"input_shape": list(result["model_input"].shape), "pose_shape": list(result["pose"].shape),
                  "lookahead_correction_frames": result["pose_lookahead_frames"],
                  "tail_padding": result["pose_tail_padding"], "device": "cpu"},
        "warnings": ["Original RMI status is retained; an explicit override never promotes it to validated.",
                     "Physical inter-device clock sync is unverified; see override_reasons for cross-session RMI reuse.",
                     "N-pose RSB fit is not independent verification of RMI. Neural priors can hide errors.",
                     "No ground-truth pose, shape, or root trajectory. Generic SMPL body with fixed root.",
                     "Existing 12-combination checkpoint, two active slots. Other input slots are all zero.",
                     "UWB, raw gyro/magnetometer, and audio are not inputs to this network.",
                     "First seconds include button interaction/settling; assess natural motion after N-pose."]}
    save_json(out / "report.json", report)
    if not args.skip_video:
        title = f"{config['watch_placement'].replace('_', ' ')} + {config['phone_placement'].replace('_', ' ')} | fixed root"
        subtitle = "Exploratory result: saved RMI is unverified" if override_reasons else "Internal checks only; no ground-truth validation"
        report["video"] = render_mesh(result["pose"], out / "mocap_mesh.mp4", times, bounds, config["fps"],
                                      config["render"], title=title, subtitle=subtitle)
        save_json(out / "report.json", report)
    print(f"Results saved to {out}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, required=True, help="One retrieved session directory")
    p.add_argument("--rmi", type=Path, required=True, help="Saved static_rmi_v1 calibration.json")
    p.add_argument("--output", type=Path, required=True, help="New output directory; never overwritten")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--npose", nargs=2, type=float, metavar=("START", "END"), help="Seconds from earliest motion sample")
    p.add_argument("--allow-unverified-rmi", action="store_true", help="Explicit diagnostic override; does not validate RMI")
    p.add_argument("--skip-video", action="store_true")
    args = p.parse_args(argv)
    try:
        run(args)
    except (CalibrationError, FileExistsError, KeyError, json.JSONDecodeError) as exc:
        p.exit(2, f"Experiment stopped: {exc}\n")


if __name__ == "__main__":
    main()
