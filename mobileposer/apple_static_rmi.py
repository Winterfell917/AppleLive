"""Estimate each device's RMI from a known fixed orientation, without walking.

Requires an explicit pose config: columns of RMS are processed sensor axes in
model coordinates. A stable recording alone cannot determine these directions.
Exit 2 means diagnostics were saved but coordinate validation did not pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from sensor_apple.offline import C, CalibrationError, input_paths
from sensor_apple.static_calibration import (
    STATIC_LIMITS, STATIC_SOURCES, calculate_static_rmi, checked_rotation,
    load_static_recording, save_json, select_static_window, sha256,
)


def plot_static(streams, bounds, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, layout="constrained")
    for col, (source, data) in enumerate(streams.items()):
        for k, label in enumerate("XYZ"):
            axes[0, col].plot(data["time"], data["gravity"][:, k], lw=1, label=label)
        axes[0, col].legend(ncol=3)
        axes[1, col].plot(data["time"], np.linalg.norm(data["gyro"], axis=1), lw=1)
        axes[2, col].plot(data["time"], np.linalg.norm(data["acc"], axis=1), lw=1)
        axes[0, col].set_title(source)
        for ax, label in zip(axes[:, col], ["Recorded gravity (g)", "Angular speed (rad/s)", "User acceleration norm (g)"]):
            ax.axvspan(*bounds, color="#54a284", alpha=.2)
            ax.grid(alpha=.2)
            ax.set_ylabel(label)
        axes[2, col].set_xlabel("Seconds from earliest device-motion sample")
    fig.suptitle("Static RMI | shaded: selected interval | stationarity and gravity checked separately")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(args):
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    config = json.loads(args.pose_config.read_text())
    if config.get("schema") != "apple_static_rmi_pose_v1" or not np.array_equal(config.get("C"), C):
        raise CalibrationError("Pose config schema or coordinate convention does not match the receiver")
    targets = {s: checked_rotation(config["RMS_targets"][s], f"{s} RMS target") for s in STATIC_SOURCES}
    files = input_paths([args.session])
    sid, origin, streams, load_report = load_static_recording(files)
    bounds = select_static_window(streams, args.window_seconds, requested=args.window)
    rows = {s: calculate_static_rmi(d, bounds, targets[s]) for s, d in streams.items()}
    blocked = any(r["validation_issues"] for r in rows.values())
    for s, data in streams.items():
        raw_gi = np.einsum("nji,nj->ni", data["Q"], data["gravity"])
        rows[s].update(attitude_reference_frame_counts=data["frames"], magnetic_accuracy_counts=data["magnetic_accuracy"],
                       samples=len(data["time"]), sequence_gaps=data["sequence_gaps"],
                       median_rate_hz=float(1 / np.median(np.diff(data["time"]))),
                       gravity_raw_reference_residual_p95_g=float(np.percentile(np.linalg.norm(raw_gi - [0, 0, -1], axis=1), 95)))
    report = {
        "schema": "static_rmi_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "sessionID": sid, "session_path": str(args.session.resolve()),
        "status": "blocked_coordinate_validation" if blocked else "passed_internal_checks",
        "eligible_for_mocap_reuse": False if blocked else "same_session_only_after_physical_pose_verification",
        "window_seconds_from_origin": bounds, "origin_unix_ns": str(origin),
        "window_unix_ns": [str(origin + round(t * 1e9)) for t in bounds],
        "window_selection": "manual_checked" if args.window else "common stable interval; minimum worst normalized pose+gyro P95",
        "thresholds": STATIC_LIMITS, "coordinates": {
            "RXY": "maps Y to X", "C": C.tolist(), "RMS_targets": {s: r.tolist() for s, r in targets.items()},
            "RIS": "C @ stored_rotation_matrix.T @ C", "RMI": "RMS_target @ mean_RIS.T",
            "placement": config.get("sensor_pose", "See pose config snapshot"),
        },
        "pose_config": {"path": str(args.pose_config.resolve()), "sha256": sha256(args.pose_config)},
        "sources": rows, "load_report": load_report,
        "input_files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files},
        "warnings": ["RMS columns must be independently established; the example target is not a universal Apple convention.",
                     "Gravity checks tilt/signs, not absolute heading. Algebraic closure is not independent validation.",
                     "Do not reuse across initialization/session changes without independent reference-frame checks, even in magnetic-north mode.",
                     "This artifact contains RMI only. Fit RSB from the worn N-pose in the mocap recording."]}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "calibration.json", report)
    save_json(out / "pose_config_snapshot.json", config)
    np.savez_compressed(out / "candidate_rmi.npz", sources=np.array(STATIC_SOURCES),
                        candidate_RMI=np.stack([rows[s]["candidate_RMI"] for s in STATIC_SOURCES]),
                        RMS_targets=np.stack([targets[s] for s in STATIC_SOURCES]), C=C,
                        sessionID=np.array(sid), status=np.array(report["status"]))
    plot_static(streams, bounds, out / "diagnostic.png")
    print(f"RMI diagnostics saved: {out}\nStatus: {report['status']}\nWindow: {bounds}")
    return 2 if blocked else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="One session directory with both devices' NDJSON files")
    parser.add_argument("--pose-config", type=Path, required=True, help="JSON containing per-device RMS_targets")
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrite")
    parser.add_argument("--window-seconds", type=float, default=8., help="Automatic static window length, 5..10 seconds")
    parser.add_argument("--window", nargs=2, type=float, metavar=("START", "END"), help="Manual checked interval in seconds from earliest motion sample")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (CalibrationError, FileExistsError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Static RMI stopped: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
