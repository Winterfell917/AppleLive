"""Static RMI and N-pose mounting calibration, independent of model inference."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .offline import C, GRAVITY, CalibrationError, canonical, load_motion, project_rotation

STATIC_SOURCES = ("iphone", "apple_watch")
STATIC_LIMITS = {"pose_p95_deg": 2., "gyro_p95_rad_s": .05, "accel_p95_g": .05,
                 "max_gap_s": .1, "minimum_coverage": .95, "gravity_angle_deg": 5.}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_rotation(value, label):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix), 1, atol=1e-5)):
        raise CalibrationError(f"{label}: expected a finite proper rotation")
    return matrix


def read_saved_rmi(path, sources, session_id, allow_unverified=False):
    artifact = json.loads(Path(path).read_text())
    if artifact.get("schema") != "static_rmi_v1":
        raise CalibrationError("Expected a static_rmi_v1 artifact")
    if not np.array_equal(np.asarray(artifact["coordinates"]["C"]), C):
        raise CalibrationError("Saved RMI and receiver use different axis conventions")
    reasons = []
    if (not artifact.get("eligible_for_mocap_reuse", False)
            or artifact.get("status") != "passed_internal_checks"):
        reasons.append("saved_coordinate_validation_not_passed")
    if artifact.get("sessionID") != session_id:
        reasons.append("cross_session_inertial_reference_not_verified")
    if reasons and not allow_unverified:
        raise CalibrationError("RMI is not verified for this recording: " + ", ".join(reasons)
                               + ". Use --allow-unverified-rmi only for an explicitly requested diagnostic.")
    matrices = {s: checked_rotation(artifact["sources"][s]["candidate_RMI"], f"{s} RMI")
                for s in sources}
    return matrices, artifact, reasons


def window_metrics(motion, origin_ns, bounds, limits):
    t = (motion.unix_ns - origin_ns) / 1e9
    keep = (t >= bounds[0]) & (t < bounds[1])
    if keep.sum() < 10:
        return {"issues": ["fewer_than_10_samples"]}, keep
    mean = project_rotation(motion.RIS[keep].mean(axis=0))
    angle = np.rad2deg(Rotation.from_matrix(motion.RIS[keep] @ mean.T).magnitude())
    selected = t[keep]
    report = {
        "samples": int(keep.sum()), "pose_p95_deg": float(np.percentile(angle, 95)),
        "gyro_p95_rad_s": float(np.percentile(np.linalg.norm(motion.gyroS[keep], axis=-1), 95)),
        "user_acc_p95_g": float(np.percentile(np.linalg.norm(motion.aS[keep], axis=-1), 95) / GRAVITY),
        "max_gap_s": float(np.diff(selected).max()),
        "coverage": float((selected[-1] - selected[0]) / (bounds[1] - bounds[0])),
    }
    report["issues"] = [k for k in ("pose_p95_deg", "gyro_p95_rad_s", "user_acc_p95_g", "max_gap_s")
                        if report[k] > limits[k]]
    if report["coverage"] < limits["minimum_coverage"]:
        report["issues"].append("insufficient_coverage")
    return report, keep


def choose_npose(motions, config, requested=None):
    origin = min(int(m.unix_ns[0]) for m in motions.values())
    limits = config["npose_thresholds"]
    if requested is not None:
        a, b = map(float, requested)
        if not np.isfinite([a, b]).all() or a < 0 or b <= a:
            raise CalibrationError("N-pose bounds must be finite, nonnegative, and increasing")
        candidates = [[a, b]]
    else:
        duration = config["npose_window_s"]
        candidates = [[round(float(a), 9), round(float(a + duration), 9)] for a in np.arange(
            config["npose_search_start_s"], config["npose_search_end_s"] - duration + 1e-8,
            config["npose_search_step_s"])]
    accepted = []
    for bounds in candidates:
        quality = {s: window_metrics(m, origin, bounds, limits)[0] for s, m in motions.items()}
        if any(q["issues"] for q in quality.values()):
            continue
        score = max(q["pose_p95_deg"] / limits["pose_p95_deg"]
                    + q["gyro_p95_rad_s"] / limits["gyro_p95_rad_s"] for q in quality.values())
        accepted.append((score, bounds, quality))
    if not accepted:
        raise CalibrationError("No quiet opening N-pose window passes the configured checks. "
                               "Inspect the recording; do not automatically use a later natural-motion pause.")
    _, bounds, quality = min(accepted, key=lambda item: item[0])
    return origin, bounds, quality


def fit_rsb(rmi, ris_npose, target):
    """RMI * mean(RIS_N) * RSB = target, an algebraic fit, not validation."""
    mean = project_rotation(np.asarray(ris_npose).mean(axis=0))
    rsb = (rmi @ mean).T @ target
    return checked_rotation(rsb, "RSB"), mean


def load_static_recording(files):
    """Use the same checked motion parser as inference, with gravity/frame metadata.

    Integer nanoseconds are retained until subtraction. Files must contain one
    session; duplicate samples, clock jumps and invalid rotations are checked by
    load_motion. These temporary slot labels do not imply body placement.
    """
    motions, stats = load_motion(files, {s: i for i, s in enumerate(STATIC_SOURCES)})
    extras = {s: {} for s in STATIC_SOURCES}
    for path in files:
        with Path(path).open() as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                e = json.loads(line)
                s = canonical(e.get("source", ""))
                if s not in extras or e.get("sensor") != "device_motion":
                    continue
                try:
                    v, ns = e["values"], int(e["timestampUnixNs"])
                    gravity = tuple(float(v[f"gravity_{k}"]) for k in "xyz")
                    frame = float(v["attitude_reference_frame"])
                    if frame not in (1., 2., 4., 8.) or not np.isfinite(gravity).all():
                        raise ValueError("Invalid gravity or attitude reference frame")
                    value = (gravity, str(int(frame)), str(v.get("magnetic_accuracy")))
                    if ns in extras[s] and extras[s][ns] != value:
                        raise ValueError("Conflicting gravity/frame metadata at the same timestamp")
                    extras[s][ns] = value
                except (KeyError, ValueError, TypeError) as exc:
                    raise CalibrationError(f"{path}:{number}: {exc}") from exc
    origin = min(int(m.unix_ns[0]) for m in motions.values())
    streams = {}
    for s, m in motions.items():
        gravity, frames, accuracy = zip(*(extras[s][int(ns)] for ns in m.unix_ns))
        counts = dict(Counter(frames))
        if len(counts) != 1:
            raise CalibrationError(f"{s}: attitude reference frame changed within the recording")
        streams[s] = {"time": (m.unix_ns - origin) / 1e9, "RIS": m.RIS,
                      "Q": C @ m.RIS.transpose(0, 2, 1) @ C,
                      "gyro": m.gyroS, "acc": m.aS / GRAVITY, "gravity": np.asarray(gravity),
                      "frames": counts, "magnetic_accuracy": dict(Counter(accuracy)),
                      "sequence_gaps": int(np.maximum(np.diff(m.sequence) - 1, 0).sum())}
    return next(iter(motions.values())).session_id, origin, streams, stats


def static_metrics(data, bounds, limits=STATIC_LIMITS):
    start, end = bounds
    keep = (data["time"] >= start) & (data["time"] < end)
    if keep.sum() < 10:
        return {"issues": ["insufficient_samples"]}, keep
    time = data["time"][keep]
    mean = project_rotation(data["RIS"][keep].mean(axis=0))
    # Keep the original static-RMI experiment's trace/SVD convention.
    angle = np.degrees(np.arccos(np.clip((np.trace(data["RIS"][keep] @ mean.T, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    result = {"samples": int(keep.sum()), "pose_p95_deg": float(np.percentile(angle, 95)),
              "gyro_p95_rad_s": float(np.percentile(np.linalg.norm(data["gyro"][keep], axis=1), 95)),
              "accel_p95_g": float(np.percentile(np.linalg.norm(data["acc"][keep], axis=1), 95)),
              "max_gap_s": float(np.diff(time).max()), "coverage": float((time[-1] - time[0]) / (end - start))}
    result["issues"] = [k for k in ("pose_p95_deg", "gyro_p95_rad_s", "accel_p95_g", "max_gap_s")
                        if result[k] > limits[k]]
    if result["coverage"] < limits["minimum_coverage"]:
        result["issues"].append("insufficient_coverage")
    return result, keep


def select_static_window(streams, seconds=8., edge=1., requested=None):
    if not np.isfinite([seconds, edge]).all() or not 5 <= seconds <= 10 or edge < 0:
        raise CalibrationError("Static window must be 5..10 seconds; edge exclusion must be nonnegative")
    if requested is not None:
        a, b = map(float, requested)
        if not np.isfinite([a, b]).all() or a < 0 or not 5 <= b - a <= 10:
            raise CalibrationError("Manual static bounds must be nonnegative and span 5..10 seconds")
        candidates = [[a, b]]
    else:
        lo = max(d["time"][0] for d in streams.values()) + edge
        hi = min(d["time"][-1] for d in streams.values()) - edge - seconds
        candidates = [[float(a), float(a + seconds)] for a in np.arange(lo, hi + 1e-8, .25)]
    accepted = []
    for bounds in candidates:
        rows = [static_metrics(d, bounds)[0] for d in streams.values()]
        if any(r["issues"] for r in rows):
            continue
        score = max(r["pose_p95_deg"] / STATIC_LIMITS["pose_p95_deg"]
                    + r["gyro_p95_rad_s"] / STATIC_LIMITS["gyro_p95_rad_s"] for r in rows)
        accepted.append((score, bounds))
    if not accepted:
        raise CalibrationError("No common static interval passes motion/coverage checks")
    return min(accepted, key=lambda item: item[0])[1]


def calculate_static_rmi(data, bounds, target):
    target = checked_rotation(target, "RMS target")
    quality, keep = static_metrics(data, bounds)
    if quality["issues"]:
        raise CalibrationError("Static interval failed: " + str(quality["issues"]))
    mean = project_rotation(data["RIS"][keep].mean(axis=0))
    rmi = checked_rotation(target @ mean.T, "RMI")
    gravity_s = data["gravity"][keep] @ C.T
    gravity_m = np.einsum("ij,njk,nk->ni", rmi, data["RIS"][keep], gravity_s)
    norm = np.linalg.norm(gravity_m, axis=1)
    if np.any(norm < .5) or np.any(norm > 1.5):
        raise CalibrationError("Invalid gravity magnitude")
    down_error = np.degrees(np.arccos(np.clip(-gravity_m[:, 1] / norm, -1, 1)))
    p95 = float(np.percentile(down_error, 95))
    closure = rmi @ mean @ target.T
    return {"candidate_RMI": rmi.tolist(), "mean_RIS": mean.tolist(), "quality": quality,
            "RMS_target": target.tolist(), "gravity_raw_mean_g": data["gravity"][keep].mean(axis=0).tolist(),
            "gravity_model_mean_g": gravity_m.mean(axis=0).tolist(),
            "gravity_error_to_down_p95_deg": p95,
            "validation_issues": [] if p95 <= STATIC_LIMITS["gravity_angle_deg"] else ["gravity_does_not_point_model_minus_y"],
            "algebraic_closure_error_deg": float(np.degrees(np.arccos(np.clip((np.trace(closure) - 1) / 2, -1, 1))))}
