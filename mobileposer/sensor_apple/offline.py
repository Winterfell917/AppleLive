"""Offline walking calibration. No sockets, device access, or model inference.

The coordinate convention and integration rule intentionally match sensor.py.
Automatic detection is a protocol detector, not an action/N-pose classifier.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation

from .sensor import APPLE_LH_TO_RH, APPLE_I_UP_RH, GRAVITY, NPOSE_RMB
from .placements import describe_placements, source_rotations

C = APPLE_LH_TO_RH.numpy().astype(float)
UP = APPLE_I_UP_RH.numpy().astype(float)
NPOSE = NPOSE_RMB.numpy().astype(float)
DEFAULT_SLOTS = {"apple_watch": 0, "iphone": 3}


class CalibrationError(ValueError):
    """Unusable/ambiguous calibration, with machine-readable diagnostics."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


@dataclass
class Motion:
    source: str
    session_id: str
    unix_ns: np.ndarray
    monotonic_s: np.ndarray
    sequence: np.ndarray
    RIS: np.ndarray
    aS: np.ndarray
    gyroS: np.ndarray
    offset_s: float = 0.0

    @property
    def time(self):
        return self.unix_ns.astype(np.float64) / 1e9 + self.offset_s

    @property
    def integration_time(self):
        # Keep local monotonic dt separate from the approximate shared Unix axis.
        if np.all(np.isfinite(self.monotonic_s)):
            return self.monotonic_s
        return (self.unix_ns - self.unix_ns[0]).astype(float) / 1e9


@dataclass
class Config:
    search_start_s: float = 0.0
    search_seconds: float = 45.0
    static_seconds: float = 1.5
    static_acc_rms: float = 0.65  # m/s^2, gravity removed
    static_gyro_rms: float = 0.25  # rad/s
    min_motion_seconds: float = 0.3
    max_motion_seconds: float = 6.0
    max_gap_s: float = 0.15
    min_displacement_m: float = 0.08
    max_return_angle_deg: float = 25.0
    max_heading_sensitivity_deg: float = 15.0


def canonical(source):
    source = source.lower().removesuffix("_simulator")
    return {"watch": "apple_watch", "applewatch": "apple_watch",
            "headphones": "airpods", "headphone": "airpods"}.get(source, source)


def input_paths(inputs):
    paths = set()
    for value in inputs:
        p = Path(value).expanduser().resolve()
        if p.is_dir():
            # Deliberately one session only: never traverse all recording folders.
            paths.update(p.glob("*.ndjson"))
        elif p.is_file() and p.suffix == ".ndjson":
            paths.add(p)
        else:
            raise CalibrationError(f"Not an NDJSON file or session directory: {p}")
    if not paths:
        raise CalibrationError("No NDJSON files found")
    return sorted(paths)


def load_motion(paths, source_slots, session_id=None, offsets=None):
    rows = {s: {} for s in source_slots}
    stats = {"duplicate_samples": 0, "files": [str(p) for p in paths]}
    for path in paths:
        with Path(path).open() as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                    source = canonical(e.get("source", ""))
                    sensor = "head_motion" if source == "airpods" else "device_motion"
                    if source not in rows or e.get("sensor") != sensor:
                        continue
                    sid = str(e["sessionID"])
                    if session_id and sid != session_id:
                        continue
                    v = e["values"]
                    ris = np.array([float(v[f"rotation_m{r}{c}"])
                                    for r in range(1, 4) for c in range(1, 4)]).reshape(3, 3).T
                    acc = np.array([float(v[f"user_accel_{k}_g"]) for k in "xyz"]) * GRAVITY
                    gyro = np.array([float(v[f"rotation_{k}_rad_s"]) for k in "xyz"])
                    ns, seq = int(e["timestampUnixNs"]), int(e["sequenceNumber"])
                    mono = float(e.get("timestampMonotonicS", np.nan))
                    if not np.all(np.isfinite(np.r_[ris.ravel(), acc, gyro])):
                        raise ValueError("non-finite motion values")
                    key = (sid, ns)
                    row = (sid, ns, mono, seq, C @ ris @ C, C @ acc, -C @ gyro)
                    if key in rows[source]:
                        old = rows[source][key]
                        if old[3] != seq or not all(np.array_equal(a, b) for a, b in zip(old[4:], row[4:])):
                            raise ValueError("conflicting samples at the same source timestamp")
                        stats["duplicate_samples"] += 1
                    else:
                        rows[source][key] = row
                except (KeyError, TypeError, ValueError) as exc:
                    raise CalibrationError(f"{path}:{line_no}: {exc}") from exc
    sessions = {r[0] for values in rows.values() for r in values.values()}
    if len(sessions) != 1:
        raise CalibrationError("Expected exactly one common sessionID; select --session-id", {"session_ids": sorted(sessions)})
    result = {}
    for source, values in rows.items():
        records = sorted(values.values(), key=lambda r: r[1])
        if len(records) < 10:
            raise CalibrationError(f"Missing/insufficient {source} motion data")
        _, ns, mono, seq, ris, acc, gyro = zip(*records)
        m = Motion(source, records[0][0], np.array(ns, dtype=np.int64), np.array(mono),
                   np.array(seq, dtype=np.int64), np.array(ris), np.array(acc), np.array(gyro),
                   (offsets or {}).get(source, 0.0))
        if np.any(np.diff(m.sequence) <= 0):
            raise CalibrationError(f"{source}: sequence reset/conflict; split recording before calibrating")
        if np.any(np.diff(m.integration_time) <= 0):
            raise CalibrationError(f"{source}: non-monotonic sample clock; split recording")
        if np.max(np.abs(np.diff(m.time) - np.diff(m.integration_time))) > 0.05:
            raise CalibrationError(f"{source}: Unix/monotonic clock discontinuity; split recording")
        errors = np.linalg.norm(m.RIS @ m.RIS.transpose(0, 2, 1) - np.eye(3), axis=(1, 2))
        if np.max(errors) > 0.01 or np.min(np.linalg.det(m.RIS)) < 0.99:
            raise CalibrationError(f"{source}: invalid rotation matrices")
        result[source] = m
    stats["sources"] = {s: {"count": len(m.time), "first_timestamp_s": float(m.time[0]),
                              "last_timestamp_s": float(m.time[-1]),
                              "median_hz": float(1 / np.median(np.diff(m.integration_time))),
                              "integration_clock": "monotonic" if np.all(np.isfinite(m.monotonic_s)) else "unix_relative",
                              "max_gap_s": float(np.max(np.diff(m.integration_time)))} for s, m in result.items()}
    return result, stats


def project_rotation(matrix):
    u, _, vt = np.linalg.svd(matrix)
    u[:, -1] *= np.linalg.det(u @ vt)
    return u @ vt


def angle_deg(a, b):
    # Project away float32 roundoff before measuring SO(3) geodesic distance.
    return float(np.rad2deg(Rotation.from_matrix(project_rotation(a @ b.T)).magnitude()))


def indices(m, bounds, max_gap=None):
    t = m.time
    ix = np.flatnonzero((t >= bounds[0]) & (t <= bounds[1]))
    if len(ix) < 3:
        raise CalibrationError(f"{m.source}: fewer than 3 samples in window")
    if max_gap is not None:
        gap = max(float(t[ix[0]] - bounds[0]), float(bounds[1] - t[ix[-1]]),
                  float(np.max(np.diff(m.integration_time[ix]))))
        if gap > max_gap:
            raise CalibrationError(f"{m.source}: window gap/coverage {gap:.3f}s > {max_gap:.3f}s")
    return ix


def integrate(m, ix, legacy=False):
    t = m.time[ix] if legacy else m.integration_time[ix]
    a = np.einsum("nij,nj->ni", m.RIS[ix], m.aS[ix])
    p, v = np.zeros(3), np.zeros(3)
    for dt, acc in zip(np.diff(t), a[1:]):
        # Same right-endpoint integration and zero-terminal-velocity correction
        # as the online implementation; not a long-term trajectory estimator.
        p += dt * v + 0.5 * dt * dt * acc
        v += dt * acc
    displacement = p - 0.5 * (t[-1] - t[0]) * v
    displacement -= displacement.dot(UP) * UP
    return displacement, p, v


def rotation_from_forward(forward):
    z = forward / np.linalg.norm(forward)
    x = np.cross(UP, z)
    x /= np.linalg.norm(x)
    return np.stack([x, UP, np.cross(x, UP)])


def calibration(motions, slots, static, walk, config, post=None, paired=None, placement_info=None):
    placement_info = placement_info or describe_placements(slots)
    if placement_info["source_slots"] != slots or set(motions) != set(slots):
        raise CalibrationError("Placement/source mapping does not match the motion data")
    if not paired:
        if static[1] - static[0] < config.static_seconds - 1e-6:
            raise CalibrationError("Initial static window is too short")
        if post and post[1] - post[0] < config.static_seconds - 1e-6:
            raise CalibrationError("Final static window is too short")
    rmi = np.repeat(np.eye(3)[None], 7, axis=0)
    rsb = rmi.copy()
    per_source = {}
    quality = {}
    for source, m in motions.items():
        si = paired[0][source] if paired else indices(m, static, config.max_gap_s)
        wi = paired[1][source] if paired else indices(m, walk, config.max_gap_s)
        n = project_rotation(np.mean(m.RIS[si], axis=0))
        d, p, v = integrate(m, wi, legacy=paired is not None)
        distance = float(np.linalg.norm(d))
        limit = 0.02 if paired else config.min_displacement_m
        if distance < limit:
            raise CalibrationError(f"{source}: horizontal displacement {distance:.4f}m < {limit}m")
        slot = slots[source]
        source_rmi = rotation_from_forward(d)
        source_rsb = (source_rmi @ n).T @ NPOSE[slot]
        per_source[source] = {"slot": slot, "RMI": source_rmi.tolist(), "RSB": source_rsb.tolist()}
        if placement_info["legacy_slot_export_available"]:
            rmi[slot], rsb[slot] = source_rmi, source_rsb
        q = {"static_sample_count": len(si), "walk_sample_count": len(wi),
             "horizontal_displacement_m": distance, "displacement_I_m": d.tolist(),
             "uncorrected_position_I_m": p.tolist(), "terminal_velocity_I_m_s": v.tolist(),
             "static_acc_rms_m_s2": float(np.sqrt(np.mean(m.aS[si] ** 2) * 3)),
             "static_gyro_rms_rad_s": float(np.sqrt(np.mean(m.gyroS[si] ** 2) * 3)),
             "static_attitude_p95_deg": float(np.percentile([angle_deg(r, n) for r in m.RIS[si]], 95))}
        if not paired:
            if q["static_acc_rms_m_s2"] > config.static_acc_rms or q["static_gyro_rms_rad_s"] > config.static_gyro_rms:
                raise CalibrationError(f"{source}: initial window is not sufficiently static", q)
            if q["static_attitude_p95_deg"] > 10:
                raise CalibrationError(f"{source}: unstable initial orientation", q)
            if post is not None:
                pi = indices(m, post, config.max_gap_s)
                q["post_acc_rms_m_s2"] = float(np.sqrt(np.mean(m.aS[pi] ** 2) * 3))
                q["post_gyro_rms_rad_s"] = float(np.sqrt(np.mean(m.gyroS[pi] ** 2) * 3))
                q["return_pose_angle_deg"] = angle_deg(project_rotation(np.mean(m.RIS[pi], axis=0)), n)
                if q["post_acc_rms_m_s2"] > config.static_acc_rms or q["post_gyro_rms_rad_s"] > config.static_gyro_rms:
                    raise CalibrationError(f"{source}: final window is not sufficiently static", q)
                if q["return_pose_angle_deg"] > config.max_return_angle_deg:
                    raise CalibrationError(f"{source}: initial/final pose differs too much", q)
            sensitivity = []
            for delta in (-0.1, 0.1):
                shifted = indices(m, (walk[0] + delta, walk[1] + delta), config.max_gap_s)
                dd, _, _ = integrate(m, shifted)
                if np.linalg.norm(dd) < limit:
                    raise CalibrationError(f"{source}: displacement unstable under 100ms window shift")
                sensitivity.append(angle_deg(rotation_from_forward(dd), source_rmi))
            q["heading_sensitivity_100ms_deg"] = max(sensitivity)
            if max(sensitivity) > config.max_heading_sensitivity_deg:
                raise CalibrationError(f"{source}: heading unstable under 100ms window shift", q)
        quality[source] = q
    windows = [{"kind": "npose_static", "start_timestamp_s": static[0], "end_timestamp_s": static[1]},
               {"kind": "forward_walk", "start_timestamp_s": walk[0], "end_timestamp_s": walk[1]}]
    # Save exact membership and offsets, unlike the old online artifacts. This
    # makes future audits possible even with different packet arrivals/rates.
    for number, bounds in enumerate((static, walk)):
        windows[number]["sample_selection"] = {}
        for s, m in motions.items():
            ix = paired[number][s] if paired else indices(m, bounds, config.max_gap_s)
            windows[number]["sample_selection"][s] = {
                "sequence_numbers": m.sequence[ix].tolist(),
                "timestamp_unix_ns": [str(t) for t in m.unix_ns[ix]],
                "clock_offset_s": m.offset_s,
                "integration_clock": "corrected_unix" if paired else (
                    "monotonic" if np.all(np.isfinite(m.monotonic_s)) else "unix_relative")}
    if post:
        windows.append({"kind": "post_static", "start_timestamp_s": post[0], "end_timestamp_s": post[1]})
    result = {"schema_version": 2, "offline_algorithm_version": 1, "method": "walking_6dof", "source_slots": slots,
              "placement": placement_info, "source_calibration": per_source,
              "session_ids": {s: m.session_id for s, m in motions.items()},
              "clock_offset_s": {s: m.offset_s for s, m in motions.items()},
              "clock_offsets_frozen": True,
              "synchronization": {"strategy": "native_independent_offline" if not paired else "historical_nearest_native_replay",
                                  "interpolation": False, "precise_cross_device_sync_established": False},
              "coordinate_convention": {"RXY": "maps Y to X", "apple_reflection": C.tolist(),
                                        "inertial_up_used": UP.tolist(), "version": "sensor.py_compatibility",
                                        "acceleration": "CoreMotion userAcceleration, gravity removed, m/s^2"},
              "windows": windows, "apply_from_timestamp_s": post[1] if post else walk[1],
              "RMI": rmi.tolist() if placement_info["legacy_slot_export_available"] else None,
              "RSB": rsb.tolist() if placement_info["legacy_slot_export_available"] else None}
    return result, quality


def detection_signals(motions, config):
    origin = max(m.time[0] for m in motions.values())
    start = origin + config.search_start_s
    end = min(origin + config.search_start_s + config.search_seconds, *(m.time[-1] for m in motions.values()))
    grid = np.arange(start, end, 0.05)
    if len(grid) < 10:
        raise CalibrationError("Insufficient common time coverage in search region")
    all_static = np.ones(len(grid), bool)
    signals = {}
    for s, m in motions.items():
        # Resampling is ONLY for the detector/plot, never for integration/output.
        acc = np.sqrt(np.maximum(0, uniform_filter1d(np.interp(grid, m.time, np.sum(m.aS ** 2, axis=1)), 5, mode="nearest")))
        gyro = np.sqrt(np.maximum(0, uniform_filter1d(np.interp(grid, m.time, np.sum(m.gyroS ** 2, axis=1)), 5, mode="nearest")))
        right = np.clip(np.searchsorted(m.time, grid), 1, len(m.time) - 1)
        valid = (m.time[right] - m.time[right - 1]) <= config.max_gap_s
        all_static &= (acc < config.static_acc_rms) & (gyro < config.static_gyro_rms) & valid
        signals[s] = {"acc": acc, "gyro": gyro}
    return origin, grid, all_static, signals


def auto_calibrate(motions, slots, config, placement_info=None):
    origin, grid, mask, _ = detection_signals(motions, config)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    runs = [(float(grid[a]), float(grid[b - 1])) for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])
            if grid[b - 1] - grid[a] >= config.static_seconds]
    accepted, rejected = [], []
    for before, after in zip(runs, runs[1:]):
        duration = after[0] - before[1]
        if not config.min_motion_seconds <= duration <= config.max_motion_seconds:
            continue
        static = (max(before[0], before[1] - 2.0), before[1])
        post = (after[0], min(after[1], after[0] + 2.0))
        walk = (before[1] - 0.2, after[0] + 0.2)
        try:
            cal, quality = calibration(motions, slots, static, walk, config, post, placement_info=placement_info)
            accepted.append((cal, quality))
        except CalibrationError as exc:
            rejected.append({"walk_relative_s": [x - origin for x in walk], "reason": str(exc), "details": exc.details})
    detection = {"origin_timestamp_s": origin, "static_runs_relative_s": [[a - origin, b - origin] for a, b in runs],
                 "accepted_count": len(accepted), "rejected_candidates": rejected}
    if len(accepted) != 1:
        detection["accepted_windows"] = [c[0]["windows"] for c in accepted]
        raise CalibrationError(f"Auto detection found {len(accepted)} valid candidates; need exactly one. "
                               "Inspect diagnostic.png, narrow --search-start/--search-seconds or supply manual windows.", detection)
    cal, quality = accepted[0]
    cal["offline_mode"] = "auto_native"
    return cal, quality, detection


def paired_indices(motions, bounds):
    native = {s: indices(m, bounds) for s, m in motions.items()}
    ref = min(motions, key=lambda s: len(native[s]))
    targets = motions[ref].time[native[ref]]
    paired, skew = {}, 0.0
    for s, m in motions.items():
        ix = native[s]
        t = m.time[ix]
        right = np.clip(np.searchsorted(t, targets), 0, len(t) - 1)
        left = np.maximum(right - 1, 0)
        picked = np.where(np.abs(t[left] - targets) <= np.abs(t[right] - targets), left, right)
        paired[s] = ix[picked]
        skew = max(skew, float(np.max(np.abs(t[picked] - targets))))
    return paired, {"reference_source": ref, "source_sample_counts": {s: len(ix) for s, ix in native.items()},
                    "paired_sample_count": len(targets), "max_pair_skew_s": skew}


def replay_reference(motions, slots, reference, config):
    by_kind = {w["kind"]: w for w in reference["windows"]}
    windows, paired, info = [], [], []
    for kind in ("npose_static", "forward_walk"):
        w = by_kind[kind]
        bounds = (w["start_timestamp_s"], w["end_timestamp_s"])
        for m in motions.values():
            if m.time[0] > bounds[0] or m.time[-1] < bounds[1]:
                raise CalibrationError(f"{m.source}: raw data does not cover historical {kind} window")
        ix, detail = paired_indices(motions, bounds)
        detail["recorded_source_sample_counts"] = w.get("source_sample_counts")
        detail["kind"] = kind
        windows.append(bounds)
        paired.append(ix)
        info.append(detail)
    cal, quality = calibration(motions, slots, *windows, config, paired=paired, placement_info=reference.get("placement"))
    cal["offline_mode"] = "reference_replay_not_auto"
    return cal, quality, {"replay_windows": info, "note": "Uses saved final clock offsets. Historical per-window offsets/received sample IDs were not saved; exact replay is not guaranteed."}


def compare(cal, reference):
    result = {}
    warnings = []
    for s, slot in cal["source_slots"].items():
        if s not in reference["source_slots"] or cal["session_ids"][s] != reference["session_ids"].get(s):
            raise CalibrationError(f"Reference does not belong to the same {s} session")
        ref_slot = reference["source_slots"][s]
        if slot != ref_slot:
            raise CalibrationError(f"Reference body slot differs for {s}")
        name = cal.get("placement", {}).get("source_placements", {}).get(s, "unspecified")
        ref_name = reference.get("placement", {}).get("source_placements", {}).get(s, "unspecified")
        if name != "unspecified" and ref_name != "unspecified" and name != ref_name:
            raise CalibrationError(f"Reference physical placement differs for {s}")
        if "unspecified" in (name, ref_name):
            warnings.append(f"{s}: physical placement cannot be verified against the reference")
        actual, expected = source_rotations(cal, s), source_rotations(reference, s)
        for key, r in zip(("RMI", "RSB"), expected):
            if r.shape != (3, 3) or not np.all(np.isfinite(r)) or np.linalg.norm(r @ r.T - np.eye(3)) > .01 or np.linalg.det(r) < .99:
                raise CalibrationError(f"Invalid reference {key} rotation for {s}")
        result[s] = {key + "_angle_deg": angle_deg(a, b) for key, a, b in zip(("RMI", "RSB"), actual, expected)}
    return {"metric": "SO(3) geodesic angular difference in degrees; reference is not ground truth",
            "sources": result, "warnings": warnings}


def apply_calibration(motions, cal, output, max_gap, clocks=None):
    artifacts = {}
    for s, m in motions.items():
        slot = cal["source_slots"][s]
        corrected = m.time if clocks is None else clocks[s].corrected_ns(m.unix_ns) / 1e9
        ix = np.flatnonzero(corrected >= cal["apply_from_timestamp_s"])
        rmi, rsb = source_rotations(cal, s)
        placement = cal.get("placement") or describe_placements(cal["source_slots"])
        aI = np.einsum("nij,nj->ni", m.RIS[ix], m.aS[ix])
        gap = np.r_[False, np.diff(m.integration_time[ix]) > max_gap] if len(ix) else np.array([], dtype=bool)
        path = Path(output) / f"{s}-calibrated.npz"
        np.savez_compressed(path, timestamp_unix_ns=m.unix_ns[ix], timestamp_monotonic_s=m.monotonic_s[ix],
                            timestamp_corrected_s=corrected[ix], sequence_number=m.sequence[ix],
                            acc_M_m_s2=(rmi @ aI.T).T, ori_MB=rmi @ m.RIS[ix] @ rsb,
                            gyro_S_rad_s=m.gyroS[ix], gyro_B_rad_s=(rsb.T @ m.gyroS[ix].T).T,
                            gap_before=gap, source=np.array(s), session_id=np.array(m.session_id), slot=np.array(slot),
                            placement=np.array(placement["source_placements"][s]),
                            target_frame=np.array(placement["source_target_frames"][s]), RMI=rmi, RSB=rsb)
        artifacts[s] = {"file": path.name, "sample_count": len(ix), "gap_count": int(np.sum(gap))}
    return artifacts


def diagnostic_plot(motions, config, path, cal=None, reference=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    origin, grid, _, signals = detection_signals(motions, config)
    fig, axes = plt.subplots(len(motions), 2, figsize=(13, 3.0 * len(motions)), squeeze=False, sharex=True)
    colors = {"npose_static": "#47ad87", "forward_walk": "#f2ad4e", "post_static": "#739bce"}
    for row, (source, sig) in enumerate(signals.items()):
        for col, (key, threshold, unit) in enumerate((("acc", config.static_acc_rms, "m/s²"), ("gyro", config.static_gyro_rms, "rad/s"))):
            ax = axes[row, col]
            ax.plot(grid - origin, sig[key], color="#304b6a", lw=1.1)
            ax.axhline(threshold, color="#84909d", linestyle=":", label="static threshold")
            for w in (cal or {}).get("windows", []):
                ax.axvspan(w["start_timestamp_s"] - origin, w["end_timestamp_s"] - origin,
                           color=colors.get(w["kind"], "gray"), alpha=.25, label=w["kind"])
            for w in (reference or {}).get("windows", []):
                for key_time in ("start_timestamp_s", "end_timestamp_s"):
                    ax.axvline(w[key_time] - origin, color="#ba5b7c", lw=.8, linestyle="--")
            ax.set_title(f"{source} · {key} RMS ({unit})", loc="left")
            ax.grid(alpha=.18)
            ax.set_xlim(grid[0] - origin, grid[-1] - origin)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("seconds from common first motion sample (approximate Unix alignment)")
    fig.suptitle("Offline calibration audit · dashed pink = historical windows (comparison only)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
