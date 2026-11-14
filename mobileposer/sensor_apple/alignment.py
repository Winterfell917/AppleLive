"""Timestamp-based offline alignment, separate from spatial calibration.

All arrays share a grid, but grid alignment is NOT proof of physical clock sync.
No network arrival timestamps are used. UWB is causal hold with an expiry, not
fabricated high-rate measurements. Original samples remain in a native archive.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp

from .offline import CalibrationError, Motion, canonical


# These independent raw channels retain the app's original Apple axes/units.
CHANNELS = {
    "accelerometer": ("raw_acc_apple_g", ("x_g", "y_g", "z_g")),
    "gyroscope": ("raw_gyro_apple_rad_s", ("x_rad_s", "y_rad_s", "z_rad_s")),
    "magnetometer": ("raw_mag_apple_uT", ("x_uT", "y_uT", "z_uT")),
    "uwb_ranging": ("uwb_distance_m", ("distance_m", "direction_x", "direction_y", "direction_z")),
}


@dataclass
class Stream:
    unix_ns: np.ndarray
    monotonic_s: np.ndarray
    sequence: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class Clock:
    origin_ns: int
    offset_s: float = 0.0
    drift_ppm: float = 0.0
    method: str = "unix_assumed"

    def relative(self, unix_ns):
        return (np.asarray(unix_ns, dtype=np.int64) - self.origin_ns) / 1e9 * (1 + self.drift_ppm * 1e-6) + self.offset_s

    def corrected_ns(self, unix_ns):
        return self.origin_ns + np.rint(self.relative(unix_ns) * 1e9).astype(np.int64)


def clock_models(motions, offsets=None, drifts=None, anchors=None):
    """Fit optional manually identified common-event anchors; never infer sync
    by correlating wrist and thigh motions, which need not have the same phase.
    anchors[source] = [{source_unix_ns: str, reference_unix_ns: str}, ...].
    """
    offsets, drifts, anchors = offsets or {}, drifts or {}, anchors or {}
    if not isinstance(anchors, dict):
        raise CalibrationError("Clock anchors JSON must be an object keyed by source")
    if (set(offsets) | set(drifts) | set(anchors)) - set(motions):
        raise CalibrationError("Clock settings contain an unselected source")
    if set(anchors) & (set(offsets) | set(drifts)):
        raise CalibrationError("Do not combine anchors and manual offset/drift for the same source")
    origin = min(int(m.unix_ns[0]) for m in motions.values())
    clocks, details = {}, {}
    for s in motions:
        offset, drift = offsets.get(s, 0.), drifts.get(s, 0.)
        method = "manual_affine" if s in offsets or s in drifts else "unix_assumed"
        extra = {}
        if s in anchors:
            points = anchors[s]
            if not isinstance(points, list) or not points:
                raise CalibrationError(f"{s}: anchors must be a nonempty list")
            try:
                x = np.array([(int(p["source_unix_ns"]) - origin) / 1e9 for p in points])
                y = np.array([(int(p["reference_unix_ns"]) - origin) / 1e9 for p in points])
            except (ValueError, TypeError, KeyError) as exc:
                raise CalibrationError(f"{s}: invalid clock anchor format") from exc
            if len(x) == 1:
                scale, offset = 1., float(y[0] - x[0])
            else:
                if np.ptp(x) < 1.0 or len(np.unique(x)) != len(x):
                    raise CalibrationError(f"{s}: distinct clock anchors must span at least 1 second")
                scale = float(np.dot(x - x.mean(), y - y.mean()) / np.sum((x - x.mean()) ** 2))
                offset = float(y.mean() - scale * x.mean())
            drift = (scale - 1) * 1e6
            residual = float(np.max(np.abs(y - (scale * x + offset))))
            if residual > .02:
                raise CalibrationError(f"{s}: clock anchor fit residual {residual:.3f}s exceeds 20ms")
            method = "user_anchor_affine" if len(x) > 1 else "user_anchor_offset"
            extra = {"anchors": points, "max_fit_residual_s": residual,
                     "note": "Fit residual is not physical synchronization accuracy; anchors must identify the same physical event."}
        if not np.isfinite(offset) or not np.isfinite(drift) or abs(drift) > 10000 or abs(offset) > 86400:
            raise CalibrationError(f"{s}: non-finite clock parameters, drift exceeds 10000 ppm, or offset exceeds one day")
        clocks[s] = Clock(origin, float(offset), float(drift), method)
        details[s] = {"origin_unix_ns": str(origin), "offset_s": float(offset), "drift_ppm": float(drift),
                      "method": method, **extra}
    return clocks, details


def mapped_motions(motions, clocks):
    """For native-calibration compatibility. Native source arrays are untouched."""
    return {s: replace(m, unix_ns=clocks[s].corrected_ns(m.unix_ns), offset_s=0.,
                       monotonic_s=m.monotonic_s * (1 + clocks[s].drift_ppm * 1e-6)) for s, m in motions.items()}


def load_auxiliary(paths, motions):
    rows = {(s, sensor): {} for s in motions for sensor in CHANNELS}
    duplicates = 0
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                    source, sensor = canonical(e.get("source", "")), e.get("sensor")
                    if (source, sensor) not in rows or str(e.get("sessionID")) != motions[source].session_id:
                        continue
                    ns, seq = int(e["timestampUnixNs"]), int(e["sequenceNumber"])
                    mono = float(e.get("timestampMonotonicS", np.nan))
                    values = np.array([float(e["values"].get(k, np.nan)) for k in CHANNELS[sensor][1]])
                    row = (ns, mono, seq, values)
                    if ns in rows[source, sensor]:
                        old = rows[source, sensor][ns]
                        if old[2] != seq or not np.array_equal(old[3], values, equal_nan=True):
                            raise ValueError("conflicting auxiliary samples at same source timestamp")
                        duplicates += 1
                    else:
                        rows[source, sensor][ns] = row
                except (ValueError, KeyError, TypeError) as exc:
                    raise CalibrationError(f"{path}:{number}: {exc}") from exc
    streams = {}
    for key, records in rows.items():
        if not records:
            continue
        ns, mono, seq, values = zip(*sorted(records.values(), key=lambda r: r[0]))
        stream = Stream(np.array(ns, dtype=np.int64), np.array(mono), np.array(seq, dtype=np.int64), np.array(values))
        if np.any(np.diff(stream.sequence) <= 0):
            raise CalibrationError(f"{key}: sequence reset/conflict; split the recording")
        if np.all(np.isfinite(stream.monotonic_s)) and len(ns) > 1:
            if np.any(np.diff(stream.monotonic_s) <= 0) or np.max(np.abs(np.diff(stream.unix_ns) / 1e9 - np.diff(stream.monotonic_s))) > .05:
                raise CalibrationError(f"{key}: sample-clock discontinuity; split the recording")
        streams[key] = stream
    return streams, {"duplicate_auxiliary_samples": duplicates}


def brackets(times, targets, max_gap):
    """No extrapolation, no interpolation across large holes; exact samples
    remain valid even if adjacent to a hole. Empty channels handled by caller.
    """
    right = np.clip(np.searchsorted(times, targets), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    exact = np.abs(times[right] - targets) <= 1e-9
    left = np.where(exact, right, left)
    span = times[right] - times[left]
    valid = ((targets >= times[0] - 1e-9) & (targets <= times[-1] + 1e-9) & (span <= max_gap))
    weight = np.divide(targets - times[left], span, out=np.zeros_like(targets), where=span > 0)
    return left, right, weight, valid, exact


def lowpass(times, values, fps, max_gap, enabled):
    """Offline zero-phase vector low-pass on each finite contiguous segment.
    Median native dt is used; large gaps are never filtered across.
    """
    info = {"enabled": False, "filtered_samples": 0, "short_segment_unfiltered_samples": 0}
    if len(times) < 3:
        return values, info
    hz = 1 / np.median(np.diff(times))
    if not enabled or hz <= 1.1 * fps:
        return values, info
    sos = butter(4, .4 * fps, btype="lowpass", fs=hz, output="sos")
    result = values.copy()
    good = np.all(np.isfinite(values), axis=1)
    boundaries = np.flatnonzero(np.r_[True, (np.diff(times) > max_gap) | (good[1:] != good[:-1]), True])
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if not good[a]:
            continue
        if b - a < 32:
            info["short_segment_unfiltered_samples"] += int(b - a)
            continue
        result[a:b] = sosfiltfilt(sos, values[a:b], axis=0)
        info["filtered_samples"] += int(b - a)
    info.update(enabled=True, native_median_hz=float(hz), cutoff_hz=.4 * fps,
                method="butterworth_order4_zero_phase_per_contiguous_segment")
    return result, info


def vector_resample(times, values, targets, max_gap, fps, anti_alias):
    filtered, report = lowpass(times, values, fps, max_gap, anti_alias)
    left, right, weight, valid, exact = brackets(times, targets, max_gap)
    valid &= np.all(np.isfinite(filtered[left]), axis=1) & np.all(np.isfinite(filtered[right]), axis=1)
    out = filtered[left] * (1 - weight[:, None]) + filtered[right] * weight[:, None]
    out[~valid] = np.nan
    return out, valid, (left, right, exact), report


def rotation_resample(times, matrices, targets, max_gap):
    _, _, _, valid, _ = brackets(times, targets, max_gap)
    out = np.full((len(targets), 3, 3), np.nan)
    # Relative seconds keep SO(3) interpolation away from Unix float precision.
    if np.any(valid):
        out[valid] = Slerp(times - times[0], Rotation.from_matrix(matrices))(
            np.clip(targets[valid], times[0], times[-1]) - times[0]).as_matrix()
    return out, valid


def hold_uwb(stream, clock, targets, max_age):
    times = clock.relative(stream.unix_ns)
    ix = np.searchsorted(times, targets, side="right") - 1
    picked = np.maximum(ix, 0)
    age = targets - times[picked]
    valid = (ix >= 0) & (age <= max_age + 1e-9) & np.isfinite(stream.values[picked, 0]) & (stream.values[picked, 0] >= 0)
    distance = stream.values[picked, 0].copy()
    distance[~valid] = np.nan
    direction = stream.values[picked, 1:].copy()
    direction_valid = valid & np.all(np.isfinite(direction), axis=1)
    direction[~direction_valid] = np.nan
    updated = valid & np.r_[True, np.diff(ix) != 0]
    stamp = stream.unix_ns[picked].copy()
    stamp[ix < 0] = -1
    corrected = clock.corrected_ns(stream.unix_ns)[picked]
    corrected[ix < 0] = -1
    sequence = stream.sequence[picked].copy()
    sequence[ix < 0] = -1
    age[ix < 0] = np.nan
    return dict(uwb_distance_m=distance, uwb_valid=valid, uwb_updated=updated,
                uwb_age_s=age, uwb_measurement_unix_ns=stamp, uwb_measurement_aligned_unix_ns=corrected,
                uwb_sequence_number=sequence, uwb_direction_local=direction, uwb_direction_valid=direction_valid)


def build_aligned(motions, streams, slots, clocks, fps=50., max_gap=.15, uwb_max_age=.3, anti_alias=True):
    if not np.isfinite(fps) or not 1 <= fps <= 500:
        raise CalibrationError("Aligned FPS must be between 1 and 500")
    sources = list(slots)
    origin = next(iter(clocks.values())).origin_ns
    time = {s: clocks[s].relative(m.unix_ns) for s, m in motions.items()}
    start = max(t[0] for t in time.values())
    end = min(t[-1] for t in time.values())
    if end <= start:
        raise CalibrationError("No common corrected motion coverage")
    count = int(np.floor((end - start) * fps)) + 1
    if count > 10_000_000:
        raise CalibrationError("Common grid exceeds 10 million frames; split the session")
    start_ns = origin + int(np.ceil(start * 1e9))
    grid_ns = start_ns + np.rint(np.arange(count) / fps * 1e9).astype(np.int64)
    targets = (grid_ns - origin) / 1e9
    targets, grid_ns = targets[targets <= end + 1e-9], grid_ns[targets <= end + 1e-9]
    n, ns = len(targets), len(sources)
    arrays = {"timestamp_unix_ns": grid_ns, "time_relative_s": (grid_ns - grid_ns[0]) / 1e9,
              "source_names": np.array(sources), "source_slots": np.array(list(slots.values())),
              "session_id": np.array(next(iter(motions.values())).session_id), "fps": np.array(fps),
              "user_acc_S_m_s2": np.full((n, ns, 3), np.nan), "gyro_S_rad_s": np.full((n, ns, 3), np.nan),
              "RIS": np.full((n, ns, 3, 3), np.nan), "motion_valid": np.zeros((n, ns), bool),
              "motion_exact_sample": np.zeros((n, ns), bool),
              "motion_bracket_unix_ns": np.full((n, ns, 2), -1, dtype=np.int64),
              "motion_bracket_sequence": np.full((n, ns, 2), -1, dtype=np.int64),
              "motion_bracket_span_s": np.full((n, ns), np.nan)}
    for sensor, (key, _) in CHANNELS.items():
        if sensor != "uwb_ranging":
            arrays[key] = np.full((n, ns, 3), np.nan)
            arrays[key + "_valid"] = np.zeros((n, ns), bool)
            arrays[key + "_bracket_unix_ns"] = np.full((n, ns, 2), -1, dtype=np.int64)
    for key in ("uwb_distance_m", "uwb_age_s"):
        arrays[key] = np.full((n, ns), np.nan)
    for key in ("uwb_valid", "uwb_updated", "uwb_direction_valid"):
        arrays[key] = np.zeros((n, ns), bool)
    for key in ("uwb_measurement_unix_ns", "uwb_measurement_aligned_unix_ns", "uwb_sequence_number"):
        arrays[key] = np.full((n, ns), -1, dtype=np.int64)
    arrays["uwb_direction_local"] = np.full((n, ns, 3), np.nan)
    report = {"schema_version": 1, "fps": fps, "frame_count": n,
              "start_unix_ns": str(grid_ns[0]), "end_unix_ns": str(grid_ns[-1]),
              "sources": {}, "clock_sync_accuracy_verified": False,
              "warnings": ["Common timestamps do not prove physical clock synchronization.",
                           "UWB timestamps are app event/callback timestamps, not verified time-of-flight measurement timestamps.",
                           "UWB hold duplicates are not new measurements. Rotation uses SLERP without an additional anti-alias filter."],
              "max_interpolation_gap_s": max_gap, "uwb_max_age_s": uwb_max_age}
    for col, s in enumerate(sources):
        m, t = motions[s], time[s]
        acc, va, pairs, fa = vector_resample(t, m.aS, targets, max_gap, fps, anti_alias)
        gyro, vg, _, fg = vector_resample(t, m.gyroS, targets, max_gap, fps, anti_alias)
        ris, vr = rotation_resample(t, m.RIS, targets, max_gap)
        valid = va & vg & vr
        acc[~valid], gyro[~valid], ris[~valid] = np.nan, np.nan, np.nan
        arrays["user_acc_S_m_s2"][:, col] = acc
        arrays["gyro_S_rad_s"][:, col] = gyro
        arrays["RIS"][:, col] = ris
        arrays["motion_valid"][:, col] = valid
        left, right, exact = pairs
        arrays["motion_exact_sample"][:, col] = valid & exact
        arrays["motion_bracket_unix_ns"][:, col] = np.stack([m.unix_ns[left], m.unix_ns[right]], axis=-1)
        arrays["motion_bracket_sequence"][:, col] = np.stack([m.sequence[left], m.sequence[right]], axis=-1)
        arrays["motion_bracket_span_s"][:, col] = t[right] - t[left]
        detail = {"motion_native_samples": len(t), "motion_valid_frames": int(valid.sum()),
                  "vector_filters": {"user_acceleration": fa, "fused_gyroscope": fg}, "channels": {}}
        for sensor, (key, _) in CHANNELS.items():
            stream = streams.get((s, sensor))
            if stream is None:
                detail["channels"][sensor] = {"native_samples": 0, "valid_frames": 0, "status": "missing_not_fabricated"}
                continue
            times = clocks[s].relative(stream.unix_ns)
            if sensor == "uwb_ranging":
                values = hold_uwb(stream, clocks[s], targets, uwb_max_age)
                for k, v in values.items():
                    arrays[k][:, col] = v
                detail["channels"][sensor] = {"native_samples": len(times), "valid_frames": int(values["uwb_valid"].sum()),
                                               "updates_on_grid": int(values["uwb_updated"].sum()), "method": "causal_hold_with_expiry"}
            else:
                values, valid, (left, right, _), filtering = vector_resample(times, stream.values, targets, max_gap, fps, anti_alias)
                arrays[key][:, col] = values
                arrays[key + "_valid"][:, col] = valid
                arrays[key + "_bracket_unix_ns"][:, col] = np.stack([stream.unix_ns[left], stream.unix_ns[right]], axis=-1)
                detail["channels"][sensor] = {"native_samples": len(times), "valid_frames": int(valid.sum()), "filter": filtering}
        report["sources"][s] = detail
    arrays["RSI"] = arrays["RIS"].swapaxes(-1, -2).copy()
    arrays["joint_motion_valid"] = arrays["motion_valid"].all(axis=1)
    return arrays, report


def grid_motions(arrays, original):
    """Use valid grid samples for calibration, leaving holes visible as gaps."""
    result = {}
    for col, source in enumerate(arrays["source_names"].tolist()):
        keep = arrays["motion_valid"][:, col]
        if keep.sum() < 10:
            raise CalibrationError(f"{source}: insufficient aligned motion samples")
        result[source] = Motion(source, original[source].session_id, arrays["timestamp_unix_ns"][keep],
                                arrays["time_relative_s"][keep], np.flatnonzero(keep), arrays["RIS"][keep, col],
                                arrays["user_acc_S_m_s2"][keep, col], arrays["gyro_S_rad_s"][keep, col])
    return result


def calibrated_arrays(arrays, cal):
    out = dict(arrays)
    rmi = np.array([cal["RMI"][slot] for slot in arrays["source_slots"]])
    rsb = np.array([cal["RSB"][slot] for slot in arrays["source_slots"]])
    ai = np.einsum("nsij,nsj->nsi", arrays["RIS"], arrays["user_acc_S_m_s2"])
    out.update(RMI=rmi, RSB=rsb, acc_M_m_s2=np.einsum("sij,nsj->nsi", rmi, ai),
               ori_MB=rmi[None] @ arrays["RIS"] @ rsb[None],
               gyro_B_rad_s=np.einsum("sji,nsj->nsi", rsb, arrays["gyro_S_rad_s"]),
               after_calibration=arrays["timestamp_unix_ns"] / 1e9 >= cal["apply_from_timestamp_s"],
               calibration_available=np.array(True))
    return out


def save_native_archive(path, motions, streams):
    out = {"session_id": np.array(next(iter(motions.values())).session_id), "source_names": np.array(list(motions))}
    for s, m in motions.items():
        prefix = s + "__motion__"
        for name, value in {"timestamp_unix_ns": m.unix_ns, "timestamp_monotonic_s": m.monotonic_s,
                            "sequence_number": m.sequence, "RIS": m.RIS, "user_acc_S_m_s2": m.aS,
                            "gyro_S_rad_s": m.gyroS}.items():
            out[prefix + name] = value
    for (s, sensor), stream in streams.items():
        prefix = s + "__" + sensor + "__"
        out.update({prefix + "timestamp_unix_ns": stream.unix_ns, prefix + "timestamp_monotonic_s": stream.monotonic_s,
                    prefix + "sequence_number": stream.sequence, prefix + "values": stream.values,
                    prefix + "value_names": np.array(CHANNELS[sensor][1])})
    np.savez_compressed(path, **out)
