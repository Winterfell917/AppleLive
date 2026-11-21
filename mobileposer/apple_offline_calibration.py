"""CLI: local NDJSON -> common multimodal timeline + walking calibration."""
import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sensor_apple.offline import (CalibrationError, Config, apply_calibration,
                                auto_calibrate, calibration, canonical, compare,
                                diagnostic_plot, input_paths, load_motion, replay_reference)
from sensor_apple.alignment import (build_aligned, calibrated_arrays, clock_models, grid_motions,
                                    load_auxiliary, mapped_motions, save_native_archive)
from sensor_apple.placements import (PHONE_PLACEMENTS, WATCH_PLACEMENTS, describe_placements,
                                     placement_arrays, resolve_placements)


def mapping(items, cast):
    result = {}
    for item in items or []:
        key, value = item.split(":", 1)
        key = canonical(key)
        if key in result:
            raise ValueError(f"Duplicate source override: {key}")
        result[key] = cast(value)
    return result


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="One session folder or one/more NDJSON files (merged or split)")
    p.add_argument("--output", required=True, type=Path, help="New/empty output directory; originals are never overwritten")
    p.add_argument("--source-slot", action="append", help="Advanced numeric mapping, e.g. iphone:3; physical placement remains unspecified")
    p.add_argument("--watch-placement", choices=WATCH_PLACEMENTS, help="Specify together with --phone-placement")
    p.add_argument("--phone-placement", choices=PHONE_PLACEMENTS, help="Specify together with --watch-placement; keep mounting/grip fixed within a segment")
    p.add_argument("--session-id")
    p.add_argument("--clock-offset", action="append", help="SOURCE:seconds to ADD to Unix time; default 0, not precise sync")
    p.add_argument("--clock-drift-ppm", action="append", help="SOURCE:ppm clock-rate correction, anchored to report origin")
    p.add_argument("--sync-anchors", type=Path, help="JSON with manually identified common physical-event timestamp pairs")
    p.add_argument("--fps", type=float, default=50, help="Common timeline FPS, default 50")
    p.add_argument("--uwb-max-age", type=float, default=.3, help="Expire held UWB distances after this many seconds")
    p.add_argument("--anti-alias", choices=("auto", "none"), default="auto", help="Vector low-pass before downsampling")
    p.add_argument("--calibration-sampling", choices=("aligned", "native"), default="aligned")
    p.add_argument("--alignment-only", action="store_true", help="Align/export modalities without requiring a calibration action")
    p.add_argument("--reference", type=Path, help="Historical calibration.json for comparison ONLY in normal mode")
    p.add_argument("--replay-reference", action="store_true", help="Explicit non-auto historical-window replay, no calibrated NPZ output")
    p.add_argument("--search-start", type=float, default=0, help="Seconds from latest source first sample")
    p.add_argument("--search-seconds", type=float, default=45)
    p.add_argument("--static-window", nargs=2, type=float, metavar=("START", "END"), help="Manual relative seconds")
    p.add_argument("--walk-window", nargs=2, type=float, metavar=("START", "END"))
    p.add_argument("--post-window", nargs=2, type=float, metavar=("START", "END"))
    p.add_argument("--static-seconds", type=float, default=1.5)
    p.add_argument("--static-acc-rms", type=float, default=.65)
    p.add_argument("--static-gyro-rms", type=float, default=.25)
    p.add_argument("--max-gap", type=float, default=.15)
    p.add_argument("--no-apply", action="store_true", help="Skip calibrated arrays; retain alignment/calibration/diagnostics")
    args = p.parse_args(argv)
    if args.replay_reference and not args.reference:
        p.error("--replay-reference requires --reference")
    manual = (args.static_window, args.walk_window, args.post_window)
    if any(x is not None for x in manual) and not all(x is not None for x in manual):
        p.error("Manual mode requires --static-window, --walk-window, and --post-window")
    if args.replay_reference and (args.clock_offset or args.clock_drift_ppm or args.sync_anchors or args.alignment_only or args.source_slot or args.watch_placement or args.phone_placement or any(manual)):
        p.error("Replay uses the reference's sources/offsets/windows; do not override them")
    if any(not math.isfinite(x) or x <= 0 for x in (args.search_seconds, args.static_seconds, args.static_acc_rms, args.static_gyro_rms, args.max_gap, args.uwb_max_age)) or not math.isfinite(args.search_start) or args.search_start < 0:
        p.error("Durations/thresholds must be positive; search-start must be nonnegative")
    if not math.isfinite(args.fps) or not 1 <= args.fps <= 500:
        p.error("FPS must be between 1 and 500")
    if not args.alignment_only and not args.replay_reference and args.calibration_sampling == "aligned" and 1 / args.fps > args.max_gap:
        p.error("Grid period exceeds --max-gap; increase FPS or use --calibration-sampling native")
    if args.alignment_only and any(manual):
        p.error("--alignment-only does not use calibration windows")
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        p.error(f"Output is not empty: {output}; use a new directory")
    reference = json.loads(args.reference.read_text()) if args.reference else None
    try:
        if args.replay_reference:
            slots = reference["source_slots"]
            placement_info = reference.get("placement") or describe_placements(slots)
        else:
            slots, placement_info = resolve_placements(mapping(args.source_slot, int) if args.source_slot else None,
                                                       args.watch_placement, args.phone_placement)
        offsets = reference.get("clock_offset_s", {}) if args.replay_reference else mapping(args.clock_offset, float)
    except ValueError as exc:
        p.error(str(exc))
    if any(v is None or not math.isfinite(v) for v in offsets.values()) or set(offsets) - set(slots):
        p.error("Clock offsets must be finite and belong to selected sources")
    config = Config(search_start_s=args.search_start, search_seconds=args.search_seconds,
                    static_seconds=args.static_seconds, static_acc_rms=args.static_acc_rms,
                    static_gyro_rms=args.static_gyro_rms, max_gap_s=args.max_gap)
    output.mkdir(parents=True, exist_ok=True)
    print("Placements: " + json.dumps(placement_info["source_placements"], ensure_ascii=False))
    for warning in placement_info["warnings"]:
        print("Warning: " + warning, file=sys.stderr)
    report = {"status": "failed", "parameters": asdict(config), "orientation_accuracy_verified": False,
              "warnings": ["Passing signal checks cannot prove an N-pose or forward walking direction.",
                           "Current sensor.py coordinate convention is preserved; this is not an independent physical-frame validation."],
              "assumptions": ["known N-pose before/after a forward step", "unchanged device mounting and CoreMotion reference frame",
                              "Unix timestamps provide approximate cross-device alignment, not guaranteed synchronization"],
              "reference": str(args.reference.resolve()) if args.reference else None}
    report["placement"] = placement_info
    report["warnings"].extend(placement_info["warnings"])
    motions, cal, native, aligned, clocks = None, None, None, None, None
    try:
        paths = input_paths(args.inputs)
        motions, report["input"] = load_motion(paths, slots, args.session_id, offsets if args.replay_reference else None)
        if not args.replay_reference:
            native = motions
            anchors = json.loads(args.sync_anchors.read_text()) if args.sync_anchors else None
            clocks, clock_report = clock_models(native, offsets, mapping(args.clock_drift_ppm, float), anchors)
            streams, aux_info = load_auxiliary(paths, native)
            aligned, alignment_report = build_aligned(native, streams, slots, clocks, args.fps, args.max_gap,
                                                      args.uwb_max_age, args.anti_alias == "auto")
            alignment_report.update(clocks=clock_report, input_files=[str(path) for path in paths], **aux_info)
            alignment_report["placement"] = placement_info
            aligned.update(placement_arrays(placement_info, aligned["source_names"].tolist()))
            save_native_archive(output / "native_modalities.npz", native, streams, placement_info)
            np.savez_compressed(output / "aligned_sensors.npz", **aligned)
            write_json(output / "alignment.json", alignment_report)
            report["alignment"] = {"report": "alignment.json", "fps": args.fps,
                                   "sampling_for_calibration": args.calibration_sampling,
                                   "frame_count": alignment_report["frame_count"]}
        if args.alignment_only:
            motions = mapped_motions(native, clocks)
            report["status"] = "aligned_only_not_calibrated"
            print(json.dumps({"output": str(output), "status": report["status"]}, indent=2))
            return 0
        if not args.replay_reference:
            motions = grid_motions(aligned, native) if args.calibration_sampling == "aligned" else mapped_motions(native, clocks)
        if args.replay_reference:
            cal, quality, detection = replay_reference(motions, slots, reference, config)
        elif all(manual):
            origin = max(m.time[0] for m in motions.values())
            static, walk, post = [tuple(origin + v for v in w) for w in manual]
            if not (static[0] < static[1] <= walk[0] < walk[1] <= post[0] < post[1]):
                raise CalibrationError("Manual windows must be ordered, positive-duration, non-overlapping")
            cal, quality = calibration(motions, slots, static, walk, config, post, placement_info=placement_info)
            cal["offline_mode"] = "manual_native"
            detection = {"origin_timestamp_s": origin}
        else:
            cal, quality, detection = auto_calibrate(motions, slots, config, placement_info)
        if not args.replay_reference:
            cal["clock_models"] = clock_report
            # Legacy constant-offset readers must not mistake an affine model for
            # a verified constant clock correction. clock_models is authoritative.
            cal["clock_offset_s"] = {s: clocks[s].offset_s for s in slots}
            cal["offline_algorithm_version"] = 3
            cal["calibration_sampling"] = args.calibration_sampling
            cal["synchronization"].update(strategy="common_timestamp_grid" if args.calibration_sampling == "aligned" else "clock_mapped_native",
                                          fps=args.fps, interpolation=args.calibration_sampling == "aligned",
                                          clock_model_report="alignment.json")
            for w in cal["windows"]:
                for s, selection in w.get("sample_selection", {}).items():
                    corrected_stamps = selection.pop("timestamp_unix_ns")
                    selection["timestamp_corrected_unix_ns"] = corrected_stamps
                    if args.calibration_sampling == "aligned":
                        selection["grid_indices"] = selection.pop("sequence_numbers")
                        selection["integration_clock"] = "common_grid_seconds"
                        selection["original_sample_provenance"] = "aligned_sensors.npz: motion_bracket_unix_ns and motion_bracket_sequence"
                    else:
                        raw_by_seq = dict(zip(native[s].sequence.tolist(), native[s].unix_ns.tolist()))
                        selection["timestamp_unix_ns"] = [str(raw_by_seq[q]) for q in selection["sequence_numbers"]]
                    selection.pop("clock_offset_s", None)
            cal["offline_mode"] = cal["offline_mode"].replace("native", args.calibration_sampling)
        report.update(status="reference_replay_only" if args.replay_reference else "passed_signal_checks",
                      per_source=quality, detection=detection)
        if reference:
            report["comparison"] = compare(cal, reference)
            if any(v > 10 for metrics in report["comparison"]["sources"].values() for v in metrics.values()):
                report["warnings"].append("More than 10 degrees from historical calibration. Review before model evaluation; neither result is established ground truth.")
        cal["quality_status"] = report["status"]
        if not args.no_apply and not args.replay_reference:
            report["applied"] = apply_calibration(native, cal, output, config.max_gap_s, clocks)
            np.savez_compressed(output / "aligned_calibrated.npz", **calibrated_arrays(aligned, cal))
            report["aligned_calibrated"] = "aligned_calibrated.npz"
        write_json(output / "calibration.json", cal)
    except CalibrationError as exc:
        report.update(status="failed", error=str(exc), details=exc.details)
    except (OSError, ValueError, KeyError) as exc:
        report.update(status="failed", error=str(exc))
    finally:
        write_json(output / "calibration_quality.json", report)
        if motions:
            try:
                diagnostic_plot(motions, config, output / "diagnostic.png", cal, reference)
            except CalibrationError as exc:
                report["plot_error"] = str(exc)
                write_json(output / "calibration_quality.json", report)
    print(json.dumps({"output": str(output), "status": report["status"],
                      "error": report.get("error"), "comparison": report.get("comparison")}, indent=2))
    return 2 if report["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
