"""Pull and prepare one curated Apple Sensor Read sequence.

The phone may contain a long NDJSON recording shared by several mocap trials.
This tool uses a livedemo sequence manifest to select the matching session and
time range, then creates native-rate raw data plus nearest-time alignment maps.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


DEFAULT_DEVICE = "iPhone"
DEFAULT_BUNDLE_ID = "com.sensorread.ios"


def json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_source(source: str) -> str:
    source = source.removesuffix("_simulator").lower()
    return {"watch": "apple_watch", "applewatch": "apple_watch"}.get(source, source)


def all_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from all_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from all_strings(item)


def run_devicectl(arguments: List[str], json_output: Optional[Path] = None) -> Optional[dict]:
    command = ["xcrun", "devicectl", *arguments]
    if json_output is not None:
        command.extend(["--json-output", str(json_output)])
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        if "DeviceLocked" in detail or "device is locked" in detail:
            raise RuntimeError("iPhone is locked; unlock it and keep it connected to this Mac")
        raise RuntimeError(f"devicectl failed ({result.returncode}): {detail}")
    return load_json(json_output) if json_output is not None else None


def remote_recording_for_manifest(
    manifest: dict,
    device: str,
    bundle_id: str,
    temporary_directory: Path,
) -> str:
    listing_path = temporary_directory / "files.json"
    listing = run_devicectl(
        [
            "device", "info", "files",
            "--device", device,
            "--domain-type", "appDataContainer",
            "--domain-identifier", bundle_id,
            "--subdirectory", "Documents/Recordings",
        ],
        json_output=listing_path,
    )
    session_prefixes = [str(value)[:8].lower() for value in manifest.get("session_ids", [])]
    if not session_prefixes:
        raise RuntimeError("manifest does not contain a Sensor Read sessionID")

    candidates = set()
    for value in all_strings(listing):
        if value.lower().endswith(".ndjson"):
            filename = Path(value).name
            if any(prefix in filename.lower() for prefix in session_prefixes):
                candidates.add(filename)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one device recording for session prefix {session_prefixes}, "
            f"found {sorted(candidates)}"
        )
    return candidates.pop()


def pull_recording(sequence_dir: Path, device: str, bundle_id: str, temporary_directory: Path) -> Path:
    manifest = load_json(sequence_dir / "manifest.json")
    filename = remote_recording_for_manifest(manifest, device, bundle_id, temporary_directory)
    download_dir = temporary_directory / "download"
    download_dir.mkdir()
    destination = download_dir / filename
    run_devicectl(
        [
            "device", "copy", "from",
            "--device", device,
            "--domain-type", "appDataContainer",
            "--domain-identifier", bundle_id,
            "--source", f"Documents/Recordings/{filename}",
            "--destination", str(destination),
        ]
    )
    if not destination.is_file():
        raise RuntimeError(f"device copy completed but did not create {destination}")
    return destination


def import_livedemo_record(
    sequence_dir: Path,
    record_path: Path,
    sequence_name: str,
    subject: Optional[str],
    action: Optional[str],
    trial: Optional[str],
    notes: Optional[str],
    model: Path,
) -> None:
    """Bootstrap a curated package from an ordinary livedemo recording."""
    if sequence_dir.exists() and any(sequence_dir.iterdir()):
        raise RuntimeError(f"sequence directory is not empty: {sequence_dir}")
    payload = torch.load(record_path, map_location="cpu")
    required = ("timestamp", "acc", "ori", "gyro", "pose", "valid", "calibration")
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"livedemo record is missing fields: {missing}")
    timestamps = payload["timestamp"].to(torch.float64)
    valid_timestamps = timestamps[timestamps > 0]
    if not len(valid_timestamps):
        raise RuntimeError("livedemo record has no positive sensor timestamp")
    start_s = float(valid_timestamps.min())
    end_s = float(valid_timestamps.max())
    calibration = payload["calibration"]
    session_ids = sorted(set(str(value) for value in calibration.get("session_ids", {}).values()))
    if not session_ids:
        raise RuntimeError("livedemo calibration has no Sensor Read sessionID")

    sequence_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(record_path, sequence_dir / "mocap.pt")
    write_json(sequence_dir / "calibration.json", json_value(calibration))
    frame_count = int(len(payload["pose"]))
    duration_s = end_s - start_s
    manifest = {
        "schema_version": 1,
        "sequence_name": sequence_name,
        "subject": subject,
        "action": action,
        "trial": trial,
        "notes": notes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_ids": session_ids,
        "source_session_ids": calibration.get("session_ids", {}),
        "sequence_start_timestamp_s": start_s,
        "sequence_end_timestamp_s": end_s,
        "save_trigger": "imported_ordinary_livedemo_record",
        "duration_s": duration_s,
        "mocap_start_timestamp_s": start_s,
        "mocap_end_timestamp_s": end_s,
        "frame_count": frame_count,
        "target_fps": 30.0,
        "measured_fps": frame_count / duration_s if duration_s > 0 else None,
        "calibration_method": calibration.get("method"),
        "calibrator": "none",
        "model": str(model.resolve()),
        "source_slots": payload.get("source_slots", calibration.get("source_slots", {})),
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
        "imported_from": str(record_path.resolve()),
        "boundary_note": "Action bounds reconstructed from first/last valid livedemo IMU timestamps.",
    }
    write_json(sequence_dir / "manifest.json", json_value(manifest))
    print(f"Imported livedemo record: {record_path}")
    print(f"Created sequence package: {sequence_dir}")
    print(f"Frames: {frame_count}; reconstructed duration: {duration_s:.3f}s")


def build_subject_manifest(subject_dir: Path, subject_id: str, collection_date: str) -> None:
    sequences = []
    for sequence_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
        manifest_path = sequence_dir / "manifest.json"
        quality_path = sequence_dir / "postprocess_quality.json"
        if not manifest_path.is_file() or not quality_path.is_file():
            continue
        manifest = load_json(manifest_path)
        quality = load_json(quality_path)
        source_quality = {
            source: {
                "estimated_rate_hz": report.get("estimated_rate_hz"),
                "max_native_gap_s": report.get("max_native_gap_s"),
                "max_resampling_bracket_s": report.get("max_resampling_bracket_s"),
                "invalid_output_frame_count": report.get("invalid_output_frame_count", 0),
                "invalid_output_fraction": report.get("invalid_output_fraction", 0.0),
            }
            for source, report in quality["output_grid"]["sources"].items()
        }
        max_invalid_fraction = max(
            report["invalid_output_fraction"] for report in source_quality.values()
        )
        quality_status = (
            "warning" if max_invalid_fraction > 0.05
            else "minor_gaps" if max_invalid_fraction > 0.0
            else "clean"
        )
        postprocess = manifest["postprocess"]
        sequences.append({
            "sequence_name": manifest["sequence_name"],
            "trial": manifest.get("trial"),
            "action": manifest.get("action"),
            "session_ids": manifest.get("session_ids", []),
            "start_timestamp_s": postprocess.get(
                "output_start_s", quality["output_grid"]["output_start_s"]
            ),
            "end_timestamp_s": postprocess.get(
                "output_end_s", quality["output_grid"]["output_end_s"]
            ),
            "duration_s": (
                quality["output_grid"]["output_end_s"]
                - quality["output_grid"]["output_start_s"]
            ),
            "fps": postprocess["fps"],
            "frame_count": postprocess["frame_count"],
            "sync_mode": postprocess["sync_mode"],
            "jump_residual_shift_s": {
                source: report["residual_shift_s"]
                for source, report in quality["synchronization"]["sources"].items()
            },
            "source_quality": source_quality,
            "quality_status": quality_status,
            "recommended_for_training": max_invalid_fraction <= 0.05,
            "artifacts": manifest["artifacts"],
        })
    if not sequences:
        raise RuntimeError(f"no processed sequences found in {subject_dir}")
    summary = {
        "schema_version": 1,
        "subject_id": subject_id,
        "collection_date": collection_date,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence_count": len(sequences),
        "total_frame_count": sum(item["frame_count"] for item in sequences),
        "total_duration_s": sum(item["duration_s"] for item in sequences),
        "sequences": sequences,
    }
    write_json(subject_dir / "subject_manifest.json", summary)
    print(f"Indexed subject {subject_id}: {len(sequences)} sequences")
    print(f"Subject manifest: {subject_dir / 'subject_manifest.json'}")


def mocap_target_times(mocap_path: Path) -> torch.Tensor:
    payload = torch.load(mocap_path, map_location="cpu")
    timestamps = payload["timestamp"].to(torch.float64)
    if timestamps.ndim == 1:
        return timestamps
    result = []
    for row in timestamps:
        valid = row[row > 0]
        if not len(valid):
            raise RuntimeError("mocap.pt contains a frame without a valid timestamp")
        result.append(valid[0])
    return torch.stack(result)


def parse_selected_events(
    raw_path: Path,
    session_ids: set,
    offsets: Dict[str, float],
    start_s: float,
    end_s: float,
) -> Tuple[List[Tuple[dict, float]], int, int]:
    selected = []
    total_lines = invalid_lines = 0
    with raw_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            total_lines += 1
            try:
                event = json.loads(line)
                source_name = canonical_source(str(event["source"]))
                timestamp_s = int(event["timestampUnixNs"]) / 1_000_000_000.0
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                invalid_lines += 1
                continue
            if session_ids and str(event.get("sessionID", "")) not in session_ids:
                continue
            corrected_s = timestamp_s + float(offsets.get(source_name, 0.0) or 0.0)
            if start_s <= corrected_s <= end_s:
                selected.append((event, corrected_s))
    return selected, total_lines, invalid_lines


def nearest_alignment(sample_times: torch.Tensor, target_times: torch.Tensor):
    if not len(sample_times):
        return (
            torch.full((len(target_times),), -1, dtype=torch.long),
            torch.full((len(target_times),), math.inf, dtype=torch.float64),
        )
    right = torch.searchsorted(sample_times, target_times).clamp(max=len(sample_times) - 1)
    left = (right - 1).clamp(min=0)
    right_distance = (sample_times[right] - target_times).abs()
    left_distance = (sample_times[left] - target_times).abs()
    choose_left = left_distance <= right_distance
    indices = torch.where(choose_left, left, right)
    return indices, (sample_times[indices] - target_times).abs()


def build_streams(events: List[Tuple[dict, float]], target_times: torch.Tensor):
    grouped = defaultdict(list)
    for event, corrected_s in events:
        key = f"{canonical_source(str(event.get('source', 'unknown')))}/{event.get('sensor', 'unknown')}"
        grouped[key].append((event, corrected_s))

    streams = {}
    reports = {}
    for key, items in sorted(grouped.items()):
        items.sort(key=lambda item: item[1])
        native_times = torch.tensor([item[1] for item in items], dtype=torch.float64)
        source_times = torch.tensor(
            [int(item[0]["timestampUnixNs"]) / 1_000_000_000.0 for item in items],
            dtype=torch.float64,
        )
        sequences = torch.tensor([int(item[0].get("sequenceNumber", -1)) for item in items])
        field_names = sorted(
            set().union(*(item[0].get("values", {}).keys() for item in items))
        )
        fields = {
            field: torch.tensor(
                [float(item[0].get("values", {}).get(field, math.nan)) for item in items],
                dtype=torch.float64,
            )
            for field in field_names
        }
        nearest_index, nearest_dt = nearest_alignment(native_times, target_times)
        time_gaps = torch.diff(native_times)
        sequence_gaps = torch.diff(sequences)
        positive_span = max(float(native_times[-1] - native_times[0]), 0.0)
        reports[key] = {
            "sample_count": len(items),
            "estimated_rate_hz": (len(items) - 1) / positive_span if positive_span > 0 else None,
            "max_timestamp_gap_s": float(time_gaps.max()) if len(time_gaps) else None,
            "sequence_gap_count": int((sequence_gaps > 1).sum()) if len(sequence_gaps) else 0,
            "missing_sequence_count": int((sequence_gaps - 1).clamp_min(0).sum()) if len(sequence_gaps) else 0,
            "max_nearest_mocap_offset_s": float(nearest_dt.max()),
            "median_nearest_mocap_offset_s": float(nearest_dt.median()),
            "fields": field_names,
        }
        streams[key] = {
            "timestamp_s": source_times,
            "corrected_timestamp_s": native_times,
            "sequence_number": sequences,
            "values": fields,
            "nearest_mocap_index": nearest_index,
            "nearest_mocap_offset_s": nearest_dt,
        }
    return streams, reports


def prepare_sequence(sequence_dir: Path, raw_path: Path, padding_s: float) -> None:
    manifest_path = sequence_dir / "manifest.json"
    calibration_path = sequence_dir / "calibration.json"
    manifest = load_json(manifest_path)
    calibration = load_json(calibration_path)
    session_ids = set(str(value) for value in manifest.get("session_ids", []))
    start_s = float(manifest["sequence_start_timestamp_s"]) - padding_s
    end_s = float(manifest["sequence_end_timestamp_s"]) + padding_s
    offsets = {
        canonical_source(source): float(value or 0.0)
        for source, value in calibration.get("clock_offset_s", {}).items()
    }
    events, total_lines, invalid_lines = parse_selected_events(
        raw_path, session_ids, offsets, start_s, end_s
    )
    if not events:
        raise RuntimeError("no raw events matched the manifest sessionID and sequence time range")

    raw_output = sequence_dir / "raw.ndjson"
    with raw_output.open("w", encoding="utf-8") as output:
        for event, _ in events:
            output.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    target_times = mocap_target_times(sequence_dir / "mocap.pt")
    streams, stream_reports = build_streams(events, target_times)
    torch.save(
        {
            "schema_version": 1,
            "target_timestamp_s": target_times,
            "streams": streams,
        },
        sequence_dir / "alignment.pt",
    )
    report = {
        "schema_version": 1,
        "source_file": raw_path.name,
        "source_total_lines": total_lines,
        "source_invalid_lines": invalid_lines,
        "selected_event_count": len(events),
        "selection_start_timestamp_s": start_s,
        "selection_end_timestamp_s": end_s,
        "padding_s": padding_s,
        "streams": stream_reports,
    }
    write_json(sequence_dir / "quality.json", report)

    manifest["artifacts"].update(
        {
            "raw_sequence": "raw.ndjson",
            "alignment": "alignment.pt",
            "quality_report": "quality.json",
        }
    )
    manifest["raw_extraction"] = {
        "source_file": raw_path.name,
        "selected_event_count": len(events),
        "padding_s": padding_s,
    }
    write_json(manifest_path, manifest)
    print(f"Prepared sequence: {sequence_dir}")
    print(f"Selected {len(events)} of {total_lines} raw events across {len(streams)} streams")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one Apple mocap dataset sequence")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="use an already exported device NDJSON")
    prepare.add_argument("--sequence-dir", type=Path, required=True)
    prepare.add_argument("--raw-file", type=Path, required=True)
    prepare.add_argument("--padding", type=float, default=0.25)

    pull = subparsers.add_parser("pull", help="find and pull the matching iPhone recording")
    pull.add_argument("--sequence-dir", type=Path, required=True)
    pull.add_argument("--device", default=DEFAULT_DEVICE)
    pull.add_argument("--bundle-id", default=DEFAULT_BUNDLE_ID)
    pull.add_argument("--padding", type=float, default=0.25)

    import_record = subparsers.add_parser(
        "import-record", help="bootstrap a sequence package from an ordinary livedemo .pt"
    )
    import_record.add_argument("--sequence-dir", type=Path, required=True)
    import_record.add_argument("--record-file", type=Path, required=True)
    import_record.add_argument("--sequence-name", required=True)
    import_record.add_argument("--subject")
    import_record.add_argument("--action")
    import_record.add_argument("--trial")
    import_record.add_argument("--notes")
    import_record.add_argument(
        "--model", type=Path,
        default=Path(__file__).resolve().parent / "data/checkpoints/base_model_12combo.pth",
    )

    index_subject = subparsers.add_parser(
        "index-subject", help="summarize processed child sequences for one subject"
    )
    index_subject.add_argument("--subject-dir", type=Path, required=True)
    index_subject.add_argument("--subject-id", required=True)
    index_subject.add_argument("--date", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "index-subject":
        build_subject_manifest(args.subject_dir.resolve(), args.subject_id, args.date)
        return
    sequence_dir = args.sequence_dir.resolve()
    if args.command == "import-record":
        import_livedemo_record(
            sequence_dir=sequence_dir,
            record_path=args.record_file.resolve(),
            sequence_name=args.sequence_name,
            subject=args.subject,
            action=args.action,
            trial=args.trial,
            notes=args.notes,
            model=args.model,
        )
        return

    if args.padding < 0:
        raise SystemExit("--padding must be non-negative")
    for required in ("manifest.json", "calibration.json", "mocap.pt"):
        if not (sequence_dir / required).is_file():
            raise SystemExit(f"missing sequence artifact: {sequence_dir / required}")

    if args.command == "prepare":
        prepare_sequence(sequence_dir, args.raw_file.resolve(), args.padding)
        return

    with tempfile.TemporaryDirectory(prefix="apple-sequence-") as temporary:
        raw_path = pull_recording(sequence_dir, args.device, args.bundle_id, Path(temporary))
        prepare_sequence(sequence_dir, raw_path, args.padding)


if __name__ == "__main__":
    main()
