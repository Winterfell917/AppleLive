"""Read-only completeness check for device-local iPhone/Watch recordings.

By default, inspect the p02 evaluation collection. No labels are generated or
raw files modified. Use --dataset train to inspect a subject's training set.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

import collect_train_sequence as collect


BASE_DIR = Path(__file__).resolve().parent / "data" / "datasets" / "subjects"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=collect.subject_id, default="p02")
    parser.add_argument("--dataset", choices=("test", "train"), default="test")
    parser.add_argument("--labels-root", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument(
        "--require-all-planned",
        action="store_true",
        help="将计划中尚未采集的序列也视为错误",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="任何逐帧缺失或未配对原始文件也使退出码为 1",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=None,
        help="可选：另存完整 JSON 检查报告；默认只打印、不写文件",
    )
    return parser.parse_args()


def issue(entry: dict, severity: str, message: str) -> None:
    entry["issues"].append({"severity": severity, "message": message})


def invalid_ranges(valid, fps: float) -> list[dict]:
    """Return inclusive zero-based frame intervals where a mask is false."""
    missing = (~valid).nonzero().flatten().tolist()
    if not missing:
        return []
    ranges = []
    first = previous = missing[0]
    for index in missing[1:]:
        if index != previous + 1:
            ranges.append(
                {
                    "first_frame": first,
                    "last_frame": previous,
                    "start_s": round(first / fps, 3),
                    "end_s": round((previous + 1) / fps, 3),
                }
            )
            first = index
        previous = index
    ranges.append(
        {
            "first_frame": first,
            "last_frame": previous,
            "start_s": round(first / fps, 3),
            "end_s": round((previous + 1) / fps, 3),
        }
    )
    return ranges


def check_capture(capture_path: Path, labels_root: Path, session_index: dict) -> dict:
    relative = capture_path.parent.relative_to(labels_root).as_posix()
    entry = {
        "sequence": relative,
        "status": "error",
        "session_id": None,
        "raw_files": {},
        "issues": [],
    }
    try:
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        session_ids = collect.capture_session_ids(capture)
        if len(session_ids) != 1:
            issue(entry, "error", f"标定清单需要唯一 sessionID，实际为 {session_ids}")
            return entry
        session_id = session_ids[0]
        entry["session_id"] = session_id

        for source in collect.DEVICES:
            candidates = session_index.get((session_id, source), [])
            if len(candidates) != 1:
                issue(entry, "error", f"{source} 原始文件匹配数为 {len(candidates)}，应为 1")
            else:
                entry["raw_files"][source] = str(candidates[0].resolve())
        if len(entry["raw_files"]) != len(collect.DEVICES):
            return entry

        trace_name = capture.get("artifacts", {}).get("live_calibration")
        if not trace_name:
            issue(entry, "error", "采集清单未记录在线 iPhone 标定样本文件")
            return entry
        trace_path = capture_path.parent / str(trace_name)
        trace = collect.load_live_calibration_samples(trace_path)
        offsets, offset_report = collect.resolve_time_offsets(None, None, trace)
        if offset_report["iphone"]["method"] == "zero_fallback_no_live_trace":
            issue(entry, "error", "无法从在线 iPhone 样本估计设备与电脑的时间偏移")
            return entry
        entry["time_offsets_s"] = offsets

        events, file_report = collect.merge_device_events(
            Path(entry["raw_files"]["iphone"]),
            Path(entry["raw_files"]["apple_watch"]),
        )
        motion, uwb, stream_report = collect.extract_offline_streams(events, offsets)
        entry["raw_event_count"] = len(events)
        entry["rejected_motion_event_count"] = stream_report["rejected_motion_event_count"]
        entry["duplicate_event_count"] = file_report["duplicate_event_count"]
        if file_report["duplicate_event_count"]:
            issue(entry, "warning", f"合并时发现 {file_report['duplicate_event_count']} 个重复事件")
        for source in collect.DEVICES:
            ids = stream_report["session_ids"][source]
            if ids != [session_id]:
                issue(entry, "error", f"{source} 原始文件含有意外 sessionID：{ids}")
            rejected = stream_report["rejected_motion_event_count"][source]
            if rejected:
                issue(entry, "warning", f"{source} 有 {rejected} 个无法解析的 device_motion 事件")

        action_start, action_end = collect.window_bounds(capture, "action")
        fps = float(capture.get("target_fps", 30.0))
        if fps <= 0 or action_end <= action_start:
            issue(entry, "error", "正式动作时间窗或帧率无效")
            return entry
        for phase in ("npose", "forward_walk"):
            phase_start, phase_end = collect.window_bounds(capture, phase)
            count = len(collect.samples_between(motion["iphone"], phase_start, phase_end))
            entry.setdefault("calibration_iphone_motion_count", {})[phase] = count
            if not count:
                issue(entry, "error", f"iPhone {phase} 标定窗口没有 device_motion")

        action_uwb = collect.samples_between(uwb, action_start, action_end)
        grid = collect.build_offline_grid(motion, action_uwb, action_start, action_end, fps)
        frame_count = int(grid["motion_valid"].shape[0])
        entry["action_frame_count"] = frame_count
        entry["fps"] = fps
        entry["devices"] = {}
        for device_index, source in enumerate(collect.DEVICES):
            motion_count = len(collect.samples_between(motion[source], action_start, action_end))
            source_uwb = [sample for sample in action_uwb if sample["source"] == source]
            uwb_count = len(source_uwb)
            motion_valid = grid["motion_valid"][:, device_index]
            uwb_distance = grid["uwb_distance_m"][:, device_index]
            invalid_value_mask = grid["uwb_valid"][:, device_index] & (
                ~torch.isfinite(uwb_distance) | (uwb_distance <= 0)
            )
            uwb_valid = grid["uwb_valid"][:, device_index] & ~invalid_value_mask
            invalid_distance_samples = [
                {
                    "sequence_number": sample["sequence"],
                    "action_time_s": round(sample["timestamp_s"] - action_start, 3),
                    "distance_m": (
                        sample["distance_m"]
                        if math.isfinite(sample["distance_m"])
                        else None
                    ),
                    "distance_text": str(sample["distance_m"]),
                }
                for sample in source_uwb
                if not math.isfinite(sample["distance_m"])
                or sample["distance_m"] <= 0
            ]
            device = {
                "action_motion_samples": motion_count,
                "action_uwb_samples": uwb_count,
                "motion_valid_frames": int(motion_valid.sum()),
                "uwb_valid_frames": int(uwb_valid.sum()),
                "motion_missing_ranges": invalid_ranges(motion_valid, fps),
                "uwb_missing_ranges": invalid_ranges(uwb_valid, fps),
                "invalid_uwb_distance_samples": invalid_distance_samples,
                "invalid_uwb_value_ranges": invalid_ranges(~invalid_value_mask, fps),
            }
            entry["devices"][source] = device
            if not motion_count or not device["motion_valid_frames"]:
                issue(entry, "error", f"{source} 正式动作窗口没有有效 IMU")
            elif device["motion_missing_ranges"]:
                issue(
                    entry,
                    "warning",
                    f"{source} IMU 缺 {frame_count - device['motion_valid_frames']} 帧",
                )
            if not uwb_count or not device["uwb_valid_frames"]:
                issue(entry, "error", f"{source} 正式动作窗口没有有效 UWB 测距")
            elif device["uwb_missing_ranges"]:
                issue(
                    entry,
                    "warning",
                    f"{source} UWB 缺 {frame_count - device['uwb_valid_frames']} 帧",
                )
            if invalid_distance_samples:
                issue(
                    entry,
                    "error",
                    f"{source} 有 {len(invalid_distance_samples)} 个非正数或非有限的 UWB 原始距离",
                )
        entry["status"] = (
            "error"
            if any(item["severity"] == "error" for item in entry["issues"])
            else "warning" if entry["issues"] else "ok"
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        issue(entry, "error", f"读取或处理失败：{error}")
        entry["status"] = "error"
    return entry


def format_ranges(ranges: list[dict]) -> str:
    return ", ".join(
        f"{item['first_frame']}-{item['last_frame']}帧"
        f"({item['start_s']:.3f}-{item['end_s']:.3f}s)"
        for item in ranges
    )


def main() -> int:
    args = parse_args()
    base = BASE_DIR / args.subject
    labels_root = (args.labels_root or base / f"{args.dataset}_labels").resolve()
    raw_root = (args.raw_root or base / f"{args.dataset}_raw").resolve()
    if not labels_root.is_dir():
        raise SystemExit(f"采集清单目录不存在：{labels_root}")
    if not raw_root.is_dir():
        raise SystemExit(f"原始文件目录不存在：{raw_root}")

    plan_path = labels_root / "plan.json"
    planned = None
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        planned = {
            collect.sequence_directory(labels_root, take).relative_to(labels_root).as_posix()
            for take in plan
        }
    captures = {
        path.parent.relative_to(labels_root).as_posix(): path
        for path in labels_root.rglob(collect.CAPTURE_FILENAME)
        if "_trial" not in path.parent.relative_to(labels_root).parts
    }
    pending = sorted(planned - captures.keys()) if planned is not None else []
    outside_plan = sorted(captures.keys() - planned) if planned is not None else []

    session_index, raw_reports = collect.index_raw_session_files(raw_root)
    results = [check_capture(path, labels_root, session_index) for _, path in sorted(captures.items())]
    used_files = {
        path
        for result in results
        for path in result["raw_files"].values()
    }
    unused_files = [
        str(item["path"].resolve())
        for item in raw_reports
        if str(item["path"].resolve()) not in used_files
    ]
    invalid_raw_files = [
        {"path": str(item["path"].resolve()), "invalid_lines": item["invalid_line_count"]}
        for item in raw_reports
        if item["invalid_line_count"]
    ]
    counts = {
        status: sum(item["status"] == status for item in results)
        for status in ("ok", "warning", "error")
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "subject": args.subject,
        "dataset": args.dataset,
        "labels_root": str(labels_root),
        "raw_root": str(raw_root),
        "planned_sequences": len(planned) if planned is not None else None,
        "captured_sequences": len(captures),
        "pending_sequences": pending,
        "captures_outside_plan": outside_plan,
        "raw_event_files": len(raw_reports),
        "unused_raw_event_files": unused_files,
        "raw_files_with_invalid_lines": invalid_raw_files,
        "counts": counts,
        "sequences": results,
    }

    print(f"受试者 {args.subject}，{args.dataset} 数据")
    print(f"采集清单：{labels_root}")
    print(f"原始文件：{raw_root}")
    plan_count = len(planned) if planned is not None else "未知（无 plan.json）"
    print(f"计划 {plan_count} 条；已录 {len(captures)} 条；待录 {len(pending)} 条。")
    print(f"已录序列：完整 {counts['ok']}，局部缺帧 {counts['warning']}，错误 {counts['error']}。")
    print(f"原始事件文件 {len(raw_reports)} 个；未匹配文件 {len(unused_files)} 个；损坏行文件 {len(invalid_raw_files)} 个。")
    for result in results:
        print(f"[{result['status'].upper()}] {result['sequence']}")
        for item in result["issues"]:
            print(f"  {item['severity']}: {item['message']}")
        for source, device in result.get("devices", {}).items():
            for signal in ("motion", "uwb"):
                ranges = device[f"{signal}_missing_ranges"]
                if ranges:
                    print(f"  {source} {signal} 缺失位置：{format_ranges(ranges)}")
            if device["invalid_uwb_distance_samples"]:
                print(f"  {source} UWB 异常样本：{device['invalid_uwb_distance_samples']}")
    if unused_files:
        print("未匹配的原始事件文件：")
        for path in unused_files:
            print(f"  {path}")
    if invalid_raw_files:
        print("含损坏行的原始文件：")
        for item in invalid_raw_files:
            print(f"  {item['path']}：{item['invalid_lines']} 行")
    if outside_plan:
        print("采集清单中存在计划外序列：", ", ".join(outside_plan))
    if args.require_all_planned and pending:
        print("尚未采集的计划条目：")
        for sequence in pending:
            print(f"  {sequence}")
    if args.json_report is not None:
        path = args.json_report.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"JSON 报告：{path}")

    failed = bool(counts["error"] or invalid_raw_files or outside_plan)
    if args.require_all_planned and pending:
        failed = True
    if args.strict and (counts["warning"] or unused_files):
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
