"""Audit finalized p02 test labels and build a portable, clean export.

The source dataset is never modified.  Accepted sequences are written with only
``labels.pt``, ``meta.json`` and ``quality.json``.  Invalid UWB distances are
masked in the exported copy, and sequences whose UWB validity falls below the
configured threshold are excluded and listed in ``audit_report.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

import collect_train_sequence as collect


BASE_DIR = Path(__file__).resolve().parent / "data" / "datasets" / "subjects"
REQUIRED_FILES = ("labels.pt", "meta.json", "quality.json")
REQUIRED_TENSORS = {
    "orientation",
    "acceleration",
    "motion_valid",
    "grid_time_s",
    "uwb_distance_m",
    "uwb_timestamp_s",
    "uwb_valid",
    "uwb_native_timestamp_s",
    "uwb_native_distance_m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=collect.subject_id, default="p02")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--min-uwb-valid-ratio",
        type=float,
        default=0.95,
        help="每台设备逐帧 UWB 的最低有效比例，默认 0.95",
    )
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def expected_identity(relative: Path) -> dict:
    action, wear, take_name = relative.parts
    watch, pocket = wear.split("_", 1)
    group_text, facing = take_name.split("_", 1)
    return {
        "action": action,
        "watch": watch,
        "phone_pocket": pocket,
        "group": int(group_text.removeprefix("g")),
        "phone_facing": facing,
    }


def longest_false_run(mask: torch.Tensor) -> int:
    longest = current = 0
    for valid in mask.tolist():
        if valid:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def audit_sequence(
    directory: Path,
    root: Path,
    subject: str,
    min_uwb_valid_ratio: float,
) -> tuple[dict, dict | None, dict | None, dict | None]:
    relative = directory.relative_to(root)
    result = {
        "sequence": relative.as_posix(),
        "status": "excluded",
        "errors": [],
        "warnings": [],
    }
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        result["errors"].append(f"缺少文件：{', '.join(missing)}")
        return result, None, None, None

    try:
        labels = torch.load(directory / "labels.pt", map_location="cpu", weights_only=False)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        quality = json.loads((directory / "quality.json").read_text(encoding="utf-8"))
    except Exception as error:  # keep auditing other sequences
        result["errors"].append(f"文件无法读取：{error}")
        return result, None, None, None

    if not isinstance(labels, dict):
        result["errors"].append("labels.pt 顶层不是字典")
        return result, None, None, None
    missing_fields = sorted(REQUIRED_TENSORS - labels.keys())
    if missing_fields:
        result["errors"].append(f"labels.pt 缺少字段：{missing_fields}")
        return result, None, None, None

    identity = expected_identity(relative)
    for key, expected in identity.items():
        if labels.get(key) != expected or meta.get(key) != expected:
            result["errors"].append(
                f"目录与 {key} 不一致：目录={expected!r}, "
                f"labels={labels.get(key)!r}, meta={meta.get(key)!r}"
            )
    if labels.get("subject") != subject or meta.get("subject") != subject:
        result["errors"].append(f"受试者字段不是 {subject}")
    if labels.get("devices") != list(collect.DEVICES):
        result["errors"].append(f"设备顺序错误：{labels.get('devices')!r}")
    expected_slots = [
        collect.WATCH_INFO[identity["watch"]]["slot"],
        collect.POCKET_INFO[identity["phone_pocket"]]["slot"],
    ]
    expected_model_slots = [
        collect.WATCH_INFO[identity["watch"]]["model_slot"],
        collect.POCKET_INFO[identity["phone_pocket"]]["model_slot"],
    ]
    if labels.get("slots") != expected_slots:
        result["errors"].append(
            f"设备槽位真值错误：{labels.get('slots')!r}，应为 {expected_slots!r}"
        )
    if labels.get("model_slots") != expected_model_slots:
        result["errors"].append(
            f"模型槽位真值错误：{labels.get('model_slots')!r}，应为 {expected_model_slots!r}"
        )

    orientation = labels["orientation"]
    acceleration = labels["acceleration"]
    motion_valid = labels["motion_valid"]
    grid_time = labels["grid_time_s"]
    uwb_distance = labels["uwb_distance_m"]
    uwb_timestamp = labels["uwb_timestamp_s"]
    uwb_valid = labels["uwb_valid"]
    expected_shapes = {
        "orientation": (600, 2, 3, 3),
        "acceleration": (600, 2, 3),
        "motion_valid": (600, 2),
        "grid_time_s": (600,),
        "uwb_distance_m": (600, 2),
        "uwb_timestamp_s": (600, 2),
        "uwb_valid": (600, 2),
    }
    for key, shape in expected_shapes.items():
        value = labels[key]
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            result["errors"].append(
                f"{key} 形状错误：{tuple(value.shape) if torch.is_tensor(value) else type(value).__name__}，应为 {shape}"
            )

    if result["errors"]:
        return result, None, None, None
    if not math.isclose(float(labels.get("fps", 0)), 30.0, abs_tol=1e-6):
        result["errors"].append(f"fps 不是 30：{labels.get('fps')!r}")
    if int(meta.get("frame_count", -1)) != 600:
        result["errors"].append(f"meta.frame_count 不是 600：{meta.get('frame_count')!r}")
    if int(quality.get("output_frame_count", -1)) != 600:
        result["errors"].append(
            f"quality.output_frame_count 不是 600：{quality.get('output_frame_count')!r}"
        )
    try:
        iphone_r_mi = torch.tensor(meta["calibration"]["iphone"]["R_MI"])
        watch_r_mi = torch.tensor(meta["calibration"]["apple_watch"]["R_MI"])
        if iphone_r_mi.shape != (3, 3) or not torch.allclose(
            iphone_r_mi, watch_r_mi, atol=1e-6, rtol=0
        ):
            result["errors"].append("手机和手表没有保存相同的 3×3 R_MI")
        if not torch.allclose(
            iphone_r_mi.transpose(0, 1) @ iphone_r_mi,
            torch.eye(3),
            atol=2e-3,
            rtol=0,
        ):
            result["errors"].append("meta 中的 R_MI 不是正交矩阵")
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        result["errors"].append(f"标定元数据无法读取：{error}")
    if not bool(motion_valid.all()):
        result["errors"].append(f"IMU 无效帧数：{int((~motion_valid).sum())}")
    if not bool(torch.isfinite(orientation).all()):
        result["errors"].append("orientation 含 NaN 或 Inf")
    if not bool(torch.isfinite(acceleration).all()):
        result["errors"].append("acceleration 含 NaN 或 Inf")
    expected_grid = torch.arange(600, dtype=grid_time.dtype) / 30.0
    if not torch.allclose(grid_time, expected_grid, atol=1e-5, rtol=0):
        result["errors"].append("grid_time_s 不是从 0 开始的 30 Hz 均匀时间轴")

    identity3 = torch.eye(3, dtype=orientation.dtype)
    gram = orientation.transpose(-1, -2) @ orientation
    max_orthogonal_error = float((gram - identity3).abs().max())
    min_determinant = float(torch.linalg.det(orientation).min())
    result["orientation_max_orthogonal_error"] = max_orthogonal_error
    result["orientation_min_determinant"] = min_determinant
    if max_orthogonal_error > 2e-3 or min_determinant < 0.99:
        result["errors"].append(
            f"姿态矩阵不是有效旋转：max|RᵀR-I|={max_orthogonal_error:.6g}, min(det)={min_determinant:.6g}"
        )

    # A UWB value is usable only when fresh, finite, and positive.  Apply this
    # rule to the exported copy without changing the source labels.
    value_valid = torch.isfinite(uwb_distance) & (uwb_distance > 0)
    clean_uwb_valid = uwb_valid.bool() & value_valid
    invalid_positive_mask = uwb_valid.bool() & ~value_valid
    invalid_positive_count = int(invalid_positive_mask.sum())
    if invalid_positive_count:
        result["warnings"].append(
            f"{invalid_positive_count} 个被原文件标为有效的 UWB 帧含非正数或非有限值，导出副本已改为无效"
        )

    uwb_stats = {}
    for device_index, source in enumerate(collect.DEVICES):
        mask = clean_uwb_valid[:, device_index]
        valid_count = int(mask.sum())
        ratio = valid_count / mask.numel()
        longest_gap = longest_false_run(mask)
        uwb_stats[source] = {
            "valid_frames": valid_count,
            "invalid_frames": mask.numel() - valid_count,
            "valid_ratio": ratio,
            "longest_invalid_run_frames": longest_gap,
            "longest_invalid_run_s": longest_gap / 30.0,
        }
        if ratio < min_uwb_valid_ratio:
            result["errors"].append(
                f"{source} UWB 有效率 {ratio:.3%} 低于 {min_uwb_valid_ratio:.1%}"
            )
    result["uwb"] = uwb_stats

    native_time = labels["uwb_native_timestamp_s"]
    native_distance = labels["uwb_native_distance_m"]
    native_source = labels.get("uwb_native_source")
    native_count = native_distance.numel() if torch.is_tensor(native_distance) else -1
    if (
        not torch.is_tensor(native_time)
        or not torch.is_tensor(native_distance)
        or native_time.ndim != 1
        or native_distance.ndim != 1
        or native_time.numel() != native_count
        or not isinstance(native_source, list)
        or len(native_source) != native_count
    ):
        result["errors"].append("原生 UWB 三个字段长度或类型不一致")

    if result["errors"]:
        return result, None, None, None

    exported_labels = dict(labels)
    exported_distance = uwb_distance.clone()
    exported_timestamp = uwb_timestamp.clone()
    exported_distance[~clean_uwb_valid] = float("nan")
    exported_timestamp[~clean_uwb_valid] = 0.0
    exported_labels["uwb_distance_m"] = exported_distance
    exported_labels["uwb_timestamp_s"] = exported_timestamp
    exported_labels["uwb_valid"] = clean_uwb_valid

    native_valid = torch.isfinite(native_distance) & (native_distance > 0)
    if not bool(native_valid.all()):
        bad_native = int((~native_valid).sum())
        result["warnings"].append(f"已从导出副本移除 {bad_native} 个无效原生 UWB 样本")
        keep = native_valid.nonzero().flatten().tolist()
        exported_labels["uwb_native_timestamp_s"] = native_time[native_valid]
        exported_labels["uwb_native_distance_m"] = native_distance[native_valid]
        exported_labels["uwb_native_source"] = [native_source[index] for index in keep]

    exported_meta = copy.deepcopy(meta)
    offline_import = exported_meta.get("offline_import")
    if isinstance(offline_import, dict):
        for key in ("iphone_file", "watch_file"):
            if offline_import.get(key):
                offline_import[key] = Path(str(offline_import[key])).name
    exported_meta["export_audit"] = {
        "uwb_rule": "valid iff original mask is true and distance is finite and > 0 m",
        "invalid_values_are_nan": True,
        "uwb_valid_ratio": {
            source: item["valid_ratio"] for source, item in uwb_stats.items()
        },
    }
    exported_quality = copy.deepcopy(quality)
    files = exported_quality.get("files")
    if isinstance(files, dict):
        for key in ("iphone_file", "watch_file"):
            item = files.get(key)
            if isinstance(item, dict) and item.get("path"):
                item["path"] = Path(str(item["path"])).name
    exported_quality["export_audit"] = {
        "passed": True,
        "minimum_uwb_valid_ratio": min_uwb_valid_ratio,
        "uwb": uwb_stats,
        "sanitized_originally_valid_uwb_frames": invalid_positive_count,
    }
    result["status"] = "included"
    return result, exported_labels, exported_meta, exported_quality


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.min_uwb_valid_ratio <= 1.0:
        raise SystemExit("--min-uwb-valid-ratio 必须在 0 到 1 之间")
    subject_root = BASE_DIR / args.subject
    source = (args.source or subject_root / "test_labels").resolve()
    output = (args.output or subject_root / "test_export").resolve()
    build = output.with_name(output.name + ".__building__")
    if not source.is_dir():
        raise SystemExit(f"源目录不存在：{source}")
    if output.exists():
        raise SystemExit(f"输出目录已经存在，为防止覆盖已停止：{output}")
    if build.exists():
        raise SystemExit(f"临时输出目录已经存在，请先人工确认后处理：{build}")

    plan_path = source / "plan.json"
    planned = []
    if plan_path.is_file():
        planned = json.loads(plan_path.read_text(encoding="utf-8"))
    planned_paths = {
        collect.sequence_directory(source, take).relative_to(source).as_posix()
        for take in planned
    }
    capture_paths = {
        path.parent.relative_to(source).as_posix()
        for path in source.rglob(collect.CAPTURE_FILENAME)
        if "_trial" not in path.relative_to(source).parts
    }
    finalized_dirs = sorted(
        path.parent
        for path in source.rglob("labels.pt")
        if "_trial" not in path.relative_to(source).parts
    )

    results = []
    build.mkdir(parents=True)
    try:
        for directory in finalized_dirs:
            result, labels, meta, quality = audit_sequence(
                directory, source, args.subject, args.min_uwb_valid_ratio
            )
            results.append(result)
            if result["status"] != "included":
                continue
            destination = build / directory.relative_to(source)
            destination.mkdir(parents=True)
            torch.save(labels, destination / "labels.pt")
            write_json(destination / "meta.json", meta)
            write_json(destination / "quality.json", quality)

        included = [item for item in results if item["status"] == "included"]
        excluded = [item for item in results if item["status"] == "excluded"]
        finalized_paths = {item["sequence"] for item in results}
        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "subject": args.subject,
            "source": str(source),
            "output": str(output),
            "policy": {
                "required_files": list(REQUIRED_FILES),
                "frame_count": 600,
                "fps": 30.0,
                "motion_all_frames_valid": True,
                "minimum_uwb_valid_ratio_per_device": args.min_uwb_valid_ratio,
                "invalid_uwb_values": "masked as invalid and stored as NaN in exported labels.pt",
            },
            "counts": {
                "planned": len(planned_paths),
                "captured": len(capture_paths),
                "finalized": len(finalized_paths),
                "included": len(included),
                "excluded": len(excluded),
                "planned_unfinalized": len(planned_paths - finalized_paths),
                "captured_unfinalized": len(capture_paths - finalized_paths),
            },
            "planned_unfinalized": sorted(planned_paths - finalized_paths),
            "captured_unfinalized": sorted(capture_paths - finalized_paths),
            "included": included,
            "excluded": excluded,
        }
        write_json(build / "audit_report.json", report)
        (build / "README.txt").write_text(
            "p02 test_export\n"
            "================\n"
            "每个序列目录只包含 labels.pt、meta.json、quality.json。\n"
            "目录层级：动作/手表_口袋/g01_手机朝向/。\n"
            "labels.pt 中无效 UWB 的 uwb_valid=False，距离保存为 NaN；使用 UWB 时必须读取 uwb_valid。\n"
            "audit_report.json 记录纳入、排除以及尚未整理的全部序列。\n",
            encoding="utf-8",
        )
        build.rename(output)
    except Exception:
        if build.exists():
            shutil.rmtree(build)
        raise

    print(f"审验完成：已整理 {len(results)} 条，纳入 {len(included)} 条，排除 {len(excluded)} 条。")
    print(f"计划未整理 {len(planned_paths - finalized_paths)} 条，其中已有采集清单 {len(capture_paths - finalized_paths)} 条。")
    print(f"导出目录：{output}")
    print(f"审验报告：{output / 'audit_report.json'}")
    if excluded:
        print("排除序列：")
        for item in excluded:
            print(f"  {item['sequence']}：{'；'.join(item['errors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
