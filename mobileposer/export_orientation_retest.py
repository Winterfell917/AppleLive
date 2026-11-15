"""Audit and package a finalized controlled room-orientation retest.

The export contains the protocol, plan, audit report, and only the portable
per-sequence files ``labels.pt``, ``meta.json``, and ``quality.json``.  Raw
NDJSON, plots, and capture manifests stay in the source collection.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent / "data" / "datasets" / "subjects"
SEQUENCE_FILES = ("labels.pt", "meta.json", "quality.json")
EXPECTED_SHAPES = {
    "orientation": (600, 2, 3, 3),
    "acceleration": (600, 2, 3),
    "motion_valid": (600, 2),
    "uwb_distance_m": (600, 2),
    "uwb_valid": (600, 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="p01")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-zip", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def take_directory(take: dict) -> Path:
    yaw = int(
        take["experiment"].get(
            "action_room_yaw_deg", take["experiment"]["room_body_yaw_deg"]
        )
    )
    return Path(
        f"yaw_{yaw:03d}",
        take["action"],
        f"{take['watch']}_{take['phone_pocket']}",
        f"g{int(take['group']):02d}_{take['phone_facing']}",
    )


def issue(result: dict, message: str) -> None:
    result["errors"].append(message)


def audit_sequence(directory: Path, root: Path, expected_take: dict) -> dict:
    relative = directory.relative_to(root)
    result = {"sequence": relative.as_posix(), "status": "excluded", "errors": []}
    missing = [name for name in SEQUENCE_FILES if not (directory / name).is_file()]
    if missing:
        issue(result, f"缺少文件：{', '.join(missing)}")
        return result
    try:
        labels = torch.load(directory / "labels.pt", map_location="cpu", weights_only=False)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        quality = json.loads((directory / "quality.json").read_text(encoding="utf-8"))
    except Exception as error:
        issue(result, f"文件无法读取：{error}")
        return result
    if not isinstance(labels, dict):
        issue(result, "labels.pt 顶层不是字典")
        return result

    for key, expected in expected_take.items():
        if key == "experiment":
            continue
        if labels.get(key) != expected or meta.get(key) != expected:
            issue(
                result,
                f"目录计划与 {key} 不一致：labels={labels.get(key)!r}, meta={meta.get(key)!r}",
            )
    expected_experiment = expected_take["experiment"]
    for container_name, container in (("labels", labels), ("meta", meta), ("quality", quality)):
        experiment = container.get("experiment")
        if not isinstance(experiment, dict):
            issue(result, f"{container_name} 缺少 experiment 元数据")
            continue
        for key in (
            "protocol_id",
            "room_body_yaw_deg",
            "calibration_room_yaw_deg",
            "action_room_yaw_deg",
            "recalibrated_before_action",
            "fixed_conditions",
        ):
            if experiment.get(key) != expected_experiment.get(key):
                issue(result, f"{container_name}.experiment.{key} 与计划不一致")

    for key, shape in EXPECTED_SHAPES.items():
        value = labels.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            issue(result, f"{key} 形状错误")
    if result["errors"]:
        return result
    if not bool(labels["motion_valid"].all()):
        issue(result, f"IMU 无效帧数：{int((~labels['motion_valid']).sum())}")
    if not bool(torch.isfinite(labels["orientation"]).all()):
        issue(result, "orientation 含非有限值")
    if not bool(torch.isfinite(labels["acceleration"]).all()):
        issue(result, "acceleration 含非有限值")
    valid_uwb = labels["uwb_valid"].bool()
    distance = labels["uwb_distance_m"]
    invalid_uwb_values = valid_uwb & (~torch.isfinite(distance) | (distance <= 0))
    if bool(invalid_uwb_values.any()):
        issue(result, f"有效 UWB 帧中有 {int(invalid_uwb_values.sum())} 个非正或非有限距离")
    ratios = {
        source: float(valid_uwb[:, index].float().mean())
        for index, source in enumerate(labels.get("devices", []))
    }
    result["uwb_valid_ratio"] = ratios
    if int(quality.get("output_frame_count", -1)) != 600:
        issue(result, "quality.json 输出帧数不是600")
    if result["errors"]:
        return result
    result["status"] = "included"
    return result


def build_zip(directory: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                output.write(path, Path(directory.name) / path.relative_to(directory))


def main() -> int:
    args = parse_args()
    subject_root = BASE_DIR / args.subject
    source = (args.source or subject_root / "orientation_retest_labels").resolve()
    output = (args.output or subject_root / "orientation_retest_test_export").resolve()
    archive = output.with_suffix(".zip")
    build = output.with_name(output.name + ".__building__")
    if not source.is_dir():
        raise SystemExit(f"源目录不存在：{source}")
    if output.exists() or build.exists() or (archive.exists() and not args.no_zip):
        raise SystemExit("输出目录、临时目录或 ZIP 已存在，为避免覆盖已停止")

    plan_path = source / "plan.json"
    protocol_path = source / "protocol.json"
    if not plan_path.is_file() or not protocol_path.is_file():
        raise SystemExit("源目录缺少 plan.json 或 protocol.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = {take_directory(take).as_posix(): take for take in plan}
    actual = {
        path.parent.relative_to(source).as_posix(): path.parent
        for path in source.rglob("labels.pt")
    }
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "subject": args.subject,
        "protocol_id": protocol.get("protocol_id"),
        "source": str(source),
        "policy": {
            "only_sequence_files": list(SEQUENCE_FILES),
            "frame_count": 600,
            "imu_must_be_valid_for_all_frames": True,
            "valid_uwb_must_be_positive_and_finite": True,
        },
        "missing_planned_sequences": sorted(set(expected) - set(actual)),
        "unexpected_sequences": sorted(set(actual) - set(expected)),
        "sequences": [],
    }
    if report["missing_planned_sequences"] or report["unexpected_sequences"]:
        raise SystemExit("计划与已整理序列不一致，未生成不完整测试包")

    build.mkdir(parents=True)
    try:
        for relative, directory in sorted(actual.items()):
            result = audit_sequence(directory, source, expected[relative])
            report["sequences"].append(result)
            if result["status"] != "included":
                continue
            destination = build / relative
            destination.mkdir(parents=True)
            for name in SEQUENCE_FILES:
                shutil.copy2(directory / name, destination / name)
        included = [item for item in report["sequences"] if item["status"] == "included"]
        excluded = [item for item in report["sequences"] if item["status"] != "included"]
        report["counts"] = {
            "planned": len(expected),
            "finalized": len(actual),
            "included": len(included),
            "excluded": len(excluded),
        }
        if excluded:
            raise RuntimeError("存在未通过审验的序列，未生成部分测试包")
        shutil.copy2(plan_path, build / "plan.json")
        shutil.copy2(protocol_path, build / "protocol.json")
        write_json(build / "audit_report.json", report)
        (build / "README.txt").write_text(
            "Controlled room-orientation retest\n"
            "==================================\n"
            "每条序列包含 labels.pt、meta.json、quality.json。\n"
            "目录层级：yaw_DDD/action/watch_pocket/g01_phone_facing。\n"
            "protocol.json 定义房间方向、固定设备条件和标定协议。\n"
            "audit_report.json 是导出前的完整审验结果。\n",
            encoding="utf-8",
        )
        build.rename(output)
        if not args.no_zip:
            build_zip(output, archive)
    except Exception:
        if build.exists():
            shutil.rmtree(build)
        if output.exists() and not archive.exists():
            shutil.rmtree(output)
        raise

    print(f"审验通过：{len(actual)} 条。")
    print(f"导出目录：{output}")
    if not args.no_zip:
        print(f"ZIP：{archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
