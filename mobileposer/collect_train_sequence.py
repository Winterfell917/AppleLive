"""Record and finalize training sequences from Sensor Read device-local files.

Nine actions, four wears (watch left/right wrist × phone left/right pocket),
three groups per wear, and two pocket facings per group: 9 × 4 × 3 × 2 = 216.
Each take records timed N-pose, forward-walk, and action windows.  The iPhone and
Apple Watch write their own files during capture; no live Watch stream is
required.  The iPhone additionally streams device_motion over UDP only while
N-pose/forward-walk calibration is checked, before the action starts.  Every
capture records the phone sessionID, so all files may be copied by USB after a
collection run.  ``--finalize-batch`` then matches iPhone/Watch NDJSON files to
formal sequence names by sessionID, estimates one shared R_MI from the iPhone,
aligns both devices at 30 FPS, and writes model-frame acceleration/orientation
plus slot truth.

Pocket facings in each group, phone head down:
    screen_in   屏幕朝内，头朝下
    screen_out  屏幕朝外，头朝下

The saved acceleration and orientation are in model coordinates.  The Watch is
not calibrated independently; it uses the R_MI estimated from the iPhone.
The watch bone is slot 0 (lw) or 1 (rw). The phone bone is slot 2 (lp) or 3 (rp).
"""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from collect_eval_sequence import (
    DEVICES,
    SensorReadReceiver,
    average_ris,
    integrate_walk,
    motion_from_values,
    record_grid,
    rmi_from_displacement,
    save_uwb_plot,
    wait_for_streams,
)
from sensor_apple.sensor import APPLE_I_UP_RH, GRAVITY, beep


ACTIONS = (
    "Walking",
    "Jogging",
    "Kicking",
    "Sitting",
    "ArmCrossing",
    "Clapping",
    "JumpingJacks",
    "Boxing",
    "Waving",
)
ACTION_LABEL = {
    "Walking": "走路",
    "Jogging": "慢跑",
    "Kicking": "踢腿",
    "Sitting": "坐",
    "ArmCrossing": "交叉臂",
    "Clapping": "拍手",
    "JumpingJacks": "开合跳",
    "Boxing": "拳击",
    "Waving": "挥手",
}
ACTION_CUE = {
    "Walking": "向前走",
    "Jogging": "慢跑",
    "Kicking": "踢腿",
    "Sitting": "坐下并保持坐姿",
    "ArmCrossing": "交叉双臂",
    "Clapping": "拍手",
    "JumpingJacks": "开合跳",
    "Boxing": "出拳",
    "Waving": "挥手",
}
WATCHES = ("lw", "rw")
POCKETS = ("lp", "rp")
FACINGS = ("screen_in", "screen_out")
GROUPS = (1, 2, 3)
WATCH_INFO = {
    "lw": {"slot": "left_wrist", "model_slot": 0, "label": "左手腕"},
    "rw": {"slot": "right_wrist", "model_slot": 1, "label": "右手腕"},
}
POCKET_INFO = {
    "lp": {"slot": "left_thigh", "model_slot": 2, "label": "左大腿口袋"},
    "rp": {"slot": "right_thigh", "model_slot": 3, "label": "右大腿口袋"},
}
FACING_LABEL = {
    "screen_in": "屏幕朝内，头朝下",
    "screen_out": "屏幕朝外，头朝下",
}

CAPTURE_FILENAME = "capture.json"
LABELS_FILENAME = "labels.pt"
LIVE_CALIBRATION_FILENAME = "live_iphone_calibration.ndjson"
OFFLINE_CAPTURE_SCHEMA_VERSION = 2
DEFAULT_INBOX_DIR = Path(__file__).resolve().parent / "data" / "device_inbox"
DEFAULT_RAW_ROOT = Path(__file__).resolve().parent / "data" / "datasets" / "train"
MAX_NPOSE_ORIENTATION_DEG = 12.0
MAX_NPOSE_ACCEL_RMS_M_S2 = 1.5
MIN_FORWARD_DISTANCE_M = 0.02
CALIBRATION_SOURCE = "iphone"
SOURCE_ALIASES = {
    "apple_watch": "apple_watch",
    "applewatch": "apple_watch",
    "watch": "apple_watch",
    "iphone": "iphone",
    "phone": "iphone",
}


class CalibrationError(RuntimeError):
    """Raised when an offline N-pose/forward-walk calibration is unusable."""


def calibration_success_beep() -> None:
    """Play a distinct ascending double tone after calibration passes."""
    beep(1040, 160)
    time.sleep(0.08)
    beep(1320, 220)


def collection_plan() -> list[dict]:
    """Fix one wear, finish every action, then move the devices.

    Inside one action and wear: group 1..3, and each group is screen_in
    then screen_out. Both facings keep the phone head down.
    """
    plan = []
    for watch in WATCHES:
        for pocket in POCKETS:
            for action in ACTIONS:
                for group in GROUPS:
                    for facing in FACINGS:
                        plan.append(
                            {
                                "action": action,
                                "watch": watch,
                                "phone_pocket": pocket,
                                "group": group,
                                "phone_facing": facing,
                            }
                        )
    return plan


def placement(take: dict) -> dict[str, dict]:
    watch = WATCH_INFO[take["watch"]]
    pocket = POCKET_INFO[take["phone_pocket"]]
    return {
        "apple_watch": watch,
        "iphone": pocket,
    }


def sequence_directory(root: Path, take: dict) -> Path:
    wear = f"{take['watch']}_{take['phone_pocket']}"
    name = f"g{take['group']:02d}_{take['phone_facing']}"
    return root / take["action"] / wear / name


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sample_identity(sample: dict) -> tuple[str, int, int]:
    """Return the stable identity shared by UDP and device-local events."""
    return (
        str(sample.get("session_id", "")),
        int(sample.get("sequence", -1)),
        int(sample.get("timestamp_unix_ns", 0)),
    )


def live_sample_summary(samples: list[dict]) -> dict:
    """Make a compact, human-readable summary for capture.json."""
    if not samples:
        return {"sample_count": 0}
    first = samples[0]
    last = samples[-1]
    return {
        "sample_count": len(samples),
        "session_ids": sorted({str(sample.get("session_id", "")) for sample in samples}),
        "first_sequence": int(first.get("sequence", -1)),
        "last_sequence": int(last.get("sequence", -1)),
        "first_timestamp_unix_ns": int(first.get("timestamp_unix_ns", 0)),
        "last_timestamp_unix_ns": int(last.get("timestamp_unix_ns", 0)),
        "first_received_unix_s": float(first.get("received_unix_s", 0.0)),
        "last_received_unix_s": float(last.get("received_unix_s", 0.0)),
    }


def save_live_calibration_samples(
    path: Path,
    npose_samples: list[dict],
    walk_samples: list[dict],
) -> None:
    """Persist every iPhone UDP sample that actually entered live calibration."""
    with path.open("w", encoding="utf-8") as output:
        for phase, samples in (
            ("npose", npose_samples),
            ("forward_walk", walk_samples),
        ):
            for sample in samples:
                record = {
                    "phase": phase,
                    "source": sample.get("source", CALIBRATION_SOURCE),
                    "sensor": sample.get("sensor", "device_motion"),
                    "sessionID": sample.get("session_id", ""),
                    "sequenceNumber": int(sample.get("sequence", -1)),
                    "timestampUnixNs": int(sample.get("timestamp_unix_ns", 0)),
                    "receivedMonotonicS": float(sample["received_mono"]),
                    "receivedUnixS": float(sample.get("received_unix_s", 0.0)),
                    "values": sample["raw_values"],
                }
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_live_calibration_samples(path: Path) -> dict[str, list[dict]]:
    """Load and validate the exact online iPhone calibration sample trace."""
    phases = {"npose": [], "forward_walk": []}
    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{path}:{line_number}: invalid calibration NDJSON: {error}") from error
            phase = record.get("phase")
            values = record.get("values")
            if phase not in phases or not isinstance(values, dict):
                raise RuntimeError(f"{path}:{line_number}: invalid calibration sample")
            if canonical_source(record.get("source"), "") != CALIBRATION_SOURCE:
                raise RuntimeError(f"{path}:{line_number}: calibration sample is not from iPhone")
            timestamp_ns = record.get("timestampUnixNs")
            if not isinstance(timestamp_ns, (int, float)) or timestamp_ns <= 0:
                raise RuntimeError(f"{path}:{line_number}: missing timestampUnixNs")
            phases[phase].append(record)
    if not phases["npose"] or not phases["forward_walk"]:
        raise RuntimeError(f"calibration trace is incomplete: {path}")
    return phases


def estimate_pc_time_offset(live_trace: dict[str, list[dict]]) -> dict | None:
    """Estimate device->PC Unix offset from live UDP receipt timestamps.

    A device event is aligned to the PC clock with
    ``device_timestamp_s + offset_s``. UDP receipt includes a small positive
    network delay, so the median is used to reject scheduling/latency spikes.
    """
    offsets = []
    for records in live_trace.values():
        for record in records:
            timestamp_ns = record.get("timestampUnixNs")
            received_unix_s = record.get("receivedUnixS")
            if (
                isinstance(timestamp_ns, (int, float))
                and timestamp_ns > 0
                and isinstance(received_unix_s, (int, float))
                and received_unix_s > 0
            ):
                offsets.append(float(received_unix_s) - float(timestamp_ns) / 1_000_000_000.0)
    if not offsets:
        return None
    ordered = sorted(offsets)
    median_offset = float(statistics.median(ordered))
    absolute_deviations = [abs(value - median_offset) for value in ordered]
    return {
        "offset_s": median_offset,
        "sample_count": len(ordered),
        "minimum_s": float(ordered[0]),
        "maximum_s": float(ordered[-1]),
        "median_absolute_deviation_s": float(statistics.median(absolute_deviations)),
        "formula": "receivedUnixS - timestampUnixNs / 1e9",
        "estimator": "median",
    }


def resolve_time_offsets(
    iphone_override: float | None,
    watch_override: float | None,
    live_trace: dict[str, list[dict]] | None,
) -> tuple[dict[str, float], dict]:
    """Resolve per-device offsets, preserving explicit CLI overrides."""
    estimate = estimate_pc_time_offset(live_trace) if live_trace is not None else None
    if iphone_override is not None:
        iphone_offset = float(iphone_override)
        iphone_method = "manual_cli"
    elif estimate is not None:
        iphone_offset = float(estimate["offset_s"])
        iphone_method = "auto_live_udp_median"
    else:
        iphone_offset = 0.0
        iphone_method = "zero_fallback_no_live_trace"

    if watch_override is not None:
        watch_offset = float(watch_override)
        watch_method = "manual_cli"
    else:
        watch_offset = iphone_offset
        watch_method = "inherited_from_iphone"

    offsets = {"iphone": iphone_offset, "apple_watch": watch_offset}
    report = {
        "iphone": {"offset_s": iphone_offset, "method": iphone_method},
        "apple_watch": {"offset_s": watch_offset, "method": watch_method},
        "live_udp_estimate": estimate,
    }
    return offsets, report


def timed_window(label: str, duration_s: float) -> dict:
    start_unix_s = time.time()
    start_monotonic_s = time.monotonic()
    print(f"{label}：{duration_s:.1f} 秒。", flush=True)
    time.sleep(duration_s)
    return {
        "start_unix_s": start_unix_s,
        "end_unix_s": time.time(),
        "requested_duration_s": duration_s,
        "measured_duration_s": time.monotonic() - start_monotonic_s,
    }


def wait_for_iphone_stream(receiver: SensorReadReceiver, timeout_s: float = 30.0) -> None:
    """Wait only for the live iPhone stream used by immediate calibration."""
    started = time.monotonic()
    last_report = 0.0
    while receiver.latest_motion(CALIBRATION_SOURCE) is None:
        now = time.monotonic()
        if now - started >= timeout_s:
            raise TimeoutError(
                "等待 iPhone device_motion 超时；请检查手机 UDP 开关、电脑 IP、"
                "端口和 Windows 防火墙"
            )
        if now - last_report >= 1.0:
            print("等待 iPhone device_motion...", flush=True)
            last_report = now
        time.sleep(0.02)


def capture_manifest(
    take: dict,
    fps: float,
    collection_start_unix_s: float,
    npose_window: dict,
    walk_window: dict,
    action_window: dict,
) -> dict:
    slots = placement(take)
    return {
        "schema_version": OFFLINE_CAPTURE_SCHEMA_VERSION,
        "attempt_id": datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        "status": "awaiting_device_stop",
        "capture_mode": "device_local_files",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "action": take["action"],
        "action_zh": ACTION_LABEL[take["action"]],
        "watch": take["watch"],
        "watch_slot": slots["apple_watch"]["slot"],
        "phone_pocket": take["phone_pocket"],
        "phone_slot": slots["iphone"]["slot"],
        "group": take["group"],
        "phone_facing": take["phone_facing"],
        "phone_facing_zh": FACING_LABEL[take["phone_facing"]],
        "devices": list(DEVICES),
        "slots": {source: slots[source]["slot"] for source in DEVICES},
        "model_slots": {source: slots[source]["model_slot"] for source in DEVICES},
        "target_fps": fps,
        "collection": {
            "start_confirmed_unix_s": collection_start_unix_s,
            "stop_confirmed_unix_s": None,
            "watch_transfer_confirmed": False,
        },
        "windows": {
            "npose": npose_window,
            "forward_walk": walk_window,
            "action": action_window,
        },
        "raw_files": {"iphone": None, "apple_watch": None},
        "failed_attempts": [],
        "artifacts": {
            "capture": CAPTURE_FILENAME,
            "live_calibration": None,
            "raw": None,
            "labels": None,
            "quality": None,
        },
        "notes": (
            "Apple Watch and iPhone recorded locally. The successful live iPhone "
            "calibration samples are saved explicitly and matched by sessionID, "
            "sequenceNumber, and timestampUnixNs during --finalize."
        ),
    }


def record_offline_take(
    root: Path,
    take: dict,
    index: int,
    total: int,
    args: argparse.Namespace,
    directory: Path | None = None,
) -> str:
    """Guide local recording with live iPhone-only calibration validation."""
    directory = directory or sequence_directory(root, take)
    capture_path = directory / CAPTURE_FILENAME
    failed_attempts = []
    if (directory / LABELS_FILENAME).is_file():
        print(f"[{index}/{total}] 已完成 labels.pt，跳过 {directory}")
        return "skipped"
    if capture_path.is_file():
        existing = json.loads(capture_path.read_text(encoding="utf-8"))
        if existing.get("status") == "calibration_failed":
            failed_attempts = list(existing.get("failed_attempts", []))
            failed_attempts.append(
                {
                    "attempt_id": existing.get("attempt_id"),
                    "windows": existing.get("windows"),
                    "raw_files": existing.get("raw_files"),
                    "calibration_failure": existing.get("calibration_failure"),
                }
            )
            print(f"[{index}/{total}] 上一次标定失败，必须重新标定并重录本条。")
        else:
            if args.defer_import:
                print(f"[{index}/{total}] 已登记待批量整理，跳过：{directory}")
            else:
                print(f"[{index}/{total}] 已有 capture.json，等待本条设备文件：{directory}")
            if args.defer_import:
                return "skipped"
            receive_status = receive_and_finalize(directory, root, args)
            if receive_status == "calibration_failed":
                answer = input("按回车重新标定并重录同一条；输入 q 停止: ").strip().lower()
                if answer == "q":
                    return "quit"
                return record_offline_take(root, take, index, total, args, directory)
            return "quit" if receive_status == "quit" else "saved"

    print(take_prompt(take, index, total), flush=True)
    if take["phone_facing"] == "screen_in":
        print("请把手机放成屏幕朝内、头朝下。不要换手腕，不要换口袋。", flush=True)
    else:
        print("请把手机翻成屏幕朝外、头仍朝下。不要换手腕，不要换口袋。", flush=True)
    answer = input(
        "回车重新标定并重录本条，q 结束: "
        if failed_attempts
        else "回车开始本条，s 跳过，q 结束: "
    ).strip().lower()
    if answer == "q":
        return "quit"
    if answer == "s" and not failed_attempts:
        print(f"已跳过 {directory}")
        return "skipped"

    input(
        "请在 iPhone 和 Apple Watch 上开始一段新的本地采集，并打开 iPhone 的 UDP "
        f"实时传输到本机端口 {args.port}；确认后立即按回车。"
    )
    collection_start_unix_s = time.time()
    live_result = None
    while live_result is None:
        try:
            with SensorReadReceiver(args.host, args.port) as receiver:
                print(f"正在监听 iPhone 标定流 udp://{args.host}:{args.port}")
                wait_for_iphone_stream(receiver)
                print("已收到 iPhone device_motion；Apple Watch 无需实时传输。")
                while live_result is None:
                    live_result = calibrate_once(
                        receiver,
                        args.npose,
                        args.walk,
                        take,
                        include_windows=True,
                    )
                    if live_result is not None:
                        break
                    answer = input(
                        "iPhone 标定未通过。直接按回车重新标定，q 停止本条: "
                    ).strip().lower()
                    if answer == "q":
                        print("请在 iPhone 和 Apple Watch 上停止当前本地采集。")
                        return "quit"
        except (OSError, RuntimeError, TimeoutError) as error:
            print(f"iPhone 实时标定流不可用：{error}")
            answer = input(
                "检查手机 UDP 地址/端口和防火墙后按回车重新监听，q 停止本条: "
            ).strip().lower()
            if answer == "q":
                print("请在 iPhone 和 Apple Watch 上停止当前本地采集。")
                return "quit"

    live_calibration, windows, live_samples = live_result
    npose_window = windows["npose"]
    walk_window = windows["forward_walk"]
    calibration_session_ids = sorted(
        {
            str(sample.get("session_id", ""))
            for phase_samples in live_samples.values()
            for sample in phase_samples
            if str(sample.get("session_id", ""))
        }
    )
    if len(calibration_session_ids) != 1:
        found = ", ".join(calibration_session_ids) or "无"
        print(
            "无法为本条建立唯一的手机文件对应关系："
            f"在线标定中发现的 sessionID 为 {found}。"
        )
        print("请停止手机和手表的本次本地采集，并在两台设备上重新开始一个新会话。")
        answer = input("准备好后按回车重新采集本条；输入 q 停止: ").strip().lower()
        if answer == "q":
            return "quit"
        return record_offline_take(root, take, index, total, args, directory)
    directory.mkdir(parents=True, exist_ok=True)
    live_trace_path = directory / LIVE_CALIBRATION_FILENAME
    save_live_calibration_samples(
        live_trace_path,
        live_samples["npose"],
        live_samples["forward_walk"],
    )
    input(
        "iPhone 标定成功。现在关闭手机的 UDP 实时传输（不要停止本地采集）；"
        "确认关闭后按回车开始准备正式动作。"
    )

    cue = ACTION_CUE[take["action"]]
    input(f"回到动作起点，准备做{ACTION_LABEL[take['action']]}（{cue}）；按回车继续。")
    beep(880, 200)
    action_window = timed_window(
        f"正式动作 {take['action']}（{ACTION_LABEL[take['action']]}）",
        args.seconds,
    )
    beep(660, 250)
    print("动作窗口结束。", flush=True)

    manifest = capture_manifest(
        take,
        args.fps,
        collection_start_unix_s,
        npose_window,
        walk_window,
        action_window,
    )
    manifest["failed_attempts"] = failed_attempts
    phone_calibration = live_calibration[CALIBRATION_SOURCE]
    manifest["live_iphone_calibration"] = {
        "status": "passed",
        "source": CALIBRATION_SOURCE,
        "R_MI": phone_calibration["R_MI"].tolist(),
        "horizontal_m": phone_calibration["horizontal_m"],
        "vertical_m": phone_calibration["vertical_m"],
        "npose_sample_count": phone_calibration["npose_sample_count"],
        "walk_sample_count": phone_calibration["walk_sample_count"],
        "npose_quality": phone_calibration["npose_quality"],
        "watch_uses_iphone_r_mi": True,
        "sample_selection_method": "explicit_event_identity",
        "sample_artifact": LIVE_CALIBRATION_FILENAME,
        "npose": live_sample_summary(live_samples["npose"]),
        "forward_walk": live_sample_summary(live_samples["forward_walk"]),
    }
    manifest["expected_session_ids"] = calibration_session_ids
    manifest["file_matching"] = {
        "strategy": "sessionID",
        "expected_session_ids": calibration_session_ids,
        "requires_same_session_for_iphone_and_watch": True,
    }
    manifest["artifacts"]["live_calibration"] = LIVE_CALIBRATION_FILENAME
    write_json(capture_path, manifest)

    answer = input(
        "现在结束手机和手表采集，并等待 Watch 文件完整传到手机。"
        "完成后按回车；输入 q 保存当前清单并退出: "
    ).strip().lower()
    manifest["collection"]["stop_confirmed_unix_s"] = time.time()
    manifest["collection"]["watch_transfer_confirmed"] = answer != "q"
    manifest["status"] = (
        "awaiting_raw_files" if answer != "q" else "device_transfer_unconfirmed"
    )
    write_json(capture_path, manifest)
    print(f"已保存采集清单：{capture_path}")
    if answer == "q":
        print("尚未接收设备文件；稍后可单条运行 --finalize，或正式序列运行 --finalize-batch。")
        return "quit"
    if args.defer_import:
        session_text = ", ".join(calibration_session_ids) or "未记录"
        print(f"已登记手机 sessionID：{session_text}")
        print("本条无需现在连接 USB；全部动作录完后运行 --finalize-batch 统一整理。")
        return "saved"
    receive_status = receive_and_finalize(directory, root, args)
    if receive_status == "calibration_failed":
        answer = input("按回车重新标定并重录同一条；输入 q 停止: ").strip().lower()
        if answer == "q":
            return "quit"
        return record_offline_take(root, take, index, total, args, directory)
    return "quit" if receive_status == "quit" else "saved"


def calibrate_once(
    receiver: SensorReadReceiver,
    npose_s: float,
    walk_s: float,
    take: dict,
    include_windows: bool = False,
) -> dict[str, dict] | tuple[dict[str, dict], dict[str, dict], dict[str, list[dict]]] | None:
    input("站成 N-pose，面向前方，按回车。保持不动，听到提示音后再向前走。")
    print(f"保持不动 {npose_s:.0f} 秒。", flush=True)
    npose_started_unix = time.time()
    npose_started_mono = time.monotonic()
    time.sleep(npose_s)
    npose_ended_mono = time.monotonic()
    npose_ended_unix = time.time()
    npose_measured_s = npose_ended_mono - npose_started_mono
    npose_samples = receiver.motion_between(
        CALIBRATION_SOURCE,
        npose_started_mono,
        npose_ended_mono,
    )
    ris_npose = average_ris(npose_samples)
    if ris_npose is None:
        print("iPhone 标定失败：N-pose 样本不足")
        return None
    stability = npose_quality(npose_samples, ris_npose)
    if stability["max_orientation_deviation_deg"] > MAX_NPOSE_ORIENTATION_DEG:
        print(
            "iPhone 标定失败：N-pose 姿态变化 "
            f"{stability['max_orientation_deviation_deg']:.1f} deg，"
            f"超过 {MAX_NPOSE_ORIENTATION_DEG:.1f} deg"
        )
        return None
    if stability["acceleration_rms_m_s2"] > MAX_NPOSE_ACCEL_RMS_M_S2:
        print(
            "iPhone 标定失败：N-pose 加速度 RMS "
            f"{stability['acceleration_rms_m_s2']:.2f} m/s^2，"
            f"超过 {MAX_NPOSE_ACCEL_RMS_M_S2:.2f} m/s^2"
        )
        return None
    print("iPhone N-pose 检查通过。", flush=True)
    beep()
    print("向前走；只使用 iPhone 计算 R_MI。", flush=True)
    walk_started_unix = time.time()
    walk_started_mono = time.monotonic()
    deadline = walk_started_mono + walk_s + 5.0
    while time.monotonic() < deadline:
        samples = receiver.motion_since(CALIBRATION_SOURCE, walk_started_mono)
        if len(samples) >= 2 and samples[-1]["timestamp_s"] - samples[0]["timestamp_s"] >= walk_s:
            break
        time.sleep(0.01)
    else:
        print("走路窗口内没有收齐 iPhone 数据。")
        samples = receiver.motion_since(CALIBRATION_SOURCE, walk_started_mono)
        if len(samples) < 2:
            print(f"  {CALIBRATION_SOURCE}: {len(samples)} 条")
        else:
            span = samples[-1]["timestamp_s"] - samples[0]["timestamp_s"]
            print(
                f"  {CALIBRATION_SOURCE}: {len(samples)} 条，"
                f"时间戳跨度 {span:.2f} s（需要 {walk_s:.0f} s）"
        )
        return None

    walk_ended_mono = time.monotonic()
    walk_ended_unix = time.time()
    walk_measured_s = walk_ended_mono - walk_started_mono
    slots = placement(take)
    walk_samples = receiver.motion_between(
        CALIBRATION_SOURCE,
        walk_started_mono,
        walk_ended_mono,
    )
    integrated = integrate_walk(walk_samples, walk_s) if walk_samples else None
    if integrated is None:
        print("iPhone 标定失败：向前走样本不足")
        return None

    position, velocity, duration_s = integrated
    corrected = position - 0.5 * duration_s * velocity
    vertical_m = float(torch.dot(corrected, APPLE_I_UP_RH))
    built = rmi_from_displacement(corrected)
    horizontal_m = 0.0 if built is None else built[1]
    print(
        "Calibration motion iphone: "
        f"pI={position.tolist()}, vI={velocity.tolist()}, "
        f"horizontal={horizontal_m:.4f} m",
        flush=True,
    )
    if built is None or horizontal_m < MIN_FORWARD_DISTANCE_M:
        print(f"iPhone 标定失败：没有积出水平前进方向（水平 {horizontal_m:.3f} m）")
        return None

    rotation_mi = built[0]
    results = {}
    for source in DEVICES:
        info = slots[source]
        item = {
            "source": source,
            "slot": info["slot"],
            "model_slot": info["model_slot"],
            "ok": True,
            "R_MI": rotation_mi.clone(),
            "calibration_source": CALIBRATION_SOURCE,
        }
        if source == CALIBRATION_SOURCE:
            item.update(
                {
                    "horizontal_m": horizontal_m,
                    "vertical_m": vertical_m,
                    "npose_sample_count": len(npose_samples),
                    "walk_sample_count": len(walk_samples),
                    "npose_quality": stability,
                }
            )
        results[source] = item
    print(
        f"iPhone 标定通过：水平 {horizontal_m:.2f} m，沿向上轴 {vertical_m:.2f} m；"
        "Apple Watch 复用 iPhone R_MI。",
        flush=True,
    )
    calibration_success_beep()
    if include_windows:
        return (
            results,
            {
                "npose": {
                    "start_unix_s": npose_started_unix,
                    "end_unix_s": npose_ended_unix,
                    "requested_duration_s": npose_s,
                    "measured_duration_s": npose_measured_s,
                },
                "forward_walk": {
                    "start_unix_s": walk_started_unix,
                    "end_unix_s": walk_ended_unix,
                    "requested_duration_s": walk_s,
                    "measured_duration_s": walk_measured_s,
                },
            },
            {
                "npose": npose_samples,
                "forward_walk": walk_samples,
            },
        )
    return results


def model_space_sequences(
    calibration: dict[str, dict],
    grid: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return R_MS and aM using the iPhone R_MI for both devices."""
    shared_rmi = calibration[CALIBRATION_SOURCE]["R_MI"]
    for source in DEVICES:
        if not torch.allclose(calibration[source]["R_MI"], shared_rmi, atol=1e-6, rtol=1e-6):
            raise RuntimeError(f"{source} R_MI is not the shared iPhone R_MI")
    orientation = torch.matmul(shared_rmi.view(1, 1, 3, 3), grid["RIS"])
    acceleration = torch.matmul(orientation, grid["aS"].unsqueeze(-1)).squeeze(-1)
    return orientation.contiguous(), acceleration.contiguous()


def save_sequence(
    directory: Path,
    take: dict,
    fps: float,
    calibration: dict[str, dict],
    grid: dict[str, torch.Tensor],
    uwb_samples: list[dict],
    record_window: tuple[float, float],
    offline_info: dict | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    slots = placement(take)
    orientation, acceleration = model_space_sequences(calibration, grid)
    payload = {
        "schema_version": 2,
        "devices": list(DEVICES),
        "slots": [slots[source]["slot"] for source in DEVICES],
        "model_slots": [slots[source]["model_slot"] for source in DEVICES],
        "action": take["action"],
        "watch": take["watch"],
        "phone_pocket": take["phone_pocket"],
        "group": take["group"],
        "phone_facing": take["phone_facing"],
        "fps": fps,
        "capture_mode": "device_local_files" if offline_info else "legacy_live_udp",
        "orientation": orientation,
        "acceleration": acceleration,
        "motion_valid": grid["motion_valid"],
    }
    torch.save(payload, directory / "labels.pt")
    save_uwb_plot(directory / "uwb_distance.png", uwb_samples, record_window[0])
    meta = {
        "action": take["action"],
        "action_zh": ACTION_LABEL[take["action"]],
        "watch": take["watch"],
        "watch_slot": slots["apple_watch"]["slot"],
        "phone_pocket": take["phone_pocket"],
        "phone_slot": slots["iphone"]["slot"],
        "group": take["group"],
        "phone_facing": take["phone_facing"],
        "phone_facing_zh": FACING_LABEL[take["phone_facing"]],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "devices": list(DEVICES),
        "slots": {source: slots[source]["slot"] for source in DEVICES},
        "model_slots": {source: slots[source]["model_slot"] for source in DEVICES},
        "fps": fps,
        "frame_count": int(acceleration.shape[0]),
        "record_start_unix_s": record_window[0],
        "record_end_unix_s": record_window[1],
        "convention": {
            "orientation": "R_MS = shared iPhone R_MI @ RIS; maps sensor vectors to model coordinates",
            "acceleration": "aM = shared iPhone R_MI @ RIS @ aS; model frame, m/s^2",
            "inertial_up": APPLE_I_UP_RH.tolist(),
            "R_MI": "rows are model X,Y,Z in I; v_M = R_MI @ v_I; up axis is APPLE_I_UP_RH",
            "uwb": "diagnostic only; UWB is not stored in labels.pt",
            "phone_facing": "screen_in = screen faces the thigh, head down; screen_out = screen faces out of the pocket, head down",
        },
        "label_fields": {
            "model_slots": "integer device-slot ground truth in devices order",
            "orientation": "[frame, device, 3, 3] model-frame device orientation",
            "acceleration": "[frame, device, 3] model-frame linear acceleration in m/s^2",
            "motion_valid": "[frame, device] freshness mask for the saved sequences",
        },
        "calibration": {
            source: {
                "slot": calibration[source]["slot"],
                "model_slot": calibration[source]["model_slot"],
                "calibration_source": calibration[source]["calibration_source"],
                "calibrated_independently": source == CALIBRATION_SOURCE,
                "horizontal_m": calibration[source].get("horizontal_m"),
                "vertical_m": calibration[source].get("vertical_m"),
                "R_MI": calibration[source]["R_MI"].tolist(),
                "npose_sample_count": calibration[source].get("npose_sample_count"),
                "walk_sample_count": calibration[source].get("walk_sample_count"),
                "npose_quality": calibration[source].get("npose_quality"),
                "sample_selection_method": calibration[source].get("sample_selection_method"),
                "r_mi_comparison": calibration[source].get("r_mi_comparison"),
            }
            for source in DEVICES
        },
    }
    if offline_info:
        meta["offline_import"] = offline_info
    (directory / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def canonical_source(value: object, fallback: str) -> str:
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in SOURCE_ALIASES:
        return SOURCE_ALIASES[key]
    return fallback if not key or key == "device" else key


def event_timestamp_s(event: dict) -> float:
    value = event.get("timestampUnixNs")
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("missing positive timestampUnixNs")
    return float(value) / 1_000_000_000.0


def load_device_events(path: Path, fallback_source: str) -> tuple[list[dict], dict]:
    events = []
    total_lines = 0
    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            total_lines += 1
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise TypeError("event is not an object")
                event_timestamp_s(event)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise RuntimeError(f"{path}:{line_number}: invalid NDJSON event: {error}") from error
            event = dict(event)
            event["source"] = canonical_source(event.get("source"), fallback_source)
            events.append(event)
    if not events:
        raise RuntimeError(f"raw file is empty: {path}")
    return events, {"path": str(path.resolve()), "line_count": total_lines}


def merge_device_events(iphone_path: Path, watch_path: Path) -> tuple[list[dict], dict]:
    iphone_events, iphone_report = load_device_events(iphone_path, "iphone")
    watch_events, watch_report = load_device_events(watch_path, "apple_watch")
    merged = []
    seen = set()
    duplicate_count = 0
    for event in iphone_events + watch_events:
        key = (
            event.get("source"),
            event.get("sensor"),
            str(event.get("sessionID", "")),
            int(event.get("sequenceNumber", -1)),
            int(event.get("timestampUnixNs", 0)),
        )
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        merged.append(event)
    merged.sort(key=event_timestamp_s)
    return merged, {
        "iphone_file": iphone_report,
        "watch_file": watch_report,
        "merged_event_count": len(merged),
        "duplicate_event_count": duplicate_count,
    }


def extract_offline_streams(
    events: list[dict],
    time_offsets_s: dict[str, float],
) -> tuple[dict[str, list[dict]], list[dict], dict]:
    motion = {source: [] for source in DEVICES}
    uwb = []
    rejected_motion = {source: 0 for source in DEVICES}
    session_ids = {source: set() for source in DEVICES}
    for event in events:
        source = canonical_source(event.get("source"), "")
        if source not in DEVICES:
            continue
        session_id = str(event.get("sessionID", ""))
        if session_id:
            session_ids[source].add(session_id)
        sensor = event.get("sensor")
        values = event.get("values")
        if not isinstance(values, dict):
            continue
        timestamp_s = event_timestamp_s(event) + time_offsets_s[source]
        sequence = int(event.get("sequenceNumber", -1))
        if sensor == "device_motion":
            try:
                acceleration_s, ris, acceleration_i = motion_from_values(values)
            except (KeyError, TypeError, ValueError):
                rejected_motion[source] += 1
                continue
            motion[source].append(
                {
                    "source": source,
                    "sensor": sensor,
                    "session_id": session_id,
                    "sequence": sequence,
                    "timestamp_unix_ns": int(event["timestampUnixNs"]),
                    "timestamp_s": timestamp_s,
                    "raw_values": dict(values),
                    "aS": acceleration_s,
                    "RIS": ris,
                    "aI": acceleration_i,
                }
            )
        elif sensor == "uwb_ranging":
            distance = values.get("distance_m")
            if isinstance(distance, (int, float)):
                uwb.append(
                    {
                        "source": source,
                        "sequence": sequence,
                        "timestamp_s": timestamp_s,
                        "distance_m": float(distance),
                    }
                )

    for source in DEVICES:
        motion[source].sort(key=lambda item: item["timestamp_s"])
        deduplicated = {}
        for sample in motion[source]:
            deduplicated[sample_identity(sample)] = sample
        motion[source] = list(deduplicated.values())
        motion[source].sort(key=lambda item: item["timestamp_s"])
    uwb.sort(key=lambda item: (item["timestamp_s"], item["source"]))
    report = {
        "motion_sample_count": {source: len(motion[source]) for source in DEVICES},
        "rejected_motion_event_count": rejected_motion,
        "uwb_sample_count": len(uwb),
        "session_ids": {source: sorted(values) for source, values in session_ids.items()},
    }
    return motion, uwb, report


def window_bounds(capture: dict, name: str) -> tuple[float, float]:
    window = capture["windows"][name]
    return float(window["start_unix_s"]), float(window["end_unix_s"])


def samples_between(samples: list[dict], start_s: float, end_s: float) -> list[dict]:
    return [sample for sample in samples if start_s <= sample["timestamp_s"] <= end_s]


def npose_quality(samples: list[dict], mean_ris: torch.Tensor) -> dict:
    if not samples:
        return {"max_orientation_deviation_deg": None, "acceleration_rms_m_s2": None}
    angles = []
    for sample in samples:
        relative = mean_ris.transpose(0, 1) @ sample["RIS"]
        cosine = ((torch.trace(relative) - 1.0) / 2.0).clamp(-1.0, 1.0)
        angles.append(float(torch.rad2deg(torch.acos(cosine))))
    acceleration_rms = float(
        torch.sqrt(
            torch.stack([sample["aS"].square().sum() for sample in samples]).mean()
        )
    )
    return {
        "max_orientation_deviation_deg": max(angles),
        "acceleration_rms_m_s2": acceleration_rms,
    }


def rotation_difference_deg(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the geodesic angle between two 3-D rotation matrices."""
    relative = left @ right.transpose(0, 1)
    cosine = ((torch.trace(relative) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)))


def trace_record_identity(record: dict) -> tuple[str, int, int]:
    return (
        str(record.get("sessionID", "")),
        int(record.get("sequenceNumber", -1)),
        int(record.get("timestampUnixNs", 0)),
    )


def match_trace_samples(
    local_samples: list[dict],
    trace_records: list[dict],
    phase: str,
) -> tuple[list[dict], dict]:
    """Select the exact device-local events that were received online."""
    local_by_identity = {sample_identity(sample): sample for sample in local_samples}
    matched = []
    missing = []
    raw_value_mismatch_count = 0
    for record in trace_records:
        identity = trace_record_identity(record)
        sample = local_by_identity.get(identity)
        if sample is None:
            missing.append(identity)
            continue
        if sample.get("raw_values") != record.get("values"):
            raw_value_mismatch_count += 1
        matched.append(sample)
    if missing:
        preview = ", ".join(str(identity) for identity in missing[:3])
        suffix = " ..." if len(missing) > 3 else ""
        raise CalibrationError(
            f"offline iPhone file is missing {len(missing)} online {phase} samples: "
            f"{preview}{suffix}"
        )
    return matched, {
        "online_trace_count": len(trace_records),
        "matched_local_count": len(matched),
        "missing_local_count": len(missing),
        "raw_value_mismatch_count": raw_value_mismatch_count,
    }


def motion_samples_from_trace(trace_records: list[dict]) -> list[dict]:
    """Rebuild processed motion from saved online raw values for audit replay."""
    samples = []
    for record in trace_records:
        acceleration_s, ris, acceleration_i = motion_from_values(record["values"])
        timestamp_ns = int(record["timestampUnixNs"])
        samples.append(
            {
                "source": CALIBRATION_SOURCE,
                "sensor": "device_motion",
                "session_id": str(record.get("sessionID", "")),
                "sequence": int(record.get("sequenceNumber", -1)),
                "timestamp_unix_ns": timestamp_ns,
                "timestamp_s": timestamp_ns / 1_000_000_000.0,
                "raw_values": dict(record["values"]),
                "aS": acceleration_s,
                "RIS": ris,
                "aI": acceleration_i,
            }
        )
    return samples


def compute_iphone_calibration(
    npose_samples: list[dict],
    walk_samples: list[dict],
    walk_span_s: float,
    coverage: str,
    context: str,
) -> dict:
    """Run the shared offline calibration checks on one selected sample set."""
    ris_npose = average_ris(npose_samples)
    integrated = integrate_walk(walk_samples, walk_span_s)
    errors = []
    if len(npose_samples) < 10 or ris_npose is None or integrated is None:
        errors.append(
            f"iphone: N-pose {len(npose_samples)} samples, walk {len(walk_samples)} "
            f"samples; corrected raw coverage {coverage}"
        )
    else:
        stability = npose_quality(npose_samples, ris_npose)
        if stability["max_orientation_deviation_deg"] > MAX_NPOSE_ORIENTATION_DEG:
            errors.append(
                "iphone: N-pose moved too much; maximum orientation deviation "
                f"{stability['max_orientation_deviation_deg']:.1f} deg "
                f"(limit {MAX_NPOSE_ORIENTATION_DEG:.1f})"
            )
        if stability["acceleration_rms_m_s2"] > MAX_NPOSE_ACCEL_RMS_M_S2:
            errors.append(
                "iphone: N-pose acceleration RMS "
                f"{stability['acceleration_rms_m_s2']:.2f} m/s^2 "
                f"(limit {MAX_NPOSE_ACCEL_RMS_M_S2:.2f})"
            )
        position, velocity, duration_s = integrated
        corrected = position - 0.5 * duration_s * velocity
        vertical_m = float(torch.dot(corrected, APPLE_I_UP_RH))
        built = rmi_from_displacement(corrected)
        horizontal_m = 0.0 if built is None else built[1]
        if built is None or horizontal_m < MIN_FORWARD_DISTANCE_M:
            errors.append(
                f"iphone: forward-walk horizontal displacement is only {horizontal_m:.3f} m "
                f"(minimum {MIN_FORWARD_DISTANCE_M:.3f})"
            )
    if errors:
        raise CalibrationError(
            f"offline iPhone calibration failed ({context}):\n  "
            + "\n  ".join(errors)
            + "\nCheck that the two files belong to this take and use --iphone-time-offset/"
            "--watch-time-offset if device timestamps differ from the PC."
        )
    return {
        "R_MI": built[0],
        "horizontal_m": horizontal_m,
        "vertical_m": vertical_m,
        "npose_sample_count": len(npose_samples),
        "walk_sample_count": len(walk_samples),
        "npose_quality": stability,
    }


def calibrate_from_files(
    motion: dict[str, list[dict]],
    capture: dict,
    take: dict,
    live_trace: dict[str, list[dict]] | None = None,
) -> dict[str, dict]:
    npose_start, npose_end = window_bounds(capture, "npose")
    walk_start, walk_end = window_bounds(capture, "forward_walk")
    slots = placement(take)
    source = CALIBRATION_SOURCE
    coverage = (
        "none"
        if not motion[source]
        else f"{motion[source][0]['timestamp_s']:.3f}..{motion[source][-1]['timestamp_s']:.3f}"
    )
    window_npose_samples = samples_between(motion[source], npose_start, npose_end)
    window_walk_samples = samples_between(motion[source], walk_start, walk_end)
    comparison = {
        "selection_method": "pc_unix_window",
        "window_sample_counts": {
            "npose": len(window_npose_samples),
            "forward_walk": len(window_walk_samples),
        },
    }

    if live_trace is None:
        selected = compute_iphone_calibration(
            window_npose_samples,
            window_walk_samples,
            walk_end - walk_start,
            coverage,
            "PC Unix time window fallback",
        )
        live_rmi_value = capture.get("live_iphone_calibration", {}).get("R_MI")
        if live_rmi_value is not None:
            live_rmi = torch.tensor(live_rmi_value, dtype=selected["R_MI"].dtype)
            comparison["online_saved_vs_offline_window_deg"] = rotation_difference_deg(
                live_rmi, selected["R_MI"]
            )
    else:
        exact_npose, npose_match = match_trace_samples(
            motion[source], live_trace["npose"], "N-pose"
        )
        exact_walk, walk_match = match_trace_samples(
            motion[source], live_trace["forward_walk"], "forward-walk"
        )
        requested_walk_s = float(
            capture["windows"]["forward_walk"].get(
                "requested_duration_s", walk_end - walk_start
            )
        )
        selected = compute_iphone_calibration(
            exact_npose,
            exact_walk,
            requested_walk_s,
            coverage,
            "exact live UDP event identities",
        )
        trace_replay = compute_iphone_calibration(
            motion_samples_from_trace(live_trace["npose"]),
            motion_samples_from_trace(live_trace["forward_walk"]),
            requested_walk_s,
            "saved online calibration trace",
            "saved online raw-value replay",
        )
        comparison.update(
            {
                "selection_method": "explicit_event_identity",
                "npose_match": npose_match,
                "forward_walk_match": walk_match,
                "offline_exact_vs_trace_replay_deg": rotation_difference_deg(
                    selected["R_MI"], trace_replay["R_MI"]
                ),
            }
        )
        live_rmi_value = capture.get("live_iphone_calibration", {}).get("R_MI")
        if live_rmi_value is not None:
            live_rmi = torch.tensor(live_rmi_value, dtype=selected["R_MI"].dtype)
            comparison["online_saved_vs_offline_exact_deg"] = rotation_difference_deg(
                live_rmi, selected["R_MI"]
            )
            comparison["online_saved_vs_trace_replay_deg"] = rotation_difference_deg(
                live_rmi, trace_replay["R_MI"]
            )
        try:
            full_window = compute_iphone_calibration(
                window_npose_samples,
                window_walk_samples,
                walk_end - walk_start,
                coverage,
                "full PC Unix time window diagnostic",
            )
        except CalibrationError as error:
            comparison["full_window"] = {"status": "failed", "reason": str(error)}
        else:
            comparison["full_window"] = {
                "status": "passed",
                "npose_sample_count": full_window["npose_sample_count"],
                "walk_sample_count": full_window["walk_sample_count"],
                "horizontal_m": full_window["horizontal_m"],
                "exact_vs_full_window_deg": rotation_difference_deg(
                    selected["R_MI"], full_window["R_MI"]
                ),
            }

    rotation_mi = selected["R_MI"]
    results = {}
    for device_source in DEVICES:
        info = slots[device_source]
        item = {
            "source": device_source,
            "slot": info["slot"],
            "model_slot": info["model_slot"],
            "ok": True,
            "R_MI": rotation_mi.clone(),
            "calibration_source": CALIBRATION_SOURCE,
        }
        if device_source == CALIBRATION_SOURCE:
            item.update(
                {
                    "horizontal_m": selected["horizontal_m"],
                    "vertical_m": selected["vertical_m"],
                    "npose_sample_count": selected["npose_sample_count"],
                    "walk_sample_count": selected["walk_sample_count"],
                    "npose_quality": selected["npose_quality"],
                    "sample_selection_method": comparison["selection_method"],
                    "r_mi_comparison": comparison,
                }
            )
        results[device_source] = item
    return results


def latest_sample_index(samples: list[dict], timestamps: list[float], target_s: float) -> int:
    return bisect.bisect_right(timestamps, target_s) - 1


def build_offline_grid(
    motion: dict[str, list[dict]],
    uwb_samples: list[dict],
    start_s: float,
    end_s: float,
    fps: float,
    stale_after_s: float = 0.5,
) -> dict[str, torch.Tensor]:
    frame_count = int(round((end_s - start_s) * fps))
    if frame_count <= 0:
        raise RuntimeError("action window has no output frames")
    grid_time = torch.arange(frame_count, dtype=torch.float32) / fps
    target_times = [start_s + index / fps for index in range(frame_count)]
    acceleration = torch.zeros(frame_count, len(DEVICES), 3)
    orientation = torch.zeros(frame_count, len(DEVICES), 3, 3)
    motion_time = torch.zeros(frame_count, len(DEVICES), dtype=torch.float64)
    motion_valid = torch.zeros(frame_count, len(DEVICES), dtype=torch.bool)
    uwb_distance = torch.full((frame_count, len(DEVICES)), float("nan"))
    uwb_time = torch.zeros(frame_count, len(DEVICES), dtype=torch.float64)
    uwb_valid = torch.zeros(frame_count, len(DEVICES), dtype=torch.bool)

    for device_index, source in enumerate(DEVICES):
        source_motion = motion[source]
        motion_timestamps = [sample["timestamp_s"] for sample in source_motion]
        source_uwb = [sample for sample in uwb_samples if sample["source"] == source]
        uwb_timestamps = [sample["timestamp_s"] for sample in source_uwb]
        for frame_index, target_s in enumerate(target_times):
            sample_index = latest_sample_index(source_motion, motion_timestamps, target_s)
            if sample_index >= 0:
                sample = source_motion[sample_index]
                acceleration[frame_index, device_index] = sample["aS"]
                orientation[frame_index, device_index] = sample["RIS"]
                motion_time[frame_index, device_index] = sample["timestamp_s"]
                motion_valid[frame_index, device_index] = (
                    0.0 <= target_s - sample["timestamp_s"] <= stale_after_s
                )
            uwb_index = latest_sample_index(source_uwb, uwb_timestamps, target_s)
            if uwb_index >= 0:
                sample = source_uwb[uwb_index]
                uwb_distance[frame_index, device_index] = sample["distance_m"]
                uwb_time[frame_index, device_index] = sample["timestamp_s"]
                uwb_valid[frame_index, device_index] = (
                    0.0 <= target_s - sample["timestamp_s"] <= stale_after_s
                )
    return {
        "grid_time_s": grid_time,
        "aS": acceleration,
        "RIS": orientation,
        "motion_timestamp_s": motion_time,
        "motion_valid": motion_valid,
        "uwb_distance_m": uwb_distance,
        "uwb_timestamp_s": uwb_time,
        "uwb_valid": uwb_valid,
    }


def stream_quality(samples: list[dict], start_s: float, end_s: float, valid: torch.Tensor) -> dict:
    selected = samples_between(samples, start_s, end_s)
    timestamps = [sample["timestamp_s"] for sample in selected]
    gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
    duration_s = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    return {
        "action_sample_count": len(selected),
        "estimated_rate_hz": (len(selected) - 1) / duration_s if duration_s > 0 else None,
        "max_timestamp_gap_s": max(gaps) if gaps else None,
        "valid_output_frame_count": int(valid.sum()),
        "invalid_output_frame_count": int((~valid).sum()),
        "output_frame_count": int(valid.numel()),
    }


def save_merged_raw(path: Path, events: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for event in events:
            output.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def finalize_offline_capture(args: argparse.Namespace) -> None:
    directory = args.finalize.resolve()
    capture_path = directory / CAPTURE_FILENAME
    if not capture_path.is_file():
        raise SystemExit(f"找不到采集清单：{capture_path}")
    if args.iphone_file is None or args.watch_file is None:
        raise SystemExit("--finalize 必须同时提供 --iphone-file 和 --watch-file")
    labels_path = directory / LABELS_FILENAME
    if labels_path.exists() and not args.force_finalize:
        raise SystemExit(f"{labels_path} 已存在；确认要重建时添加 --force-finalize")

    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    take = {
        "action": capture["action"],
        "watch": capture["watch"],
        "phone_pocket": capture["phone_pocket"],
        "group": int(capture["group"]),
        "phone_facing": capture["phone_facing"],
    }
    live_trace = None
    live_trace_name = capture.get("artifacts", {}).get("live_calibration")
    if live_trace_name:
        live_trace_path = directory / str(live_trace_name)
        if not live_trace_path.is_file():
            raise CalibrationError(
                f"capture.json requires the online calibration trace, but it is missing: "
                f"{live_trace_path}"
            )
        live_trace = load_live_calibration_samples(live_trace_path)
    offsets, offset_resolution = resolve_time_offsets(
        args.iphone_time_offset,
        args.watch_time_offset,
        live_trace,
    )
    print(
        "时间偏移："
        f"iPhone {offsets['iphone']:+.6f} s "
        f"({offset_resolution['iphone']['method']})，"
        f"Watch {offsets['apple_watch']:+.6f} s "
        f"({offset_resolution['apple_watch']['method']})"
    )
    events, file_report = merge_device_events(
        args.iphone_file.resolve(), args.watch_file.resolve()
    )
    motion, uwb, stream_report = extract_offline_streams(events, offsets)
    missing = [source for source in DEVICES if not motion[source]]
    if missing:
        raise RuntimeError(
            "raw files contain no complete device_motion for: " + ", ".join(missing)
        )
    calibration = calibrate_from_files(motion, capture, take, live_trace=live_trace)
    action_start, action_end = window_bounds(capture, "action")
    fps = float(capture.get("target_fps", args.fps))
    action_uwb = samples_between(uwb, action_start, action_end)
    grid = build_offline_grid(motion, action_uwb, action_start, action_end, fps)
    offline_info = {
        "iphone_file": str(args.iphone_file.resolve()),
        "watch_file": str(args.watch_file.resolve()),
        "time_offsets_s": offsets,
        "time_offset_resolution": offset_resolution,
        "merged_event_count": len(events),
        "session_ids": stream_report["session_ids"],
        "calibration_selection_method": calibration[CALIBRATION_SOURCE].get(
            "sample_selection_method"
        ),
    }
    save_sequence(
        directory,
        take,
        fps,
        calibration,
        grid,
        action_uwb,
        (action_start, action_end),
        offline_info=offline_info,
    )
    save_merged_raw(directory / "raw.ndjson", events)

    quality = {
        "schema_version": 1,
        "capture_mode": "device_local_files",
        "files": file_report,
        "streams": stream_report,
        "time_offsets_s": offsets,
        "time_offset_resolution": offset_resolution,
        "action_window": {"start_unix_s": action_start, "end_unix_s": action_end},
        "output_fps": fps,
        "output_frame_count": int(grid["aS"].shape[0]),
        "calibration_acceptance": {
            "source": CALIBRATION_SOURCE,
            "watch_uses_iphone_r_mi": True,
            "max_npose_orientation_deviation_deg": MAX_NPOSE_ORIENTATION_DEG,
            "max_npose_acceleration_rms_m_s2": MAX_NPOSE_ACCEL_RMS_M_S2,
            "min_forward_distance_m": MIN_FORWARD_DISTANCE_M,
        },
        "r_mi_comparison": calibration[CALIBRATION_SOURCE].get("r_mi_comparison"),
        "per_source": {
            source: stream_quality(
                motion[source],
                action_start,
                action_end,
                grid["motion_valid"][:, device_index],
            )
            for device_index, source in enumerate(DEVICES)
        },
    }
    write_json(directory / "quality.json", quality)
    capture["status"] = "finalized"
    capture["finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    capture["raw_files"] = {
        "iphone": str(args.iphone_file.resolve()),
        "apple_watch": str(args.watch_file.resolve()),
    }
    capture["time_offsets_s"] = offsets
    capture["time_offset_resolution"] = offset_resolution
    capture["session_ids"] = stream_report["session_ids"]
    capture["offline_calibration"] = {
        "sample_selection_method": calibration[CALIBRATION_SOURCE].get(
            "sample_selection_method"
        ),
        "r_mi_comparison": calibration[CALIBRATION_SOURCE].get("r_mi_comparison"),
    }
    capture.setdefault("artifacts", {}).update(
        {"raw": "raw.ndjson", "labels": LABELS_FILENAME, "quality": "quality.json"}
    )
    write_json(capture_path, capture)
    valid_both = int(grid["motion_valid"].all(dim=1).sum())
    print(f"离线整理完成：{directory}")
    print(f"  labels: {labels_path}")
    print(f"  帧数 {grid['aS'].shape[0]}，两设备都有效的帧 {valid_both}")
    print(
        f"  时间偏移：iPhone {offsets['iphone']:+.6f} s，"
        f"Watch {offsets['apple_watch']:+.6f} s"
    )
    for source in DEVICES:
        item = quality["per_source"][source]
        print(
            f"  {source}: 原始动作样本 {item['action_sample_count']}，"
            f"无效输出帧 {item['invalid_output_frame_count']}"
        )
    comparison = calibration[CALIBRATION_SOURCE].get("r_mi_comparison", {})
    print(f"  标定样本选择：{comparison.get('selection_method', 'unknown')}")
    exact_angle = comparison.get("online_saved_vs_offline_exact_deg")
    if exact_angle is not None:
        print(f"  在线 R_MI 与离线精确复算差异：{exact_angle:.6f} deg")


def inspect_inbox_file(path: Path) -> dict:
    counts = {source: 0 for source in DEVICES}
    session_ids = {source: set() for source in DEVICES}
    invalid_lines = 0
    with path.open("r", encoding="utf-8") as source_file:
        for line in source_file:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if not isinstance(event, dict) or event.get("sensor") != "device_motion":
                continue
            source = canonical_source(event.get("source"), "")
            if source in counts:
                counts[source] += 1
                session_id = str(event.get("sessionID", ""))
                if session_id:
                    session_ids[source].add(session_id)
    lower_name = path.name.lower()
    if not any(counts.values()):
        if "watch" in lower_name:
            counts["apple_watch"] = 1
        elif "iphone" in lower_name or "phone" in lower_name:
            counts["iphone"] = 1
    return {
        "path": path,
        "motion_counts": counts,
        "session_ids": {source: sorted(values) for source, values in session_ids.items()},
        "invalid_line_count": invalid_lines,
    }


def discover_inbox_files(inbox_dir: Path) -> tuple[Path, Path, list[dict]]:
    candidates = sorted(
        path
        for path in inbox_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".ndjson", ".jsonl"}
    )
    reports = [inspect_inbox_file(path) for path in candidates]
    iphone_only = [
        item for item in reports
        if item["motion_counts"]["iphone"] > 0
        and item["motion_counts"]["apple_watch"] == 0
    ]
    watch_only = [
        item for item in reports
        if item["motion_counts"]["apple_watch"] > 0
        and item["motion_counts"]["iphone"] == 0
    ]
    combined = [
        item for item in reports
        if item["motion_counts"]["iphone"] > 0
        and item["motion_counts"]["apple_watch"] > 0
    ]

    if len(iphone_only) == 1:
        iphone_path = iphone_only[0]["path"]
    elif not iphone_only and len(combined) == 1:
        iphone_path = combined[0]["path"]
    else:
        iphone_path = None
    if len(watch_only) == 1:
        watch_path = watch_only[0]["path"]
    elif not watch_only and len(combined) == 1:
        watch_path = combined[0]["path"]
    else:
        watch_path = None

    if iphone_path is None or watch_path is None:
        detail = "; ".join(
            f"{item['path'].name}: iphone={item['motion_counts']['iphone']}, "
            f"watch={item['motion_counts']['apple_watch']}"
            for item in reports
        ) or "no NDJSON/JSONL files"
        raise RuntimeError(
            "无法唯一识别 iPhone 和 Watch 文件。请让该收件箱只保留本条的两份文件，"
            f"或将文件名改为包含 iphone/watch。当前：{detail}"
        )
    return iphone_path, watch_path, reports


def capture_session_ids(capture: dict) -> list[str]:
    """Return session IDs recorded by live iPhone calibration."""
    found = {
        str(value)
        for value in capture.get("expected_session_ids", [])
        if str(value)
    }
    live = capture.get("live_iphone_calibration", {})
    for phase in ("npose", "forward_walk"):
        for value in live.get(phase, {}).get("session_ids", []):
            if str(value):
                found.add(str(value))
    return sorted(found)


def write_session_map(root: Path) -> Path:
    """Write the formal-sequence to Sensor Read sessionID correspondence."""
    entries = []
    if root.is_dir():
        for capture_path in sorted(root.rglob(CAPTURE_FILENAME)):
            sequence_dir = capture_path.parent
            try:
                relative = sequence_dir.relative_to(root)
            except ValueError:
                continue
            if "_trial" in relative.parts:
                continue
            try:
                capture = json.loads(capture_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            entries.append(
                {
                    "formal_sequence": relative.as_posix(),
                    "session_ids": capture_session_ids(capture),
                    "status": capture.get("status"),
                    "action": capture.get("action"),
                    "watch": capture.get("watch"),
                    "phone_pocket": capture.get("phone_pocket"),
                    "group": capture.get("group"),
                    "phone_facing": capture.get("phone_facing"),
                    "capture": str(capture_path.resolve()),
                }
            )
    output = root / "session_map.json"
    write_json(
        output,
        {
            "schema_version": 1,
            "matching_key": "sessionID",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sequence_count": len(entries),
            "sequences": entries,
        },
    )
    return output


def index_raw_session_files(raw_root: Path) -> tuple[dict[tuple[str, str], list[Path]], list[dict]]:
    """Index exported NDJSON files by (sessionID, source)."""
    if not raw_root.is_dir():
        raise RuntimeError(f"原始文件目录不存在：{raw_root}")
    candidates = sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".ndjson", ".jsonl"}
    )
    reports = [inspect_inbox_file(path) for path in candidates]
    index: dict[tuple[str, str], list[Path]] = {}
    for report in reports:
        for source in DEVICES:
            if report["motion_counts"][source] <= 0:
                continue
            for session_id in report["session_ids"][source]:
                index.setdefault((session_id, source), []).append(report["path"])
    return index, reports


def finalize_batch(args: argparse.Namespace, root: Path) -> None:
    """Match every pending formal capture to exported files by sessionID."""
    if not root.is_dir():
        raise RuntimeError(f"正式采集目录不存在：{root}")
    raw_root = args.raw_root.resolve()
    session_index, raw_reports = index_raw_session_files(raw_root)
    capture_paths = []
    for capture_path in sorted(root.rglob(CAPTURE_FILENAME)):
        relative = capture_path.parent.relative_to(root)
        if "_trial" not in relative.parts:
            capture_paths.append(capture_path)

    results = []
    for capture_path in capture_paths:
        directory = capture_path.parent
        relative = directory.relative_to(root).as_posix()
        labels_path = directory / LABELS_FILENAME
        if labels_path.is_file() and not args.force_finalize:
            results.append({"sequence": relative, "status": "already_finalized"})
            continue
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        session_ids = capture_session_ids(capture)
        if len(session_ids) != 1:
            results.append(
                {
                    "sequence": relative,
                    "status": "unmatched",
                    "reason": f"expected exactly one calibration sessionID, found {session_ids}",
                }
            )
            continue
        session_id = session_ids[0]
        iphone_candidates = session_index.get((session_id, "iphone"), [])
        watch_candidates = session_index.get((session_id, "apple_watch"), [])
        if len(iphone_candidates) != 1 or len(watch_candidates) != 1:
            results.append(
                {
                    "sequence": relative,
                    "session_id": session_id,
                    "status": "unmatched",
                    "reason": (
                        f"iPhone candidates={len(iphone_candidates)}, "
                        f"Watch candidates={len(watch_candidates)}"
                    ),
                    "iphone_candidates": [str(path.resolve()) for path in iphone_candidates],
                    "watch_candidates": [str(path.resolve()) for path in watch_candidates],
                }
            )
            continue

        iphone_path = iphone_candidates[0]
        watch_path = watch_candidates[0]
        capture["raw_files"] = {
            "iphone": str(iphone_path.resolve()),
            "apple_watch": str(watch_path.resolve()),
        }
        capture["batch_file_match"] = {
            "matched_by": "sessionID",
            "session_id": session_id,
            "raw_root": str(raw_root),
        }
        capture["status"] = "validating_calibration"
        write_json(capture_path, capture)
        finalize_args = argparse.Namespace(
            finalize=directory,
            iphone_file=iphone_path,
            watch_file=watch_path,
            force_finalize=args.force_finalize,
            iphone_time_offset=args.iphone_time_offset,
            watch_time_offset=args.watch_time_offset,
            fps=args.fps,
        )
        try:
            finalize_offline_capture(finalize_args)
        except (CalibrationError, OSError, RuntimeError, ValueError, KeyError, SystemExit) as error:
            capture = json.loads(capture_path.read_text(encoding="utf-8"))
            capture["status"] = "batch_finalize_failed"
            capture["batch_finalize_failure"] = {
                "detected_at_utc": datetime.now(timezone.utc).isoformat(),
                "reason": str(error),
            }
            write_json(capture_path, capture)
            results.append(
                {
                    "sequence": relative,
                    "session_id": session_id,
                    "status": "failed",
                    "reason": str(error),
                }
            )
            continue
        results.append(
            {
                "sequence": relative,
                "session_id": session_id,
                "status": "finalized",
                "iphone_file": str(iphone_path.resolve()),
                "watch_file": str(watch_path.resolve()),
            }
        )

    report_path = root / "batch_finalize_report.json"
    counts = {
        status: sum(item["status"] == status for item in results)
        for status in ("finalized", "already_finalized", "unmatched", "failed")
    }
    write_json(
        report_path,
        {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_root": str(raw_root),
            "scanned_raw_file_count": len(raw_reports),
            "counts": counts,
            "results": results,
        },
    )
    session_map_path = write_session_map(root)
    print(f"批量整理完成：成功 {counts['finalized']}，已完成 {counts['already_finalized']}，"
          f"未匹配 {counts['unmatched']}，失败 {counts['failed']}")
    print(f"批量报告：{report_path}")
    print(f"序列与手机会话对应表：{session_map_path}")


def receive_and_finalize(
    directory: Path,
    root: Path,
    args: argparse.Namespace,
) -> str:
    try:
        relative = directory.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(directory.name)
    capture_path = directory / CAPTURE_FILENAME
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    attempt_id = str(capture.get("attempt_id") or "legacy-attempt")
    inbox_dir = args.inbox.resolve() / relative / attempt_id
    inbox_dir.mkdir(parents=True, exist_ok=True)
    print("请通过 USB 将本条的 iPhone 和 Watch 文件复制到专属收件箱：")
    print(f"  {inbox_dir}")
    print("支持 .ndjson/.jsonl；脚本会根据 source 字段识别设备，不要求固定文件名。")
    while True:
        answer = input("复制完成后按回车立即导入；d 暂缓导入；q 保存清单并退出: ").strip().lower()
        if answer == "d":
            print(f"已暂缓；稍后可用 --finalize 处理 {directory}")
            return "deferred"
        if answer == "q":
            return "quit"
        try:
            iphone_path, watch_path, _reports = discover_inbox_files(inbox_dir)
            print(f"识别到 iPhone 文件：{iphone_path.name}")
            print(f"识别到 Watch 文件：{watch_path.name}")
            capture = json.loads(capture_path.read_text(encoding="utf-8"))
            capture["raw_files"] = {
                "iphone": str(iphone_path.resolve()),
                "apple_watch": str(watch_path.resolve()),
            }
            capture["status"] = "validating_calibration"
            write_json(capture_path, capture)
            finalize_args = argparse.Namespace(
                finalize=directory,
                iphone_file=iphone_path,
                watch_file=watch_path,
                force_finalize=args.force_finalize,
                iphone_time_offset=args.iphone_time_offset,
                watch_time_offset=args.watch_time_offset,
                fps=args.fps,
            )
            finalize_offline_capture(finalize_args)
            return "finalized"
        except CalibrationError as error:
            capture = json.loads(capture_path.read_text(encoding="utf-8"))
            capture["status"] = "calibration_failed"
            capture["calibration_failure"] = {
                "detected_at_utc": datetime.now(timezone.utc).isoformat(),
                "reason": str(error),
                "inbox": str(inbox_dir),
            }
            write_json(capture_path, capture)
            print("标定失败，本条不能进入训练数据集：")
            print(str(error))
            return "calibration_failed"
        except (OSError, RuntimeError, ValueError, KeyError, SystemExit) as error:
            print(f"导入尚未完成：{error}")
            print("请检查文件是否复制完整；修正后再次按回车，或输入 d/q。")


def take_prompt(take: dict, index: int, total: int) -> str:
    watch = WATCH_INFO[take["watch"]]
    pocket = POCKET_INFO[take["phone_pocket"]]
    facing = FACING_LABEL[take["phone_facing"]]
    return (
        f"\n[{index}/{total}] {take['action']}（{ACTION_LABEL[take['action']]}）\n"
        f"手表戴在{watch['label']}（{take['watch']}）。"
        f"手机放在{pocket['label']}（{take['phone_pocket']}）。\n"
        f"第 {take['group']}/3 组，本条朝向：{facing}（{take['phone_facing']}）。"
    )


def record_take(
    receiver: SensorReadReceiver,
    root: Path,
    take: dict,
    index: int,
    total: int,
    args: argparse.Namespace,
) -> str:
    """Return ``saved``, ``skipped``, or ``quit``."""
    directory = sequence_directory(root, take)
    if (directory / "labels.pt").is_file():
        print(f"[{index}/{total}] 已存在，跳过 {directory}")
        return "skipped"
    print(take_prompt(take, index, total), flush=True)
    if take["phone_facing"] == "screen_in":
        print("请把手机放成屏幕朝内、头朝下。不要换手腕，不要换口袋。", flush=True)
    else:
        print("请把手机翻成屏幕朝外、头仍朝下。不要换手腕，不要换口袋。", flush=True)
    answer = input("回车开始标定并录制，s 跳过，q 结束: ").strip().lower()
    if answer == "q":
        return "quit"
    if answer == "s":
        print(f"已跳过 {directory}")
        return "skipped"
    calibration = None
    while calibration is None:
        calibration = calibrate_once(receiver, args.npose, args.walk, take)
        if calibration is None:
            answer = input("标定未通过。直接回车重做，s 跳过本条，q 结束: ").strip().lower()
            if answer == "q":
                return "quit"
            if answer == "s":
                return "skipped"
    cue = ACTION_CUE[take["action"]]
    print(f"标定完成。开始做{ACTION_LABEL[take['action']]}：{cue}。录制 {args.seconds:.0f} 秒。", flush=True)
    beep(880, 200)
    record_started = time.time()
    record_started_mono = time.monotonic()
    grid = record_grid(receiver, args.seconds, args.fps)
    record_ended_mono = time.monotonic()
    record_ended = time.time()
    uwb_samples = receiver.uwb_between(record_started_mono, record_ended_mono)
    save_sequence(
        directory,
        take,
        args.fps,
        calibration,
        grid,
        uwb_samples,
        (record_started, record_ended),
    )
    valid_frames = int(grid["motion_valid"].all(dim=1).sum())
    print(f"已保存 {directory}")
    print(f"  帧数 {grid['aS'].shape[0]}，两设备都新鲜的帧 {valid_frames}")
    print(f"  uwb 原始样本 {len(uwb_samples)}")
    return "saved"


def run_self_test() -> None:
    plan = collection_plan()
    if len(plan) != 9 * 4 * 3 * 2:
        raise AssertionError(f"expected 216 takes, got {len(plan)}")
    if len({item["action"] for item in plan}) != 9:
        raise AssertionError("expected 9 actions")
    wears = {(item["watch"], item["phone_pocket"]) for item in plan}
    if wears != {("lw", "lp"), ("lw", "rp"), ("rw", "lp"), ("rw", "rp")}:
        raise AssertionError(f"wear set mismatch: {wears}")
    groups = {item["group"] for item in plan}
    facings = {item["phone_facing"] for item in plan}
    if groups != {1, 2, 3} or facings != {"screen_out", "screen_in"}:
        raise AssertionError(f"group/facing mismatch: {groups} {facings}")
    directories = {sequence_directory(Path("train_labels"), item).as_posix() for item in plan}
    if len(directories) != 216:
        raise AssertionError(f"directory collision: {len(directories)}")
    right = placement({"watch": "rw", "phone_pocket": "rp"})
    left = placement({"watch": "lw", "phone_pocket": "lp"})
    if right["apple_watch"]["model_slot"] != 1 or right["iphone"]["model_slot"] != 3:
        raise AssertionError(f"rw/rp slots mismatch: {right}")
    if left["apple_watch"]["model_slot"] != 0 or left["iphone"]["model_slot"] != 2:
        raise AssertionError(f"lw/lp slots mismatch: {left}")
    first = plan[0]
    if (first["watch"], first["phone_pocket"], first["action"], first["group"], first["phone_facing"]) != (
        "lw",
        "lp",
        "Walking",
        1,
        "screen_in",
    ):
        raise AssertionError(f"plan should start at lw_lp Walking g01 screen_in, got {first}")
    synthetic_motion = {source: [] for source in DEVICES}
    for frame in range(181):
        timestamp_s = frame / 20.0
        acceleration_i = torch.zeros(3)
        if 3.0 <= timestamp_s < 4.5:
            acceleration_i[0] = 1.0
        elif 4.5 <= timestamp_s <= 6.0:
            acceleration_i[0] = -1.0
        values = {
            "rotation_m11": 1.0,
            "rotation_m12": 0.0,
            "rotation_m13": 0.0,
            "rotation_m21": 0.0,
            "rotation_m22": 1.0,
            "rotation_m23": 0.0,
            "rotation_m31": 0.0,
            "rotation_m32": 0.0,
            "rotation_m33": 1.0,
            "user_accel_x_g": float(acceleration_i[0] / GRAVITY),
            "user_accel_y_g": 0.0,
            "user_accel_z_g": 0.0,
            "rotation_x_rad_s": 0.0,
            "rotation_y_rad_s": 0.0,
            "rotation_z_rad_s": 0.0,
        }
        acceleration_s, ris, converted_acceleration_i = motion_from_values(values)
        for source in DEVICES:
            synthetic_motion[source].append(
                {
                    "source": source,
                    "sensor": "device_motion",
                    "session_id": "self-test-session",
                    "sequence": frame,
                    "timestamp_unix_ns": int(timestamp_s * 1_000_000_000),
                    "timestamp_s": timestamp_s,
                    "raw_values": dict(values),
                    "aS": acceleration_s.clone(),
                    "RIS": ris.clone(),
                    "aI": converted_acceleration_i.clone(),
                }
            )
    synthetic_capture = {
        "windows": {
            "npose": {"start_unix_s": 1.0, "end_unix_s": 3.0},
            "forward_walk": {
                "start_unix_s": 3.0,
                "end_unix_s": 6.0,
                "requested_duration_s": 3.0,
            },
            "action": {"start_unix_s": 7.0, "end_unix_s": 8.0},
        }
    }
    calibration = calibrate_from_files(synthetic_motion, synthetic_capture, first)
    if set(calibration) != set(DEVICES):
        raise AssertionError("offline calibration did not produce both devices")
    if not torch.equal(
        calibration["apple_watch"]["R_MI"], calibration["iphone"]["R_MI"]
    ):
        raise AssertionError("Apple Watch should reuse the iPhone R_MI")
    trace_source = synthetic_motion[CALIBRATION_SOURCE]
    trace = {
        "npose": [
            {
                "phase": "npose",
                "source": CALIBRATION_SOURCE,
                "sensor": "device_motion",
                "sessionID": sample["session_id"],
                "sequenceNumber": sample["sequence"],
                "timestampUnixNs": sample["timestamp_unix_ns"],
                "receivedUnixS": sample["timestamp_s"] - 0.5,
                "values": sample["raw_values"],
            }
            for sample in trace_source
            if 1.0 <= sample["timestamp_s"] <= 3.0
        ],
        "forward_walk": [
            {
                "phase": "forward_walk",
                "source": CALIBRATION_SOURCE,
                "sensor": "device_motion",
                "sessionID": sample["session_id"],
                "sequenceNumber": sample["sequence"],
                "timestampUnixNs": sample["timestamp_unix_ns"],
                "receivedUnixS": sample["timestamp_s"] - 0.5,
                "values": sample["raw_values"],
            }
            for sample in trace_source
            if 3.0 <= sample["timestamp_s"] <= 6.0 and sample["sequence"] != 70
        ],
    }
    trace_reference = compute_iphone_calibration(
        motion_samples_from_trace(trace["npose"]),
        motion_samples_from_trace(trace["forward_walk"]),
        3.0,
        "self-test trace",
        "self-test trace",
    )
    estimated_offset = estimate_pc_time_offset(trace)
    if estimated_offset is None or abs(estimated_offset["offset_s"] + 0.5) > 1e-9:
        raise AssertionError(f"automatic time-offset estimate failed: {estimated_offset}")
    automatic_offsets, automatic_report = resolve_time_offsets(None, None, trace)
    if automatic_offsets != {"iphone": -0.5, "apple_watch": -0.5}:
        raise AssertionError(f"automatic time-offset resolution failed: {automatic_offsets}")
    if automatic_report["apple_watch"]["method"] != "inherited_from_iphone":
        raise AssertionError("Watch should inherit the iPhone automatic time offset")
    manual_offsets, _ = resolve_time_offsets(0.25, -0.75, trace)
    if manual_offsets != {"iphone": 0.25, "apple_watch": -0.75}:
        raise AssertionError(f"manual time-offset overrides failed: {manual_offsets}")
    synthetic_capture["live_iphone_calibration"] = {
        "R_MI": trace_reference["R_MI"].tolist()
    }
    exact_calibration = calibrate_from_files(
        synthetic_motion,
        synthetic_capture,
        first,
        live_trace=trace,
    )
    exact_comparison = exact_calibration[CALIBRATION_SOURCE]["r_mi_comparison"]
    if exact_comparison["selection_method"] != "explicit_event_identity":
        raise AssertionError("exact live-event calibration selection was not used")
    if exact_comparison["forward_walk_match"]["matched_local_count"] != len(
        trace["forward_walk"]
    ):
        raise AssertionError("exact live-event matching count is incorrect")
    if exact_comparison["online_saved_vs_offline_exact_deg"] > 1e-3:
        raise AssertionError("exact offline calibration did not reproduce online R_MI")
    session_test_capture = {
        "expected_session_ids": ["self-test-session", ""],
        "live_iphone_calibration": {
            "npose": {"session_ids": ["self-test-session"]},
            "forward_walk": {"session_ids": ["self-test-session"]},
        },
    }
    if capture_session_ids(session_test_capture) != ["self-test-session"]:
        raise AssertionError("capture sessionID extraction failed")
    grid = build_offline_grid(synthetic_motion, [], 7.0, 8.0, 30.0)
    if grid["aS"].shape != (30, 2, 3) or not bool(grid["motion_valid"].all()):
        raise AssertionError("offline grid self-test failed")
    orientation, acceleration = model_space_sequences(calibration, grid)
    if orientation.shape != (30, 2, 3, 3) or acceleration.shape != (30, 2, 3):
        raise AssertionError("model-space sequence shapes are incorrect")
    watch_unstable_motion = {
        source: [
            {
                **sample,
                "aS": (
                    torch.tensor([3.0, 0.0, 0.0])
                    if source == "apple_watch" and 1.0 <= sample["timestamp_s"] <= 3.0
                    else sample["aS"].clone()
                ),
            }
            for sample in samples
        ]
        for source, samples in synthetic_motion.items()
    }
    calibrate_from_files(watch_unstable_motion, synthetic_capture, first)
    unstable_motion = {
        source: [
            {
                **sample,
                "aS": (
                    torch.tensor([3.0, 0.0, 0.0])
                    if 1.0 <= sample["timestamp_s"] <= 3.0
                    else sample["aS"].clone()
                ),
            }
            for sample in samples
        ]
        for source, samples in synthetic_motion.items()
    }
    try:
        calibrate_from_files(unstable_motion, synthetic_capture, first)
    except CalibrationError:
        pass
    else:
        raise AssertionError("unstable N-pose should fail offline calibration")
    print("self-test passed")
    print(f"  {len(plan)} takes, first {sequence_directory(Path('train_labels'), first)}")
    print(
        "  iPhone-only shared R_MI, exact live-sample replay, N-pose failure "
        "detection, and model-space output passed"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "采集 216 条设备本地训练序列，并从 iPhone/Watch NDJSON 离线生成 labels.pt"
        )
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="监听 iPhone 标定 UDP；--legacy-live 也使用该地址",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="监听 iPhone 标定 UDP；--legacy-live 也使用该端口",
    )
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--npose", type=float, default=2.0)
    parser.add_argument("--walk", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--inbox",
        type=Path,
        default=DEFAULT_INBOX_DIR,
        help="每条采集后接收 iPhone/Watch 文件的电脑目录",
    )
    parser.add_argument(
        "--defer-import",
        action="store_true",
        help="兼容参数：采集后只登记 sessionID；正式采集现在默认如此",
    )
    parser.add_argument(
        "--import-each",
        action="store_true",
        help="每条结束后立即等待 USB 文件并整理（恢复旧的逐条导入流程）",
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--legacy-live",
        action="store_true",
        help="旧版双设备实时 UDP 模式；默认不再使用",
    )
    parser.add_argument("--once", action="store_true", help="只试录一条，写到 _trial/，不占用 216 条正式计划")
    parser.add_argument("--action", choices=ACTIONS, default="Walking")
    parser.add_argument("--watch", choices=WATCHES, default="lw")
    parser.add_argument("--phone-pocket", choices=POCKETS, default="lp")
    parser.add_argument("--group", type=int, choices=GROUPS, default=1)
    parser.add_argument("--phone-facing", choices=FACINGS, default="screen_in")
    parser.add_argument(
        "--finalize",
        type=Path,
        default=None,
        metavar="SEQUENCE_DIR",
        help="用两份设备本地 NDJSON 完成一个 capture.json 序列",
    )
    parser.add_argument(
        "--finalize-batch",
        action="store_true",
        help="扫描 USB 导出的全部原始文件，按 sessionID 配对并整理所有正式序列",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"--finalize-batch 扫描的原始文件根目录（默认：{DEFAULT_RAW_ROOT}）",
    )
    parser.add_argument("--iphone-file", type=Path, default=None)
    parser.add_argument("--watch-file", type=Path, default=None)
    parser.add_argument(
        "--iphone-time-offset",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "离线导入时加到 iPhone timestampUnixNs 的时间偏移；默认从在线标定 "
            "UDP 自动估计，显式传入 0 可禁用自动偏移"
        ),
    )
    parser.add_argument(
        "--watch-time-offset",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "离线导入时加到 Watch timestampUnixNs 的时间偏移；默认继承本条 "
            "iPhone 的自动或手动偏移，显式传入可单独覆盖"
        ),
    )
    parser.add_argument(
        "--force-finalize",
        action="store_true",
        help="允许重建已经存在的 labels.pt",
    )
    return parser.parse_args()


def run_legacy_once(args: argparse.Namespace, root: Path) -> None:
    take = {
        "action": args.action,
        "watch": args.watch,
        "phone_pocket": args.phone_pocket,
        "group": args.group,
        "phone_facing": args.phone_facing,
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = root / "_trial" / f"{stamp}_{take['action']}_{take['watch']}_{take['phone_pocket']}_g{take['group']:02d}_{take['phone_facing']}"
    print("试录一条。结果写到 _trial/，不占用 216 条正式计划。")
    print(take_prompt(take, 1, 1))
    print("标定里的向前走不是动作本身；提示音之后的 20 秒才录这个动作。")
    with SensorReadReceiver(args.host, args.port) as receiver:
        print(f"正在监听 udp://{args.host}:{args.port}")
        print("请先在手机上开始采集。")
        wait_for_streams(receiver)
        print("已收到手表和手机的 device_motion。")
        calibration = None
        while calibration is None:
            calibration = calibrate_once(receiver, args.npose, args.walk, take)
            if calibration is None:
                answer = input("标定未通过。直接回车重做，q 结束: ").strip().lower()
                if answer == "q":
                    return
        cue = ACTION_CUE[take["action"]]
        print(f"标定完成。开始做{ACTION_LABEL[take['action']]}：{cue}。录制 {args.seconds:.0f} 秒。", flush=True)
        beep(880, 200)
        record_started = time.time()
        record_started_mono = time.monotonic()
        grid = record_grid(receiver, args.seconds, args.fps)
        record_ended_mono = time.monotonic()
        record_ended = time.time()
        uwb_samples = receiver.uwb_between(record_started_mono, record_ended_mono)
        save_sequence(directory, take, args.fps, calibration, grid, uwb_samples, (record_started, record_ended))
    valid_frames = int(grid["motion_valid"].all(dim=1).sum())
    print(f"已保存 {directory}")
    print(f"  帧数 {grid['aS'].shape[0]}，两设备都新鲜的帧 {valid_frames}")
    print(f"  uwb 原始样本 {len(uwb_samples)}")


def run_offline_once(args: argparse.Namespace, root: Path) -> None:
    take = {
        "action": args.action,
        "watch": args.watch,
        "phone_pocket": args.phone_pocket,
        "group": args.group,
        "phone_facing": args.phone_facing,
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = (
        root
        / "_trial"
        / (
            f"{stamp}_{take['action']}_{take['watch']}_{take['phone_pocket']}_"
            f"g{take['group']:02d}_{take['phone_facing']}"
        )
    )
    print("离线试采一条。结果先写 capture.json，不占用 216 条正式计划。")
    record_offline_take(root, take, 1, 1, args, directory=directory)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    root = args.output or (Path(__file__).resolve().parent / "data" / "datasets" / "train_labels")
    if args.finalize is not None:
        if args.finalize_batch:
            raise SystemExit("--finalize 和 --finalize-batch 不能同时使用")
        finalize_offline_capture(args)
        return
    if args.finalize_batch:
        finalize_batch(args, root)
        return
    if args.seconds <= 0 or args.fps <= 0 or args.npose <= 0 or args.walk <= 0:
        raise SystemExit("时长和帧率必须为正")
    if not 1 <= args.port <= 65535:
        raise SystemExit("端口必须在 1 到 65535 之间")
    if args.import_each and args.defer_import:
        raise SystemExit("--import-each 和 --defer-import 不能同时使用")
    if not args.once and not args.legacy_live and not args.import_each:
        args.defer_import = True
    if args.once:
        if args.legacy_live:
            run_legacy_once(args, root)
        else:
            run_offline_once(args, root)
        return
    plan = collection_plan()
    root.mkdir(parents=True, exist_ok=True)
    (root / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"计划 {len(plan)} 条：9 个动作 × 4 种佩戴（lw/rw × lp/rp）× 3 组 × 2 种口袋朝向。")
    print("顺序：先固定一种佩戴，做完九个动作，再换手表或口袋。")
    print("每一组先屏幕朝内、头朝下，再屏幕朝外、头朝下。")
    print("标定走步不是动作本身；提示音后的 20 秒才是该动作。")
    if args.legacy_live:
        print("旧版模式：已有 labels.pt 的条目会自动跳过。")
        with SensorReadReceiver(args.host, args.port) as receiver:
            print(f"正在监听 udp://{args.host}:{args.port}")
            print("请先在手机上开始采集。")
            wait_for_streams(receiver)
            print("已收到手表和手机的 device_motion。")
            previous_wear = None
            for index, take in enumerate(plan, start=1):
                wear = (take["watch"], take["phone_pocket"])
                if wear != previous_wear:
                    watch = WATCH_INFO[take["watch"]]
                    pocket = POCKET_INFO[take["phone_pocket"]]
                    print(
                        f"\n下一阶段：手表戴在{watch['label']}（{take['watch']}），"
                        f"手机放在{pocket['label']}（{take['phone_pocket']}）。\n"
                        f"本阶段 54 条：9 个动作，每个动作 3 组，每组 2 种朝向。",
                        flush=True,
                    )
                    previous_wear = wear
                status = record_take(receiver, root, take, index, len(plan), args)
                if status == "quit":
                    print("采集结束。已保存的序列仍在输出目录中。")
                    return
    else:
        print("本地文件模式：只在标定阶段监听 iPhone UDP，不等待 Watch 实时数据。")
        print("每条都要在两台设备上开始一段新的本地采集；结束后等待 Watch 文件传到手机。")
        if args.defer_import:
            print("批量模式（正式采集默认）：每条只登记 sessionID，不逐条连接 USB。")
            print("所有动作录完后，把手机数据一次复制到电脑，再运行 --finalize-batch。")
        else:
            print(f"逐条导入模式：每条结束后等待文件并立即整理：{args.inbox.resolve()}")
        print("已有 capture.json 或 labels.pt 的条目会自动跳过，可以中断后继续。")
        session_map_path = write_session_map(root)
        previous_wear = None
        for index, take in enumerate(plan, start=1):
            wear = (take["watch"], take["phone_pocket"])
            if wear != previous_wear:
                watch = WATCH_INFO[take["watch"]]
                pocket = POCKET_INFO[take["phone_pocket"]]
                print(
                    f"\n下一阶段：手表戴在{watch['label']}（{take['watch']}），"
                    f"手机放在{pocket['label']}（{take['phone_pocket']}）。\n"
                    f"本阶段 54 条：9 个动作，每个动作 3 组，每组 2 种朝向。",
                    flush=True,
                )
                previous_wear = wear
            status = record_offline_take(root, take, index, len(plan), args)
            if status in {"saved", "quit"}:
                session_map_path = write_session_map(root)
            if status == "quit":
                print("采集结束。已保存的 capture.json 仍在输出目录中。")
                print(f"序列与手机会话对应表：{session_map_path}")
                return
        session_map_path = write_session_map(root)
        print(f"\n216 条计划已走完。采集清单在 {root}")
        print(f"序列与手机会话对应表：{session_map_path}")
        print("USB 一次复制完手机中的全部会话后，运行：")
        print(f'  python collect_train_sequence.py --finalize-batch --raw-root "{args.raw_root}"')


if __name__ == "__main__":
    main()
