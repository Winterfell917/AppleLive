"""Collect a controlled room-orientation sensitivity retest.

Default protocol:
    subject: p02
    fixed placement: left Watch wrist + left trouser pocket
    fixed phone facing: screen inward, phone head down
    room-relative body yaw: 0, 90, 180, 270 degrees
    actions at every yaw: Boxing, Waving, Walking (control)
    calibration: the existing iPhone-only calibration is repeated per sequence

The script reuses the device-local capture, live iPhone calibration, session-ID
matching, and offline finalization implemented by ``collect_train_sequence.py``.
Its data are isolated from the existing train/test datasets.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import collect_train_sequence as train


DEFAULT_ACTIONS = ("Boxing", "Waving", "Walking")
DEFAULT_DIRECTIONS = (0, 90, 180, 270)
DEFAULT_SUBJECT = "p02"
PROTOCOL_ID = "orientation_sensitivity_retest_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=train.subject_id, default=DEFAULT_SUBJECT)
    parser.add_argument("--watch", choices=train.WATCHES, default="lw")
    parser.add_argument("--phone-pocket", choices=train.POCKETS, default="lp")
    parser.add_argument("--phone-facing", choices=train.FACINGS, default="screen_in")
    parser.add_argument(
        "--actions",
        nargs="+",
        choices=train.ACTIONS,
        default=list(DEFAULT_ACTIONS),
        help="每个房间朝向重复的动作；默认 Boxing Waving Walking",
    )
    parser.add_argument(
        "--room-directions",
        nargs="+",
        type=int,
        default=list(DEFAULT_DIRECTIONS),
        metavar="DEG",
        help="相对房间固定 0° 的身体朝向；默认 0 90 180 270",
    )
    parser.add_argument(
        "--room-zero-note",
        default="采集开始前在固定房间内人工标记的0°方向",
        help="说明房间 0° 指向的实际参照物，建议明确写墙、门或窗",
    )
    parser.add_argument(
        "--yaw-convention",
        default="从房间0°起，俯视时顺时针为正",
        help="房间朝向角的符号约定",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--inbox", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--npose", type=float, default=2.0)
    parser.add_argument("--walk", type=float, default=3.0)
    parser.add_argument(
        "--import-each",
        action="store_true",
        help="每条结束后立即导入；默认全部录完后通过 USB 一次导入",
    )
    parser.add_argument("--finalize-batch", action="store_true")
    parser.add_argument("--force-finalize", action="store_true")
    parser.add_argument("--iphone-time-offset", type=float, default=None)
    parser.add_argument("--watch-time-offset", type=float, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Path:
    subject_root = (
        Path(__file__).resolve().parent
        / "data"
        / "datasets"
        / "subjects"
        / args.subject
    )
    root = (args.output or subject_root / "orientation_retest_labels").resolve()
    args.raw_root = (
        args.raw_root or subject_root / "orientation_retest_raw"
    ).resolve()
    args.inbox = (
        args.inbox or subject_root / "orientation_retest_inbox"
    ).resolve()
    args.defer_import = not args.import_each
    return root


def normalize_directions(values: list[int]) -> list[int]:
    normalized = []
    for value in values:
        direction = value % 360
        if direction not in normalized:
            normalized.append(direction)
    if not normalized:
        raise SystemExit("至少需要一个房间朝向")
    return normalized


def make_experiment(args: argparse.Namespace, direction: int, action: str) -> dict:
    watch = train.WATCH_INFO[args.watch]
    pocket = train.POCKET_INFO[args.phone_pocket]
    return {
        "protocol_id": PROTOCOL_ID,
        "purpose": "controlled_retest_of_room_orientation_sensitivity",
        "room_body_yaw_deg": direction,
        "room_zero_note": args.room_zero_note,
        "yaw_convention": args.yaw_convention,
        "action_role": "stable_control" if action == "Walking" else "orientation_focus",
        "recalibrated_before_action": True,
        "calibration_policy": "repeat existing iPhone N-pose and forward-walk calibration for every sequence; Watch reuses iPhone R_MI",
        "fixed_conditions": {
            "watch": args.watch,
            "watch_slot": watch["slot"],
            "phone_pocket": args.phone_pocket,
            "phone_slot": pocket["slot"],
            "phone_facing": args.phone_facing,
            "phone_facing_description": train.FACING_LABEL[args.phone_facing],
            "same_physical_devices_required": True,
        },
        "timestamp_locations": {
            "capture_start_stop": "capture.json.collection",
            "npose_forward_walk_action": "capture.json.windows",
            "device_samples": "timestampUnixNs in the matched NDJSON files",
        },
    }


def build_plan(args: argparse.Namespace) -> list[dict]:
    directions = normalize_directions(args.room_directions)
    actions = list(dict.fromkeys(args.actions))
    return [
        {
            "action": action,
            "watch": args.watch,
            "phone_pocket": args.phone_pocket,
            "group": 1,
            "phone_facing": args.phone_facing,
            "subject": args.subject,
            "experiment": make_experiment(args, direction, action),
        }
        for direction in directions
        for action in actions
    ]


def sequence_directory(root: Path, take: dict) -> Path:
    direction = int(take["experiment"]["room_body_yaw_deg"])
    wear = f"{take['watch']}_{take['phone_pocket']}"
    name = f"g{take['group']:02d}_{take['phone_facing']}"
    return root / f"yaw_{direction:03d}" / take["action"] / wear / name


def retest_prompt(take: dict, index: int, total: int) -> str:
    direction = int(take["experiment"]["room_body_yaw_deg"])
    watch = train.WATCH_INFO[take["watch"]]
    pocket = train.POCKET_INFO[take["phone_pocket"]]
    facing = train.FACING_LABEL[take["phone_facing"]]
    role = "稳定对照" if take["experiment"]["action_role"] == "stable_control" else "重点动作"
    return (
        f"\n[{index}/{total}] 房间朝向 {direction}°，{take['action']}"
        f"（{train.ACTION_LABEL[take['action']]}，{role}）\n"
        f"身体面向相对房间0°的 {direction}°方向；标定走路和正式动作都保持该方向。\n"
        f"手表固定在{watch['label']}（{take['watch']}），手机固定在{pocket['label']}"
        f"（{take['phone_pocket']}），{facing}（{take['phone_facing']}）。\n"
        "本条会按原流程重新标定；不要更换设备、手腕、口袋或手机朝向。"
    )


def protocol_document(args: argparse.Namespace, plan: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "subject": args.subject,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "检验朝向敏感性在受控复测中是否稳定复现",
        "room_directions_deg": normalize_directions(args.room_directions),
        "room_zero_note": args.room_zero_note,
        "yaw_convention": args.yaw_convention,
        "actions": list(dict.fromkeys(args.actions)),
        "focus_actions": [action for action in args.actions if action in {"Boxing", "Waving"}],
        "control_actions": [action for action in args.actions if action == "Walking"],
        "fixed_conditions": plan[0]["experiment"]["fixed_conditions"],
        "calibration": {
            "recalibrated_before_every_action": True,
            "policy": plan[0]["experiment"]["calibration_policy"],
        },
        "sequence_count": len(plan),
        "directory_layout": "yaw_DDD/action/watch_pocket/g01_phone_facing",
    }


def write_protocol_once(root: Path, args: argparse.Namespace, plan: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    protocol_path = root / "protocol.json"
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise SystemExit(
                f"现有计划与本次参数不同：{plan_path}\n"
                "为避免混合实验条件，请恢复原参数，或用 --output 指定新目录。"
            )
    else:
        train.write_json(plan_path, plan)
    if not protocol_path.is_file():
        train.write_json(protocol_path, protocol_document(args, plan))


def run_self_test(args: argparse.Namespace) -> None:
    plan = build_plan(args)
    expected = len(normalize_directions(args.room_directions)) * len(
        list(dict.fromkeys(args.actions))
    )
    if len(plan) != expected:
        raise AssertionError(f"计划应为 {expected} 条，实际 {len(plan)} 条")
    identities = {
        (take["experiment"]["room_body_yaw_deg"], take["action"])
        for take in plan
    }
    if len(identities) != expected:
        raise AssertionError("计划中存在重复的房间朝向/动作组合")
    if not all(take["experiment"]["recalibrated_before_action"] for take in plan):
        raise AssertionError("每条都必须记录为重新标定")
    print(f"self-test passed：{len(plan)} 条唯一的房间朝向 × 动作组合")


def main() -> None:
    args = parse_args()
    root = resolve_paths(args)
    args.room_directions = normalize_directions(args.room_directions)
    if args.self_test:
        run_self_test(args)
        return
    if args.seconds <= 0 or args.fps <= 0 or args.npose <= 0 or args.walk <= 0:
        raise SystemExit("时长和帧率必须为正")
    if not 1 <= args.port <= 65535:
        raise SystemExit("端口必须在 1 到 65535 之间")
    if args.finalize_batch:
        train.finalize_batch(args, root)
        return

    plan = build_plan(args)
    write_protocol_once(root, args, plan)
    train.take_prompt = retest_prompt

    print(f"受试者：{args.subject}")
    print(f"协议：{PROTOCOL_ID}")
    print(f"采集清单目录：{root}")
    print(f"原始文件目录：{args.raw_root}")
    print(
        f"计划 {len(plan)} 条：{len(args.room_directions)} 个房间朝向 × "
        f"{len(list(dict.fromkeys(args.actions)))} 个动作。"
    )
    print(
        f"固定条件：{args.watch} + {args.phone_pocket} + {args.phone_facing}；"
        "同一台 iPhone 和 Apple Watch；每条按原流程重新标定。"
    )
    print(f"房间0°参照：{args.room_zero_note}")
    print(f"角度约定：{args.yaw_convention}")
    print("请先在地面或房间参照物上明确标出0°、90°、180°、270°。")
    print("标定向前走与正式动作必须面向本条指定方向。")
    if args.defer_import:
        print("默认批量模式：全部录完后把手机文件一次复制到原始文件目录，再统一整理。")
    else:
        print(f"逐条导入模式：每条结束后从 {args.inbox} 接收并整理。")

    session_map_path = train.write_session_map(root)
    previous_direction = None
    for index, take in enumerate(plan, start=1):
        direction = int(take["experiment"]["room_body_yaw_deg"])
        if direction != previous_direction:
            print(
                f"\n下一阶段：身体及标定走路方向切换到房间 {direction}°。"
                "设备佩戴和手机朝向保持不变。",
                flush=True,
            )
            previous_direction = direction
        directory = sequence_directory(root, take)
        status = train.record_offline_take(
            root, take, index, len(plan), args, directory=directory
        )
        if status in {"saved", "quit"}:
            session_map_path = train.write_session_map(root)
        if status == "quit":
            print("采集已停止；已有清单保留，可用同一命令继续。")
            print(f"序列与手机会话对应表：{session_map_path}")
            return

    session_map_path = train.write_session_map(root)
    print(f"\n{len(plan)} 条计划已走完。")
    print(f"序列与手机会话对应表：{session_map_path}")
    print("USB 复制全部手机会话到原始文件目录后运行：")
    command = (
        f"python collect_orientation_retest.py --finalize-batch --subject {args.subject} "
        f'--raw-root "{args.raw_root}"'
    )
    if args.output is not None:
        command += f' --output "{root}"'
    print(f"  {command}")


if __name__ == "__main__":
    main()
