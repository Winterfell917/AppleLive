"""Collect an independent cross-subject evaluation set.

This entry point reuses the complete capture and offline-finalization pipeline
from ``collect_train_sequence.py``.  The only sampling change is one repetition
for every action, wear, and phone-facing combination:

    15 actions x 4 wears x 2 facings = 120 sequences

The default participant is ``p02``.  Its files are isolated under
``data/datasets/subjects/p02/test_*`` so they cannot collide with the existing
training captures or the earlier partial p02 collection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import collect_train_sequence as train


TEST_GROUPS = (1,)
DEFAULT_SUBJECT = "p02"

# All imported capture functions read GROUPS from collect_train_sequence.
# Keeping a single group preserves the existing g01_* directory convention.
train.GROUPS = TEST_GROUPS


def evaluation_paths(args: argparse.Namespace) -> Path:
    """Resolve isolated test paths for the selected participant."""
    args.subject = args.subject or DEFAULT_SUBJECT
    base = (
        Path(__file__).resolve().parent
        / "data"
        / "datasets"
        / "subjects"
        / args.subject
    )
    args.raw_root = args.raw_root or base / "test_raw"
    args.inbox = args.inbox or base / "test_inbox"
    return args.output.resolve() if args.output is not None else base / "test_labels"


def evaluation_plan(subject: str, only_action: str | None = None) -> list[dict]:
    """Build one take per action, wear, and phone facing."""
    plan = train.collection_plan()
    if only_action is not None:
        plan = [take for take in plan if take["action"] == only_action]
    return [{**take, "subject": subject} for take in plan]


def test_take_prompt(take: dict, index: int, total: int) -> str:
    """Describe the single evaluation repetition without training-group wording."""
    watch = train.WATCH_INFO[take["watch"]]
    pocket = train.POCKET_INFO[take["phone_pocket"]]
    facing = train.FACING_LABEL[take["phone_facing"]]
    return (
        f"\n[{index}/{total}] {take['action']}（{train.ACTION_LABEL[take['action']]}）\n"
        f"手表戴在{watch['label']}（{take['watch']}）。"
        f"手机放在{pocket['label']}（{take['phone_pocket']}）。\n"
        f"本组合只录一遍，手机朝向：{facing}（{take['phone_facing']}）。"
    )


# The reused recording functions resolve this name from their source module.
train.take_prompt = test_take_prompt


def run_self_test() -> None:
    plan = evaluation_plan(DEFAULT_SUBJECT)
    expected = len(train.ACTIONS) * len(train.WATCHES) * len(train.POCKETS) * len(train.FACINGS)
    if len(plan) != expected:
        raise AssertionError(f"expected {expected} takes, got {len(plan)}")
    identities = {
        (
            take["action"],
            take["watch"],
            take["phone_pocket"],
            take["group"],
            take["phone_facing"],
        )
        for take in plan
    }
    if len(identities) != expected:
        raise AssertionError(f"duplicate take identities: {len(identities)} of {expected}")
    if {take["group"] for take in plan} != {1}:
        raise AssertionError("evaluation collection must contain only group 1")
    expected_prefix = [
        (train.ACTIONS[0], 1, "screen_in"),
        (train.ACTIONS[1], 1, "screen_in"),
    ]
    actual_prefix = [
        (take["action"], take["group"], take["phone_facing"])
        for take in plan[:2]
    ]
    if actual_prefix != expected_prefix:
        raise AssertionError(f"unexpected plan order: {actual_prefix}")
    print("self-test passed")
    print(f"  {len(plan)} unique takes: action x wear x facing, one repetition each")
    print("  capture/calibration/finalization settings are inherited from collect_train_sequence.py")


def main() -> None:
    args = train.parse_args()
    root = evaluation_paths(args)

    if args.self_test:
        run_self_test()
        return
    if args.upgrade_existing_uwb:
        if args.finalize is not None or args.finalize_batch:
            raise SystemExit("--upgrade-existing-uwb 不能和 --finalize/--finalize-batch 同时使用")
        train.upgrade_existing_uwb(args, root)
        return
    if args.finalize is not None:
        if args.finalize_batch:
            raise SystemExit("--finalize 和 --finalize-batch 不能同时使用")
        train.finalize_offline_capture(args)
        return
    if args.finalize_batch:
        train.finalize_batch(args, root)
        return

    if args.seconds <= 0 or args.fps <= 0 or args.npose <= 0 or args.walk <= 0:
        raise SystemExit("时长和帧率必须为正")
    if not 1 <= args.port <= 65535:
        raise SystemExit("端口必须在 1 到 65535 之间")
    if args.import_each and args.defer_import:
        raise SystemExit("--import-each 和 --defer-import 不能同时使用")
    if args.once and args.only_action is not None:
        raise SystemExit("--once 和 --only-action 不能同时使用；--once 使用 --action 指定单条试采")
    if not args.once and not args.legacy_live and not args.import_each:
        args.defer_import = True

    print(f"受试者编号：{args.subject}")
    print(f"测试采集及标签目录：{root.resolve()}")
    print(f"测试原始文件目录：{args.raw_root.resolve()}")

    if args.once:
        if args.legacy_live:
            train.run_legacy_once(args, root)
        else:
            train.run_offline_once(args, root)
        return

    plan = evaluation_plan(args.subject, args.only_action)
    root.mkdir(parents=True, exist_ok=True)
    (root / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.only_action is None:
        print(
            f"计划 {len(plan)} 条：{len(train.ACTIONS)} 个动作 × 4 种佩戴"
            "（lw/rw × lp/rp）× 1 遍 × 2 种手机朝向。"
        )
    else:
        print(
            f"本次仅采集动作 {args.only_action}：4 种佩戴 × 1 遍 × "
            f"2 种手机朝向，共 {len(plan)} 条。"
        )
    print("顺序：固定一种佩戴和手机朝向，依次录完全部动作，再切换手机朝向；两种朝向完成后再换佩戴。")
    print("每种佩戴先录屏幕朝内、头朝下的全部动作，再录屏幕朝外、头朝下的全部动作。")
    print("每个动作、佩戴和手机朝向组合只录一遍，目录仍使用 g01 以兼容现有数据格式。")
    print("标定走步不是动作本身；提示音后的 20 秒才是该动作。")

    if args.legacy_live:
        print("旧版模式：已有 labels.pt 的条目会自动跳过。")
        with train.SensorReadReceiver(args.host, args.port) as receiver:
            print(f"正在监听 udp://{args.host}:{args.port}")
            print("请先在手机上开始采集。")
            train.wait_for_streams(receiver)
            print("已收到手表和手机的 device_motion。")
            previous_stage = None
            for index, take in enumerate(plan, start=1):
                stage = (take["watch"], take["phone_pocket"], take["phone_facing"])
                if stage != previous_stage:
                    print_stage(plan, take, stage)
                    previous_stage = stage
                status = train.record_take(receiver, root, take, index, len(plan), args)
                if status == "quit":
                    print("采集结束。已保存的序列仍在输出目录中。")
                    return
        return

    print("本地文件模式：只在标定阶段监听 iPhone UDP，不等待 Watch 实时数据。")
    print("每条都要在两台设备上开始一段新的本地采集；结束后等待 Watch 文件传到手机。")
    if args.defer_import:
        print("批量模式（默认）：每条只登记 sessionID，不逐条连接 USB。")
        print("所有动作录完后，把手机数据一次复制到测试原始文件目录，再运行 --finalize-batch。")
    else:
        print(f"逐条导入模式：每条结束后等待文件并立即整理：{args.inbox.resolve()}")
    print("已有 capture.json 或 labels.pt 的条目会自动跳过，可以中断后继续。")

    session_map_path = train.write_session_map(root)
    previous_stage = None
    for index, take in enumerate(plan, start=1):
        stage = (take["watch"], take["phone_pocket"], take["phone_facing"])
        if stage != previous_stage:
            print_stage(plan, take, stage)
            previous_stage = stage
        status = train.record_offline_take(root, take, index, len(plan), args)
        if status in {"saved", "quit"}:
            session_map_path = train.write_session_map(root)
        if status == "quit":
            print("采集结束。已保存的 capture.json 仍在测试标签目录中。")
            print(f"序列与手机会话对应表：{session_map_path}")
            return

    session_map_path = train.write_session_map(root)
    print(f"\n{len(plan)} 条计划已走完。采集清单在 {root}")
    print(f"序列与手机会话对应表：{session_map_path}")
    print("USB 一次复制完手机中的全部会话后，运行：")
    command = f"python collect_eval_test.py --finalize-batch --subject {args.subject}"
    if args.output is not None:
        command += f' --output "{args.output}"'
    command += f' --raw-root "{args.raw_root}"'
    print(f"  {command}")


def print_stage(plan: list[dict], take: dict, stage: tuple[str, str, str]) -> None:
    """Print a prompt whenever wear or phone facing changes."""
    watch = train.WATCH_INFO[take["watch"]]
    pocket = train.POCKET_INFO[take["phone_pocket"]]
    facing = train.FACING_LABEL[take["phone_facing"]]
    stage_count = sum(
        (
            planned["watch"],
            planned["phone_pocket"],
            planned["phone_facing"],
        )
        == stage
        for planned in plan
    )
    print(
        f"\n下一阶段：手表戴在{watch['label']}（{take['watch']}），"
        f"手机放在{pocket['label']}（{take['phone_pocket']}），{facing}。\n"
        f"本阶段 {stage_count} 条，每个动作一遍。",
        flush=True,
    )


if __name__ == "__main__":
    main()
