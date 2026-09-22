"""Benchmark the rolling-window Plain Transformer used by livedemo_apple.py."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from config import model_config
from livedemo_apple import apply_combo_calibrator, load_combo_calibrator, make_mocap_input
from utils.model_utils import load_model


BASE_DIR = Path(__file__).resolve().parent


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BASE_DIR / "data/checkpoints/chi2027_calibrator_plain/best.pt",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument(
        "--with-mobileposer",
        action="store_true",
        help="also benchmark calibrator plus one frozen MobilePoser forward_frame call",
    )
    parser.add_argument(
        "--mobileposer",
        type=Path,
        default=BASE_DIR / "data/checkpoints/base_model_12combo.pth",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model_config.device = device
    calibrator = load_combo_calibrator(args.checkpoint, device)
    frame = torch.randn(3, 12, device=device)

    # Fill the history so the benchmark measures the steady-state 125-frame path.
    for _ in range(calibrator.max_seq_len):
        calibrator.forward_frame(frame)
    for _ in range(args.warmup):
        calibrator.forward_frame(frame)
    synchronize(device)

    elapsed_ms = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        calibrator.forward_frame(frame)
        synchronize(device)
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    elapsed_ms.sort()
    mean = statistics.fmean(elapsed_ms)
    median = statistics.median(elapsed_ms)
    p95 = elapsed_ms[int(0.95 * (len(elapsed_ms) - 1))]
    parameters = sum(parameter.numel() for parameter in calibrator.model.parameters())
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"parameters={parameters}")
    print(f"window={calibrator.max_seq_len}, stride=1")
    print(f"mean_ms={mean:.3f}, median_ms={median:.3f}, p95_ms={p95:.3f}")
    print(f"throughput_fps={1000.0 / mean:.1f}")

    if args.with_mobileposer:
        mocap = load_model(str(args.mobileposer)).to(device).eval()
        mocap.reset()
        acceleration = torch.randn(7, 3, device=device)
        orientation = torch.eye(3, device=device).repeat(7, 1, 1)

        def step():
            calibrated_acceleration, calibrated_orientation = apply_combo_calibrator(
                calibrator, acceleration, orientation
            )
            mocap_input = make_mocap_input(
                calibrated_acceleration, calibrated_orientation, device
            )
            mocap.forward_frame(mocap_input)

        for _ in range(args.warmup):
            step()
        synchronize(device)
        elapsed_ms = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            step()
            synchronize(device)
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)
        elapsed_ms.sort()
        mean = statistics.fmean(elapsed_ms)
        median = statistics.median(elapsed_ms)
        p95 = elapsed_ms[int(0.95 * (len(elapsed_ms) - 1))]
        print("pipeline=plain_calibrator+mobileposer")
        print(f"pipeline_mean_ms={mean:.3f}, pipeline_median_ms={median:.3f}, pipeline_p95_ms={p95:.3f}")
        print(f"pipeline_throughput_fps={1000.0 / mean:.1f}")


if __name__ == "__main__":
    main()
