"""Record 20 s evaluation labels from Sensor Read.

Acceleration and orientation use the same conversion as ``sensor_apple/sensor.py``:

    RIS_left = transpose(CMAttitude.rotationMatrix)
    aS_left  = userAcceleration_g * 9.80665
    RIS      = C @ RIS_left @ C
    aS       = C @ aS_left
    aI       = RIS @ aS

``C = diag(1, 1, -1)`` reflects Z. ``R_MI`` and ``R_SB`` follow this receiver's
walking calibration: average the still N-pose, integrate the forward step,
remove terminal-velocity drift, and take inertial up from ``APPLE_I_UP_RH``.

UWB follows SmartApple. Each ``uwb_ranging`` event carries ``values.distance_m``
in meters. Phone and Watch both emit this stream at about 6.7 Hz.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import torch

from sensor_apple.sensor import (
    APPLE_I_UP_RH,
    GRAVITY,
    NPOSE_RMB,
    _RIS_from_values,
    _left_to_right_handed,
    _project_rotation,
    _vector,
    beep,
)


DEVICES = ("apple_watch", "iphone")
WATCH_SLOT = 0
PHONE_SLOT = {"right_thigh": 3, "left_thigh": 2}
POCKET_LABEL = {"right_thigh": "右大腿口袋", "left_thigh": "左大腿口袋"}
ACTIONS = (
    "Walking",
    "Jogging",
    "Kicking",
    "Standing",
    "ArmCrossing",
    "Clapping",
    "JumpingJacks",
    "Waving",
    "Boxing",
    "Hopping",
)


def collection_plan() -> list[dict]:
    """Ten actions, two right-pocket trials then two left-pocket trials each."""
    plan = []
    for action in ACTIONS:
        for phone_pocket in ("right_thigh", "left_thigh"):
            for repeat in (1, 2):
                plan.append(
                    {
                        "action": action,
                        "phone_pocket": phone_pocket,
                        "repeat": repeat,
                    }
                )
    return plan


def model_slots(phone_pocket: str) -> dict[str, int]:
    return {"apple_watch": WATCH_SLOT, "iphone": PHONE_SLOT[phone_pocket]}


def body_slots(phone_pocket: str) -> dict[str, str]:
    return {"apple_watch": "left_wrist", "iphone": phone_pocket}


def motion_from_values(values: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return aS, RIS, aI after the live receiver's transpose and Z reflection."""
    ris_left = _RIS_from_values(values)
    acceleration_left = GRAVITY * _vector(
        values,
        ("user_accel_x_g", "user_accel_y_g", "user_accel_z_g"),
    )
    gyro_left = _vector(
        values,
        ("rotation_x_rad_s", "rotation_y_rad_s", "rotation_z_rad_s"),
    )
    ris, acceleration_s, _gyro = _left_to_right_handed(ris_left, acceleration_left, gyro_left)
    acceleration_i = ris @ acceleration_s
    return acceleration_s, ris, acceleration_i


def average_ris(samples: list[dict]) -> torch.Tensor | None:
    if not samples:
        return None
    return _project_rotation(torch.stack([sample["RIS"] for sample in samples]).mean(dim=0))


def rmi_from_displacement(displacement: torch.Tensor) -> tuple[torch.Tensor, float] | None:
    """Build R_MI with the live receiver's inertial up axis."""
    up = APPLE_I_UP_RH
    horizontal = displacement - torch.dot(displacement, up) * up
    distance_m = float(horizontal.norm())
    if distance_m < 1e-6:
        return None
    forward = horizontal / distance_m
    right = torch.linalg.cross(up, forward)
    right = right / right.norm()
    forward = torch.linalg.cross(right, up)
    rotation_mi = torch.stack((right, up, forward), dim=0)
    return rotation_mi, distance_m


class SensorReadReceiver:
    """Keep device_motion and uwb_ranging from Sensor Read UDP JSON."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._motion: dict[str, deque] = {source: deque(maxlen=8192) for source in DEVICES}
        self._uwb: dict[str, deque] = {source: deque(maxlen=4096) for source in DEVICES}
        self._latest_sequence: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="eval-label-udp", daemon=True)

    def __enter__(self):
        self._thread.start()
        if not self._ready.wait(timeout=2.0):
            raise RuntimeError("UDP receiver did not start")
        if self._error is not None:
            raise RuntimeError(f"UDP receiver failed: {self._error}") from self._error
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            sock.bind((self.host, self.port))
            sock.settimeout(0.25)
            self._ready.set()
            while not self._stop.is_set():
                try:
                    packet, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                for raw_line in packet.splitlines():
                    self._consume(raw_line)
        except OSError as error:
            if not self._stop.is_set():
                self._error = error
            self._ready.set()
        finally:
            if sock is not None:
                sock.close()

    def _consume(self, raw_line: bytes) -> None:
        try:
            event = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if not isinstance(event, dict):
            return
        source = str(event.get("source", "")).removesuffix("_simulator")
        sensor = event.get("sensor")
        values = event.get("values")
        if source not in DEVICES or not isinstance(values, dict):
            return
        if sensor not in {"device_motion", "uwb_ranging"}:
            return
        timestamp_ns = event.get("timestampUnixNs")
        if not isinstance(timestamp_ns, (int, float)) or timestamp_ns <= 0:
            return
        sequence = event.get("sequenceNumber")
        sequence_number = int(sequence) if isinstance(sequence, int) else -1
        key = (source, sensor)
        with self._lock:
            previous = self._latest_sequence.get(key)
            if previous is not None and sequence_number >= 0 and sequence_number <= previous:
                return
            if sequence_number >= 0:
                self._latest_sequence[key] = sequence_number
            received_mono = time.monotonic()
            timestamp_s = float(timestamp_ns) / 1_000_000_000.0
            if sensor == "uwb_ranging":
                distance = values.get("distance_m")
                if not isinstance(distance, (int, float)):
                    return
                self._uwb[source].append(
                    {
                        "sequence": sequence_number,
                        "timestamp_s": timestamp_s,
                        "received_mono": received_mono,
                        "distance_m": float(distance),
                    }
                )
                return
            try:
                acceleration_s, ris, acceleration_i = motion_from_values(values)
            except (KeyError, TypeError, ValueError):
                return
            self._motion[source].append(
                {
                    "sequence": sequence_number,
                    "timestamp_s": timestamp_s,
                    "received_mono": received_mono,
                    "aS": acceleration_s,
                    "RIS": ris,
                    "aI": acceleration_i,
                }
            )

    def latest_motion(self, source: str) -> dict | None:
        with self._lock:
            if not self._motion[source]:
                return None
            return dict(self._motion[source][-1])

    def motion_since(self, source: str, received_after: float) -> list[dict]:
        with self._lock:
            return [
                dict(sample)
                for sample in self._motion[source]
                if sample["received_mono"] >= received_after
            ]

    def uwb_between(self, received_after: float, received_before: float) -> list[dict]:
        found = []
        with self._lock:
            for source, samples in self._uwb.items():
                for sample in samples:
                    if received_after <= sample["received_mono"] <= received_before:
                        item = dict(sample)
                        item["source"] = source
                        found.append(item)
        found.sort(key=lambda item: (item["timestamp_s"], item["source"]))
        return found

    def latest_uwb(self, source: str) -> dict | None:
        with self._lock:
            if not self._uwb[source]:
                return None
            return dict(self._uwb[source][-1])


def wait_for_streams(receiver: SensorReadReceiver, timeout_s: float = 30.0) -> None:
    started = time.monotonic()
    last_report = 0.0
    while True:
        missing = [source for source in DEVICES if receiver.latest_motion(source) is None]
        if not missing:
            return
        now = time.monotonic()
        if now - started >= timeout_s:
            raise TimeoutError("等待设备超时: " + ", ".join(missing))
        if now - last_report >= 1.0:
            print("等待 device_motion: " + ", ".join(missing), flush=True)
            last_report = now
        time.sleep(0.02)


def integrate_walk(samples: list[dict], span_s: float) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    """Integrate aI over ``span_s`` of each device's own timestamps."""
    ordered = sorted(samples, key=lambda sample: sample["timestamp_s"])
    if len(ordered) < 2:
        return None
    begin_t = ordered[0]["timestamp_s"]
    position = torch.zeros(3)
    velocity = torch.zeros(3)
    last_t = begin_t
    for sample in ordered[1:]:
        timestamp_s = sample["timestamp_s"]
        if timestamp_s <= last_t:
            continue
        if timestamp_s - begin_t > span_s:
            break
        dt = timestamp_s - last_t
        last_t = timestamp_s
        acceleration_i = sample["aI"]
        position = position + dt * velocity + 0.5 * dt * dt * acceleration_i
        velocity = velocity + dt * acceleration_i
    duration_s = last_t - begin_t
    if duration_s < span_s * 0.8:
        return None
    return position, velocity, duration_s


def calibrate_once(
    receiver: SensorReadReceiver,
    npose_s: float,
    walk_s: float,
    phone_pocket: str,
) -> dict[str, dict] | None:
    input("站成 N-pose，面向前方，按回车。保持不动，听到提示音后再向前走。")
    print(f"保持不动 {npose_s:.0f} 秒。", flush=True)
    npose_started = time.monotonic()
    time.sleep(npose_s)
    npose_samples = {
        source: receiver.motion_since(source, npose_started) for source in DEVICES
    }
    beep()
    print("向前走。", flush=True)
    walk_started = time.monotonic()
    deadline = walk_started + walk_s + 5.0
    while time.monotonic() < deadline:
        ready = True
        for source in DEVICES:
            samples = receiver.motion_since(source, walk_started)
            if len(samples) < 2:
                ready = False
                break
            span = samples[-1]["timestamp_s"] - samples[0]["timestamp_s"]
            if span < walk_s:
                ready = False
                break
        if ready:
            break
        time.sleep(0.01)
    else:
        print("走路窗口内没有收齐两个设备的数据。")
        return None

    results = {}
    failed = False
    up = APPLE_I_UP_RH
    slots = model_slots(phone_pocket)
    names = body_slots(phone_pocket)
    for source in DEVICES:
        ris_npose = average_ris(npose_samples[source])
        walk_samples = receiver.motion_since(source, walk_started)
        integrated = integrate_walk(walk_samples, walk_s) if walk_samples else None
        item = {"source": source, "slot": names[source], "ok": False}
        if ris_npose is None or integrated is None:
            item["reason"] = f"{source} 的 N-pose 或走路样本不足"
            results[source] = item
            failed = True
            print(f"{source} 标定失败：{item['reason']}")
            continue
        position, velocity, duration_s = integrated
        corrected = position - 0.5 * duration_s * velocity
        vertical_m = float(torch.dot(corrected, up))
        built = rmi_from_displacement(corrected)
        horizontal_m = 0.0 if built is None else built[1]
        item["vertical_m"] = vertical_m
        item["horizontal_m"] = horizontal_m
        print(
            f"Calibration motion {source}: "
            f"pI={position.tolist()}, vI={velocity.tolist()}, "
            f"horizontal={horizontal_m:.4f} m",
            flush=True,
        )
        if built is None or horizontal_m < 0.02:
            item["reason"] = f"{source} 没有积出水平前进方向（水平 {horizontal_m:.3f} m）"
            results[source] = item
            failed = True
            print(f"{source} 标定失败：{item['reason']}")
            continue
        rotation_mi = built[0]
        bone = NPOSE_RMB[slots[source]]
        item["ok"] = True
        item["R_MI"] = rotation_mi
        item["R_SB"] = (rotation_mi @ ris_npose).transpose(0, 1) @ bone
        item["RIS_Npose"] = ris_npose
        results[source] = item
        print(f"{source} 标定通过：水平 {horizontal_m:.2f} m，沿向上轴 {vertical_m:.2f} m", flush=True)
    if failed:
        return None
    return results


def record_grid(receiver: SensorReadReceiver, seconds: float, fps: float) -> dict[str, torch.Tensor]:
    frame_count = int(round(seconds * fps))
    period_s = 1.0 / fps
    acceleration = torch.zeros(frame_count, len(DEVICES), 3)
    orientation = torch.zeros(frame_count, len(DEVICES), 3, 3)
    motion_time = torch.zeros(frame_count, len(DEVICES))
    motion_valid = torch.zeros(frame_count, len(DEVICES), dtype=torch.bool)
    uwb_distance = torch.full((frame_count, len(DEVICES)), float("nan"))
    uwb_time = torch.zeros(frame_count, len(DEVICES))
    uwb_valid = torch.zeros(frame_count, len(DEVICES), dtype=torch.bool)
    grid_time = torch.arange(frame_count, dtype=torch.float32) * period_s

    next_tick = time.perf_counter()
    for index in range(frame_count):
        now = time.monotonic()
        for device_index, source in enumerate(DEVICES):
            motion = receiver.latest_motion(source)
            if motion is not None:
                acceleration[index, device_index] = motion["aS"]
                orientation[index, device_index] = motion["RIS"]
                motion_time[index, device_index] = motion["timestamp_s"]
                motion_valid[index, device_index] = now - motion["received_mono"] <= 0.5
            uwb = receiver.latest_uwb(source)
            if uwb is not None:
                uwb_distance[index, device_index] = uwb["distance_m"]
                uwb_time[index, device_index] = uwb["timestamp_s"]
                uwb_valid[index, device_index] = now - uwb["received_mono"] <= 0.5
        next_tick += period_s
        delay_s = next_tick - time.perf_counter()
        if delay_s > 0:
            time.sleep(delay_s)
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


def save_uwb_plot(path: Path, samples: list[dict], record_start_unix_s: float) -> None:
    if not samples:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 3.2))
    for source in DEVICES:
        chosen = [sample for sample in samples if sample["source"] == source]
        if not chosen:
            continue
        times = [sample["timestamp_s"] - record_start_unix_s for sample in chosen]
        distances = [sample["distance_m"] for sample in chosen]
        axis.plot(times, distances, linewidth=1.6, label=source)
    axis.set_ylim(0.0, 4.0)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("m")
    axis.set_title("Phone–Watch distance")
    axis.grid(True, alpha=0.35)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def save_sequence(
    directory: Path,
    action: str,
    phone_pocket: str,
    repeat: int,
    fps: float,
    calibration: dict[str, dict],
    grid: dict[str, torch.Tensor],
    uwb_samples: list[dict],
    record_window: tuple[float, float],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "devices": list(DEVICES),
        "slots": [body_slots(phone_pocket)[source] for source in DEVICES],
        "model_slots": [model_slots(phone_pocket)[source] for source in DEVICES],
        "phone_pocket": phone_pocket,
        "repeat": repeat,
        "fps": fps,
        "R_MI": torch.stack([calibration[source]["R_MI"] for source in DEVICES]),
        "R_SB": torch.stack([calibration[source]["R_SB"] for source in DEVICES]),
        "R_MB": torch.stack([NPOSE_RMB[model_slots(phone_pocket)[source]] for source in DEVICES]),
        **grid,
    }
    if uwb_samples:
        payload["uwb_native_timestamp_s"] = torch.tensor(
            [sample["timestamp_s"] for sample in uwb_samples], dtype=torch.float64
        )
        payload["uwb_native_distance_m"] = torch.tensor(
            [sample["distance_m"] for sample in uwb_samples], dtype=torch.float32
        )
        payload["uwb_native_source"] = [sample["source"] for sample in uwb_samples]
    else:
        payload["uwb_native_timestamp_s"] = torch.zeros(0, dtype=torch.float64)
        payload["uwb_native_distance_m"] = torch.zeros(0)
        payload["uwb_native_source"] = []
    torch.save(payload, directory / "labels.pt")
    save_uwb_plot(directory / "uwb_distance.png", uwb_samples, record_window[0])
    meta = {
        "action": action,
        "phone_pocket": phone_pocket,
        "repeat": repeat,
        "watch": "left_wrist",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "devices": list(DEVICES),
        "slots": body_slots(phone_pocket),
        "model_slots": model_slots(phone_pocket),
        "fps": fps,
        "frame_count": int(grid["aS"].shape[0]),
        "record_start_unix_s": record_window[0],
        "record_end_unix_s": record_window[1],
        "convention": {
            "aS": "userAcceleration_g * 9.80665, then Z reflection C=diag(1,1,-1); sensor frame, m/s^2",
            "RIS": "transpose of rotation_m*, then C @ RIS @ C; aI = RIS @ aS",
            "inertial_up": APPLE_I_UP_RH.tolist(),
            "R_MI": "rows are model X,Y,Z in I; v_M = R_MI @ v_I; up axis is APPLE_I_UP_RH",
            "R_SB": "(R_MI @ RIS_Npose).T @ R_MB, RIS_Npose is the mean still pose",
            "uwb": "uwb_ranging.values.distance_m in meters; native samples plus 30 Hz hold of the latest sample",
        },
        "calibration": {
            source: {
                "slot": calibration[source]["slot"],
                "model_slot": model_slots(phone_pocket)[source],
                "horizontal_m": calibration[source]["horizontal_m"],
                "vertical_m": calibration[source]["vertical_m"],
                "R_MI": calibration[source]["R_MI"].tolist(),
                "R_SB": calibration[source]["R_SB"].tolist(),
            }
            for source in DEVICES
        },
    }
    (directory / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _rotation_values(matrix: list[list[float]], accel_g: tuple[float, float, float]) -> dict:
    values = {
        "user_accel_x_g": accel_g[0],
        "user_accel_y_g": accel_g[1],
        "user_accel_z_g": accel_g[2],
        "rotation_x_rad_s": 0.0,
        "rotation_y_rad_s": 0.0,
        "rotation_z_rad_s": 0.0,
    }
    for row in range(3):
        for column in range(3):
            values[f"rotation_m{row + 1}{column + 1}"] = matrix[row][column]
    return values


def run_self_test() -> None:
    acceleration_s, ris, acceleration_i = motion_from_values(
        _rotation_values(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            (0.0, 0.0, 1.0),
        )
    )
    expected_acceleration = torch.tensor([0.0, 0.0, -GRAVITY])
    if not torch.allclose(acceleration_s, expected_acceleration, atol=1e-5):
        raise AssertionError(f"Z reflection mismatch: {acceleration_s}")
    if not torch.allclose(ris, torch.eye(3), atol=1e-5):
        raise AssertionError(f"identity RIS mismatch: {ris}")
    if not torch.allclose(acceleration_i, acceleration_s, atol=1e-5):
        raise AssertionError("aI = RIS @ aS failed for identity")

    reference_to_sensor = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    _, transposed, _ = motion_from_values(_rotation_values(reference_to_sensor, (1.0, 0.0, 0.0)))
    expected_ris = torch.tensor(reference_to_sensor, dtype=torch.float32).transpose(0, 1)
    if not torch.allclose(transposed, expected_ris, atol=1e-5):
        raise AssertionError(f"RIS transpose mismatch: {transposed}")

    built = rmi_from_displacement(torch.tensor([1.0, 0.0, 0.2]))
    if built is None:
        raise AssertionError("R_MI was not built")
    rotation_mi, distance_m = built
    forward = rotation_mi @ torch.tensor([1.0, 0.0, 0.0])
    if not torch.allclose(forward, torch.tensor([0.0, 0.0, 1.0]), atol=1e-5):
        raise AssertionError(f"forward axis mismatch: {forward}")
    if abs(distance_m - 1.0) > 1e-5:
        raise AssertionError(f"horizontal distance mismatch: {distance_m}")
    if not torch.equal(APPLE_I_UP_RH, torch.tensor([0.0, 0.0, 1.0])):
        raise AssertionError(f"unexpected inertial up: {APPLE_I_UP_RH}")
    plan = collection_plan()
    if len(plan) != 40 or len({item["action"] for item in plan}) != 10:
        raise AssertionError(f"expected 10 actions and 40 takes, got {len(plan)}")
    if model_slots("right_thigh")["iphone"] != 3 or model_slots("left_thigh")["iphone"] != 2:
        raise AssertionError(f"phone slots mismatch: {PHONE_SLOT}")
    if model_slots("left_thigh")["apple_watch"] != 0:
        raise AssertionError("watch must stay on the left wrist")
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="录制 20 秒测评标签：aS、RIS、UWB 距离、R_MI、R_SB")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--npose", type=float, default=2.0)
    parser.add_argument("--walk", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--once",
        action="store_true",
        help="只试录一条，写到 _trial/，不占用 40 条正式计划",
    )
    parser.add_argument("--action", choices=ACTIONS, default="Walking")
    parser.add_argument(
        "--phone-pocket",
        choices=tuple(PHONE_SLOT),
        default="right_thigh",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=None,
        metavar="SEQUENCE_DIR",
        help="画出一条已保存序列的加速度、朝向和 UWB 距离",
    )
    return parser.parse_args()


def sequence_directory(root: Path, take: dict) -> Path:
    return root / take["action"] / f"{take['phone_pocket']}_{take['repeat']:02d}"


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
    action = take["action"]
    phone_pocket = take["phone_pocket"]
    repeat = take["repeat"]
    print(
        f"\n[{index}/{total}] {action}\n"
        f"手表戴在左手。手机放在{POCKET_LABEL[phone_pocket]}。"
        f"这是该摆放的第 {repeat}/2 次。",
        flush=True,
    )
    answer = input("回车开始标定并录制，s 跳过，q 结束: ").strip().lower()
    if answer == "q":
        return "quit"
    if answer == "s":
        print(f"已跳过 {action} / {phone_pocket} / {repeat}")
        return "skipped"
    calibration = None
    while calibration is None:
        calibration = calibrate_once(receiver, args.npose, args.walk, phone_pocket)
        if calibration is None:
            answer = input("标定未通过。直接回车重做，s 跳过本条，q 结束: ").strip().lower()
            if answer == "q":
                return "quit"
            if answer == "s":
                return "skipped"
    print(f"标定完成。开始做 {action}，录制 {args.seconds:.0f} 秒。", flush=True)
    beep(880, 200)
    record_started = time.time()
    record_started_mono = time.monotonic()
    grid = record_grid(receiver, args.seconds, args.fps)
    record_ended_mono = time.monotonic()
    record_ended = time.time()
    uwb_samples = receiver.uwb_between(record_started_mono, record_ended_mono)
    save_sequence(
        directory,
        action,
        phone_pocket,
        repeat,
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


def _euler_degrees(orientations: torch.Tensor) -> torch.Tensor:
    from articulate.math.angular import rotation_matrix_to_euler_angle

    return torch.rad2deg(rotation_matrix_to_euler_angle(orientations))


def preview_sequence(directory: Path) -> None:
    labels_path = directory / "labels.pt" if directory.is_dir() else directory
    if not labels_path.is_file():
        raise SystemExit(f"找不到 {labels_path}")
    sequence_dir = labels_path.parent
    payload = torch.load(labels_path, map_location="cpu", weights_only=False)
    time_s = payload["grid_time_s"].numpy()
    acceleration = payload["aS"].numpy()
    devices = list(payload["devices"])
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axis_names = ("x", "y", "z")
    for device_index, name in enumerate(devices):
        axis = axes[0, device_index]
        for channel, label in enumerate(axis_names):
            axis.plot(time_s, acceleration[:, device_index, channel], label=label, linewidth=1.2)
        axis.set_title(f"{name} aS")
        axis.set_ylabel("m/s²")
        axis.grid(True, alpha=0.35)
        axis.legend(frameon=False, ncol=3)
        euler = _euler_degrees(payload["RIS"][:, device_index]).numpy()
        orientation_axis = axes[1, 0]
        for channel, label in enumerate(("roll", "pitch", "yaw")):
            orientation_axis.plot(
                time_s,
                euler[:, channel],
                label=f"{name} {label}",
                linewidth=1.1,
            )
    axes[1, 0].set_title("RIS as roll / pitch / yaw")
    axes[1, 0].set_ylabel("degree")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].grid(True, alpha=0.35)
    axes[1, 0].legend(frameon=False, ncol=2, fontsize=8)

    distance_axis = axes[1, 1]
    native_time = payload["uwb_native_timestamp_s"]
    native_distance = payload["uwb_native_distance_m"]
    native_source = payload["uwb_native_source"]
    if native_time.numel():
        start_s = float(native_time.min())
        for source in devices:
            chosen = [i for i, item in enumerate(native_source) if item == source]
            if not chosen:
                continue
            distance_axis.plot(
                native_time[chosen].numpy() - start_s,
                native_distance[chosen].numpy(),
                linewidth=1.4,
                label=source,
            )
    else:
        for device_index, name in enumerate(devices):
            distance_axis.plot(
                time_s,
                payload["uwb_distance_m"][:, device_index].numpy(),
                linewidth=1.4,
                label=name,
            )
    distance_axis.set_ylim(0.0, 4.0)
    distance_axis.set_title("Phone–Watch distance")
    distance_axis.set_ylabel("m")
    distance_axis.set_xlabel("Time (s)")
    distance_axis.grid(True, alpha=0.35)
    distance_axis.legend(frameon=False)
    figure.suptitle(sequence_dir.name)
    figure.tight_layout()
    figure.savefig(sequence_dir / "preview.png", dpi=120)
    valid = payload["motion_valid"]
    fresh = int(valid.all(dim=1).sum()) if valid.numel() else 0
    print(f"序列 {sequence_dir}")
    print(f"  帧数 {acceleration.shape[0]}，两设备都新鲜的帧 {fresh}")
    print(f"  UWB 原始样本 {int(native_time.numel())}")
    print(f"  预览图 {sequence_dir / 'preview.png'}")
    plt.show()


def run_once(args: argparse.Namespace, root: Path) -> None:
    action = args.action
    phone_pocket = args.phone_pocket
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = root / "_trial" / f"{stamp}_{action}_{phone_pocket}"
    print("试录一条。结果写到 _trial/，不占用 40 条正式计划。")
    print(f"动作 {action}。手表戴在左手，手机放在{POCKET_LABEL[phone_pocket]}。")
    print("标定里的向前走不是动作本身；提示音之后的 20 秒才录这个动作。")
    with SensorReadReceiver(args.host, args.port) as receiver:
        print(f"正在监听 udp://{args.host}:{args.port}")
        print("请先在手机上开始采集。")
        wait_for_streams(receiver)
        print("已收到手表和手机的 device_motion。")
        calibration = None
        while calibration is None:
            calibration = calibrate_once(receiver, args.npose, args.walk, phone_pocket)
            if calibration is None:
                answer = input("标定未通过。直接回车重做，q 结束: ").strip().lower()
                if answer == "q":
                    return
        print(f"标定完成。开始做 {action}，录制 {args.seconds:.0f} 秒。", flush=True)
        beep(880, 200)
        record_started = time.time()
        record_started_mono = time.monotonic()
        grid = record_grid(receiver, args.seconds, args.fps)
        record_ended_mono = time.monotonic()
        record_ended = time.time()
        uwb_samples = receiver.uwb_between(record_started_mono, record_ended_mono)
        save_sequence(
            directory,
            action,
            phone_pocket,
            0,
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
    print("查看这条试录：")
    print(f"  python collect_eval_sequence.py --preview {directory}")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.preview is not None:
        preview_sequence(args.preview)
        return
    if args.seconds <= 0 or args.fps <= 0 or args.npose <= 0 or args.walk <= 0:
        raise SystemExit("时长和帧率必须为正")
    if not 1 <= args.port <= 65535:
        raise SystemExit("端口必须在 1 到 65535 之间")
    root = args.output or (Path(__file__).resolve().parent / "data" / "datasets" / "eval_labels")
    if args.once:
        run_once(args, root)
        return
    plan = collection_plan()
    root.mkdir(parents=True, exist_ok=True)
    (root / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"计划 {len(plan)} 条：{len(ACTIONS)} 个动作，每个动作右口袋 2 次、左口袋 2 次。")
    print("手表始终戴在左手。标定走步不是动作本身；提示音后的 20 秒才是该动作。")
    print("已有 labels.pt 的条目会自动跳过，可以中断后继续。")
    with SensorReadReceiver(args.host, args.port) as receiver:
        print(f"正在监听 udp://{args.host}:{args.port}")
        print("请先在手机上开始采集。")
        wait_for_streams(receiver)
        print("已收到手表和手机的 device_motion。")
        previous_pocket = None
        for index, take in enumerate(plan, start=1):
            pocket = take["phone_pocket"]
            if previous_pocket != pocket:
                print(f"\n下一阶段请把手机放在{POCKET_LABEL[pocket]}。", flush=True)
                previous_pocket = pocket
            status = record_take(receiver, root, take, index, len(plan), args)
            if status == "quit":
                print("采集结束。已保存的序列仍在输出目录中。")
                return
        print(f"\n40 条计划已走完。数据在 {root}")


if __name__ == "__main__":
    main()
