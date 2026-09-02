"""UDP receiver and calibration adapter for the Sensor Read Apple app.

Sensor Read sends one JSON event per UDP datagram.  This adapter consumes the
``device_motion`` streams and exposes the same model-space acceleration and
orientation convention used by MobilePoser.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Mapping, Optional

import torch


GRAVITY = 9.80665
NUM_MODEL_SLOTS = 7


def beep() -> None:
    """Play an audible calibration cue, with a terminal bell as fallback."""
    if sys.platform == "darwin":
        try:
            subprocess.Popen(
                ["/usr/bin/afplay", "/System/Library/Sounds/Ping.aiff"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return
        except OSError:
            pass
    print("\a", end="", flush=True)


# MobilePoser slot order: left wrist, right wrist, left thigh, right thigh,
# head, left foot, right foot.
DEFAULT_SOURCE_SLOTS = {
    "apple_watch": 0,
    "iphone": 3,
    "airpods": 4,
}

SOURCE_ALIASES = {
    "watch": "apple_watch",
    "applewatch": "apple_watch",
    "headphones": "airpods",
    "headphone": "airpods",
}

# Empirical Apple IMU convention used by this project is left-handed. Reflect
# Z at the receiver boundary so every downstream S/I quantity is right-handed.
APPLE_LH_TO_RH = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
APPLE_I_UP_RH = torch.tensor([0.0, 0.0, 1.0])

# Coordinate convention follows the Huawei receiver:
# For any frames X and Y, RXY maps vectors from Y into X.
# M: model/world, I: inertial, S: sensor, B: bone.
# Expected RMB while standing in the MobilePoser N-pose.
NPOSE_RMB = torch.tensor(
    [
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],  # left wrist
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],  # right wrist
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],   # left thigh
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],   # right thigh / phone
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],   # head
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],   # left foot
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],   # right foot
    ],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class AppleSensorFrame:
    source: str
    session_id: str
    sequence_number: int
    timestamp_s: float
    received_at_s: float
    RIS: torch.Tensor
    aS: torch.Tensor
    gyroS: torch.Tensor


def _canonical_source(source: str) -> str:
    source = source.removesuffix("_simulator").lower()
    return SOURCE_ALIASES.get(source, source)


def _RIS_from_values(values: Mapping[str, float]) -> torch.Tensor:
    keys = [f"rotation_m{row}{column}" for row in range(1, 4) for column in range(1, 4)]
    if not all(key in values for key in keys):
        raise ValueError("device_motion event does not contain a complete rotation matrix")

    # CMAttitude.rotationMatrix maps the reference frame into device space.
    # MobilePoser expects RIS, which maps sensor/device space into inertial
    # space, hence the transpose.
    reference_to_sensor = torch.tensor([float(values[key]) for key in keys], dtype=torch.float32).view(3, 3)
    return reference_to_sensor.transpose(0, 1).contiguous()


def _vector(values: Mapping[str, float], names: Iterable[str]) -> torch.Tensor:
    return torch.tensor([float(values.get(name, 0.0)) for name in names], dtype=torch.float32)


def _left_to_right_handed(RIS: torch.Tensor, aS: torch.Tensor, gyroS: torch.Tensor):
    """Reflect Z consistently for rotations, polar vectors, and axial vectors."""
    C = APPLE_LH_TO_RH
    RIS_right = C @ RIS @ C
    aS_right = C @ aS
    # Angular velocity is an axial vector: w' = det(C) * C * w.
    gyroS_right = -C @ gyroS
    return RIS_right, aS_right, gyroS_right


def _project_rotation(matrix: torch.Tensor) -> torch.Tensor:
    """Project an averaged 3x3 matrix back onto SO(3)."""
    u, _, vh = torch.linalg.svd(matrix)
    rotation = u @ vh
    if torch.det(rotation) < 0:
        u = u.clone()
        u[:, -1].neg_()
        rotation = u @ vh
    return rotation


def _rotation_error_degrees(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return the geodesic angle between two rotation matrices."""
    relative = actual @ expected.transpose(0, 1)
    cosine = ((torch.trace(relative) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)).item())


class AppleMocapSensor:
    """Receive Sensor Read events and produce calibrated MobilePoser tensors."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9000,
        source_slots: Optional[Mapping[str, int]] = None,
        stale_after_s: float = 0.5,
        calibration_wait_timeout_s: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.source_slots = {
            _canonical_source(source): int(slot)
            for source, slot in (source_slots or DEFAULT_SOURCE_SLOTS).items()
        }
        if not self.source_slots:
            raise ValueError("At least one Apple source must be configured")
        if any(slot < 0 or slot >= NUM_MODEL_SLOTS for slot in self.source_slots.values()):
            raise ValueError(f"Source slots must be within [0, {NUM_MODEL_SLOTS - 1}]")
        if len(set(self.source_slots.values())) != len(self.source_slots):
            raise ValueError("Each Apple source must map to a different MobilePoser slot")
        if calibration_wait_timeout_s <= 0:
            raise ValueError("Calibration arrival timeout must be positive")
        self.stale_after_s = stale_after_s
        self.calibration_wait_timeout_s = calibration_wait_timeout_s
        self._latest: Dict[str, AppleSensorFrame] = {}
        self._buffers: Dict[str, Deque[AppleSensorFrame]] = {
            source: deque(maxlen=2048) for source in self.source_slots
        }
        self._clock_offset_s: Dict[str, Optional[float]] = {
            source: None for source in self.source_slots
        }
        self._frozen_clock_offset_s: Optional[Dict[str, float]] = None
        self._recording_start_state = None
        self._recording_end_state = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._error: Optional[BaseException] = None
        self.packet_errors = 0
        self._socket: Optional[socket.socket] = None
        self._thread = threading.Thread(target=self._receive_loop, name="apple-imu-udp", daemon=True)

        self.RMI = torch.eye(3).repeat(NUM_MODEL_SLOTS, 1, 1)
        self.RSB = torch.eye(3).repeat(NUM_MODEL_SLOTS, 1, 1)
        self.calibration_method = "none"
        self.calibration_windows = []
        self._thread.start()
        if not self._ready_event.wait(timeout=2.0):
            raise RuntimeError("Apple IMU receiver did not start")
        self._raise_receiver_error()

    @property
    def required_sources(self):
        return tuple(self.source_slots)

    def _raise_receiver_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Apple IMU UDP receiver failed: {self._error}") from self._error

    def _receive_loop(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            sock.bind((self.host, self.port))
            sock.settimeout(0.25)
            self._socket = sock
            self._ready_event.set()

            while not self._stop_event.is_set():
                try:
                    packet, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                for raw_line in packet.splitlines():
                    try:
                        self._consume_packet(raw_line)
                    except (ValueError, TypeError, KeyError, json.JSONDecodeError, UnicodeDecodeError):
                        # UDP is intentionally lossy and the port may receive
                        # older schema/simulator packets. A bad datagram must
                        # not tear down an otherwise healthy live session.
                        self.packet_errors += 1
        except OSError as error:
            if not self._stop_event.is_set():
                self._error = error
            self._ready_event.set()
        finally:
            if self._socket is not None:
                self._socket.close()
                self._socket = None

    def _consume_packet(self, raw_line: bytes) -> None:
        event = json.loads(raw_line)
        source = _canonical_source(str(event.get("source", "")))
        if source == "apple_watch" and event.get("sensor") == "control_event":
            values = event.get("values", {})
            if not isinstance(values, dict) or not values.get("accepted"):
                return
            action = None
            if values.get("action_start"):
                action = "start"
            elif values.get("action_stop") and values.get("phone_was_recording"):
                action = "stop"
            if action is None:
                return
            session_id = str(event.get("sessionID", ""))
            raw_timestamp_s = float(event.get("timestampUnixNs", 0)) / 1_000_000_000.0
            with self._lock:
                effective_offsets = self._frozen_clock_offset_s or self._clock_offset_s
                offset_s = effective_offsets.get("apple_watch")
                state = {
                    "event_type": f"watch_control_{action}",
                    "source": "apple_watch",
                    "session_id": session_id,
                    "raw_timestamp_s": raw_timestamp_s,
                    "timestamp_s": raw_timestamp_s + (offset_s or 0.0),
                    "received_at_s": time.monotonic(),
                    "values": {
                        "action_start": float(values.get("action_start", 0)),
                        "action_stop": float(values.get("action_stop", 0)),
                        "accepted": float(values.get("accepted", 0)),
                        "phone_was_recording": float(
                            values.get("phone_was_recording", 0)
                        ),
                    },
                }
                if action == "start":
                    self._recording_start_state = state
                else:
                    # An accepted Watch stop is the strongest save signal and
                    # takes precedence over the redundant iPhone end marker.
                    self._recording_end_state = state
            return
        if source == "iphone" and event.get("sensor") == "recording_end":
            values = event.get("values", {})
            if not isinstance(values, dict) or not values.get("completed"):
                return
            session_id = str(event.get("sessionID", ""))
            raw_timestamp_s = float(event.get("timestampUnixNs", 0)) / 1_000_000_000.0
            with self._lock:
                previous = self._recording_end_state
                if previous is not None and previous["session_id"] == session_id:
                    return
                effective_offsets = self._frozen_clock_offset_s or self._clock_offset_s
                offset_s = effective_offsets.get("iphone")
                self._recording_end_state = {
                    "event_type": "iphone_recording_end",
                    "source": "iphone",
                    "session_id": session_id,
                    "raw_timestamp_s": raw_timestamp_s,
                    "timestamp_s": raw_timestamp_s + (offset_s or 0.0),
                    "received_at_s": time.monotonic(),
                }
            return
        expected_sensor = "head_motion" if source == "airpods" else "device_motion"
        if source not in self.source_slots or event.get("sensor") != expected_sensor:
            return
        values = event.get("values")
        if not isinstance(values, dict):
            raise ValueError("Sensor Read event has no values object")

        RIS_left = _RIS_from_values(values)
        aS_left = GRAVITY * _vector(
            values,
            ("user_accel_x_g", "user_accel_y_g", "user_accel_z_g"),
        )
        gyroS_left = _vector(
            values,
            ("rotation_x_rad_s", "rotation_y_rad_s", "rotation_z_rad_s"),
        )
        RIS, aS, gyroS = _left_to_right_handed(RIS_left, aS_left, gyroS_left)

        timestamp_s = float(event.get("timestampUnixNs", 0)) / 1_000_000_000.0
        received_at_s = time.monotonic()
        received_wall_s = time.time()
        frame = AppleSensorFrame(
            source=source,
            session_id=str(event.get("sessionID", "")),
            sequence_number=int(event.get("sequenceNumber", -1)),
            timestamp_s=timestamp_s,
            received_at_s=received_at_s,
            RIS=RIS,
            aS=aS,
            gyroS=gyroS,
        )
        with self._lock:
            previous = self._latest.get(source)
            if previous is not None and previous.session_id == frame.session_id:
                if frame.sequence_number <= previous.sequence_number:
                    return
            elif previous is not None:
                self._buffers[source].clear()
                self._clock_offset_s[source] = None

            observed_offset_s = received_wall_s - timestamp_s
            clock_offset_s = self._clock_offset_s[source]
            if clock_offset_s is None or observed_offset_s < clock_offset_s:
                self._clock_offset_s[source] = observed_offset_s
            self._latest[source] = frame
            self._buffers[source].append(frame)

    def close(self) -> None:
        self._stop_event.set()
        if self._socket is not None:
            self._socket.close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def latest_frames(self) -> Dict[str, AppleSensorFrame]:
        self._raise_receiver_error()
        with self._lock:
            return dict(self._latest)

    def recording_end_state(self):
        """Return the explicit iPhone recording-end marker, if one arrived."""
        self._raise_receiver_error()
        with self._lock:
            return (
                None
                if self._recording_end_state is None
                else dict(self._recording_end_state)
            )

    def recording_start_state(self):
        """Return the accepted Watch start-control event, if one arrived."""
        self._raise_receiver_error()
        with self._lock:
            return (
                None
                if self._recording_start_state is None
                else dict(self._recording_start_state)
            )

    def _synchronization_snapshot(self):
        now = time.monotonic()
        with self._lock:
            buffers = {source: tuple(buffer) for source, buffer in self._buffers.items()}
            offsets = dict(self._frozen_clock_offset_s or self._clock_offset_s)
            latest = dict(self._latest)
        return now, buffers, offsets, latest

    def freeze_clock_offsets(self) -> Dict[str, float]:
        """Freeze the clock mapping used by one reproducible dataset sequence."""
        with self._lock:
            missing = [source for source, value in self._clock_offset_s.items() if value is None]
            if missing:
                raise RuntimeError(f"Cannot freeze clock offsets; missing: {', '.join(missing)}")
            self._frozen_clock_offset_s = {
                source: float(value) for source, value in self._clock_offset_s.items()
            }
            return dict(self._frozen_clock_offset_s)

    @staticmethod
    def _nearest_real_frame(frames, corrected_times, target_time_s, offset_s):
        """Return one received sample nearest to target; never synthesize values."""
        right_index = bisect_left(corrected_times, target_time_s)
        if right_index == 0:
            index = 0
        elif right_index == len(frames):
            index = len(frames) - 1
        else:
            left_dt = target_time_s - corrected_times[right_index - 1]
            right_dt = corrected_times[right_index] - target_time_s
            index = right_index - 1 if left_dt <= right_dt else right_index
        frame = frames[index]
        return AppleSensorFrame(
            source=frame.source,
            session_id=frame.session_id,
            sequence_number=frame.sequence_number,
            timestamp_s=frame.timestamp_s + offset_s,
            received_at_s=frame.received_at_s,
            RIS=frame.RIS,
            aS=frame.aS,
            gyroS=frame.gyroS,
        )

    def _matched_frames(self, snapshot=None):
        """Match real samples at the slowest stream's current timestamp watermark."""
        if snapshot is None:
            snapshot = self._synchronization_snapshot()
        now, buffers, offsets, latest = snapshot

        for source in self.required_sources:
            newest = latest.get(source)
            if (
                newest is None
                or offsets.get(source) is None
                or now - newest.received_at_s > self.stale_after_s
                or not buffers[source]
            ):
                return None, {}

        # The minimum latest timestamp is a data-driven watermark. It naturally
        # waits for a delayed Watch stream without imposing a fixed delay.
        target_time_s = min(
            latest[source].timestamp_s + offsets[source]
            for source in self.required_sources
        )
        matched = {}
        for source in self.required_sources:
            offset_s = offsets.get(source)
            frames = buffers[source]
            corrected_times = [frame.timestamp_s + offset_s for frame in frames]
            frame = self._nearest_real_frame(
                frames, corrected_times, target_time_s, offset_s
            )
            if abs(frame.timestamp_s - target_time_s) > self.stale_after_s:
                return None, {}
            matched[source] = frame
        return target_time_s, matched

    def _collect_matched_window(
        self,
        start_time_s: float,
        duration_s: float,
    ):
        """Pair native samples in a physical-time window without interpolation."""
        end_time_s = start_time_s + duration_s
        wait_started_s = time.monotonic()
        deadline_s = wait_started_s + duration_s + self.calibration_wait_timeout_s
        next_status_s = wait_started_s + duration_s
        reported_late_wait = False

        def missing_detail(source, now_s):
            newest = latest.get(source)
            offset_s = offsets.get(source)
            if newest is None:
                return f"{source}(no frame received)"
            if offset_s is None:
                return f"{source}(clock offset unavailable)"
            corrected_latest_s = newest.timestamp_s + offset_s
            behind_s = max(0.0, end_time_s - corrected_latest_s)
            receive_age_s = max(0.0, now_s - newest.received_at_s)
            return (
                f"{source}(behind={behind_s:.3f}s, "
                f"last_received={receive_age_s:.3f}s ago)"
            )

        while True:
            snapshot = self._synchronization_snapshot()
            now, buffers, offsets, latest = snapshot
            missing = []
            for source in self.required_sources:
                newest = latest.get(source)
                offset_s = offsets.get(source)
                if newest is None or offset_s is None:
                    missing.append(source)
                elif newest.timestamp_s + offset_s < end_time_s:
                    missing.append(source)
            if not missing:
                if reported_late_wait:
                    print("\nDelayed calibration frames received.", flush=True)
                break
            if time.monotonic() >= deadline_s:
                if reported_late_wait:
                    print()
                raise RuntimeError(
                    "Could not collect calibration window through its end; missing: "
                    + ", ".join(missing_detail(source, now) for source in missing)
                )
            if now >= next_status_s:
                grace_left_s = max(0.0, deadline_s - now)
                print(
                    "\rWaiting for delayed calibration frames: "
                    + ", ".join(missing_detail(source, now) for source in missing)
                    + f"; grace remaining={grace_left_s:.1f}s",
                    end="", flush=True,
                )
                reported_late_wait = True
                next_status_s = now + 0.5
            time.sleep(0.01)

        native = {}
        for source in self.required_sources:
            offset_s = offsets[source]
            native[source] = [
                frame for frame in buffers[source]
                if start_time_s <= frame.timestamp_s + offset_s <= end_time_s
            ]
        empty = [source for source, frames in native.items() if not frames]
        if empty:
            raise RuntimeError(
                "No native calibration samples in the selected window for: "
                + ", ".join(empty)
            )

        # Anchor on the sparsest stream (normally Watch). Every paired value is
        # a native received sample; faster streams are selected by nearest time.
        reference_source = min(self.required_sources, key=lambda source: len(native[source]))
        samples = []
        max_pair_skew_s = 0.0
        for reference_frame in native[reference_source]:
            target_time_s = reference_frame.timestamp_s + offsets[reference_source]
            matched = {}
            for source in self.required_sources:
                frames = native[source]
                corrected_times = [frame.timestamp_s + offsets[source] for frame in frames]
                frame = self._nearest_real_frame(
                    frames, corrected_times, target_time_s, offsets[source]
                )
                matched[source] = frame
                max_pair_skew_s = max(max_pair_skew_s, abs(frame.timestamp_s - target_time_s))
            samples.append(matched)

        info = {
            "pairing": "nearest_native_sample",
            "interpolation": False,
            "late_arrival_timeout_s": self.calibration_wait_timeout_s,
            "late_arrival_wait_s": max(
                0.0, time.monotonic() - wait_started_s - duration_s
            ),
            "reference_source": reference_source,
            "source_sample_counts": {
                source: len(frames) for source, frames in native.items()
            },
            "paired_sample_count": len(samples),
            "max_pair_skew_s": max_pair_skew_s,
        }
        return samples, info

    def wait_until_ready(self, timeout_s: Optional[float] = None) -> None:
        started = time.monotonic()
        last_report = 0.0
        while True:
            frames = self.latest_frames()
            missing = [source for source in self.required_sources if source not in frames]
            if not missing:
                return
            now = time.monotonic()
            if timeout_s is not None and now - started >= timeout_s:
                raise TimeoutError(f"Timed out waiting for Apple sources: {', '.join(missing)}")
            if now - last_report >= 1.0:
                print(f"Waiting for Sensor Read streams: {', '.join(missing)}")
                last_report = now
            time.sleep(0.02)

    def _raw_tensors(self):
        _, frames = self._matched_frames()
        now = time.monotonic()
        RIS = torch.eye(3).repeat(NUM_MODEL_SLOTS, 1, 1)
        aI = torch.zeros(NUM_MODEL_SLOTS, 3)
        gyroS = torch.zeros(NUM_MODEL_SLOTS, 3)
        timestamps = torch.zeros(NUM_MODEL_SLOTS, dtype=torch.float64)
        valid = torch.zeros(NUM_MODEL_SLOTS, dtype=torch.bool)

        for source, slot in self.source_slots.items():
            frame = frames.get(source)
            if frame is None or now - frame.received_at_s > self.stale_after_s:
                continue
            RIS[slot] = frame.RIS
            aI[slot] = frame.RIS @ frame.aS
            gyroS[slot] = frame.gyroS
            timestamps[slot] = frame.timestamp_s
            valid[slot] = True
        return timestamps, aI, RIS, gyroS, valid

    def get(self):
        """Return full seven-slot model-space tensors and a validity mask.

        Returns ``timestamp, aM, RMB, gyroS, valid``.  The transforms follow
        ``aI = RIS @ aS``, ``aM = RMI @ aI``, and
        ``RMB = RMI @ RIS @ RSB``.
        """
        timestamps, aI, RIS, gyroS, valid = self._raw_tensors()
        aM = (self.RMI @ aI.unsqueeze(-1)).squeeze(-1)
        RMB = self.RMI @ RIS @ self.RSB
        # PoseDataset encodes an unavailable IMU with zero acceleration and a
        # zero orientation matrix.  Do not expose the identity initialization
        # above as if it were a real, forward-facing sensor measurement.
        RMB[~valid] = 0.0
        return timestamps, aM, RMB, gyroS, valid

    def npose_orientation_errors(self, RMB: torch.Tensor, valid: torch.Tensor):
        """Report each configured slot's angular distance from calibrated N-pose."""
        errors = {}
        for source, slot in self.source_slots.items():
            errors[source] = (
                _rotation_error_degrees(RMB[slot], NPOSE_RMB[slot])
                if bool(valid[slot])
                else None
            )
        return errors

    def _collect_static_RIS(self, duration_s: float):
        RIS_samples = {source: [] for source in self.required_sources}
        start_time_s = time.time()
        matched_samples, window_info = self._collect_matched_window(
            start_time_s=start_time_s,
            duration_s=duration_s,
        )
        for frames in matched_samples:
            for source in self.required_sources:
                RIS_samples[source].append(frames[source].RIS)
        self.calibration_windows.append(
            {
                "kind": "npose_static",
                "start_timestamp_s": start_time_s,
                "end_timestamp_s": start_time_s + duration_s,
                **window_info,
            }
        )
        return RIS_samples

    def calibrate_npose(self, duration_s: float = 2.0) -> None:
        """One-pose calibration suitable for a first end-to-end integration.

        All devices must use a common CoreMotion reference frame.  This is true
        in many magnetically corrected sessions, but walking calibration is more
        robust when the devices start with different yaw references.
        """
        self.calibration_method = "npose"
        self.calibration_windows = []
        input("Stand in an N-pose, face forward, then press Enter to calibrate. ")
        RIS_samples = self._collect_static_RIS(duration_s)
        for source, slot in self.source_slots.items():
            RIS_N = _project_rotation(torch.stack(RIS_samples[source]).mean(dim=0))
            self.RMI[slot] = torch.eye(3)
            self.RSB[slot] = RIS_N.transpose(0, 1) @ NPOSE_RMB[slot]
        print("Apple IMU N-pose calibration complete.")

    def calibrate_walking_6dof(self) -> None:
        """Align CoreMotion frames using a fixed three-second forward-step window."""
        self.calibration_method = "walking_6dof"
        self.calibration_windows = []
        static_duration_s = 2.0
        walk_duration_s = 3.0
        input(
            "Stand in an N-pose facing forward, then press Enter. "
            "Hold still until the beep, then step forward. "
        )
        RIS_samples = self._collect_static_RIS(static_duration_s)
        RIS_N0 = {
            source: _project_rotation(torch.stack(RIS_samples[source]).mean(dim=0))
            for source in self.required_sources
        }
        beep()
        print("Step forward now.", flush=True)
        walk_start_time_s = time.time()
        matched_samples, window_info = self._collect_matched_window(
            start_time_s=walk_start_time_s,
            duration_s=walk_duration_s,
        )
        self.calibration_windows.append(
            {
                "kind": "forward_walk",
                "start_timestamp_s": walk_start_time_s,
                "end_timestamp_s": walk_start_time_s + walk_duration_s,
                **window_info,
            }
        )
        print("Native-frame-matched walking window captured.", flush=True)

        pI = {source: torch.zeros(3) for source in self.required_sources}
        vI = {source: torch.zeros(3) for source in self.required_sources}
        for sample_index in range(1, len(matched_samples)):
            previous_frames = matched_samples[sample_index - 1]
            frames = matched_samples[sample_index]
            for source in self.required_sources:
                frame = frames[source]
                dt = frame.timestamp_s - previous_frames[source].timestamp_s
                aI = frame.RIS @ frame.aS
                pI[source] += dt * vI[source] + 0.5 * dt * dt * aI
                vI[source] += dt * aI
        integration_duration_s = {
            source: (
                matched_samples[-1][source].timestamp_s
                - matched_samples[0][source].timestamp_s
            )
            for source in self.required_sources
        }

        # CoreMotion was +Z-up before the Z reflection, so physical up is -Z
        # in the converted right-handed I frame. Displacement supplies model +Z.
        yI = APPLE_I_UP_RH
        for source, slot in self.source_slots.items():
            # Remove the residual constant-acceleration drift that appears as
            # non-zero terminal velocity after the subject has stopped.
            zI = pI[source] - 0.5 * integration_duration_s[source] * vI[source]
            zI -= torch.dot(zI, yI) * yI
            forward_distance_m = zI.norm().item()
            print(
                f"Calibration motion {source}: "
                f"pI={pI[source].tolist()}, vI={vI[source].tolist()}, "
                f"horizontal={forward_distance_m:.4f} m",
                flush=True,
            )
            if forward_distance_m < 0.02:
                raise RuntimeError(f"Not enough forward motion to calibrate {source}")
            zI /= forward_distance_m
            xI = torch.linalg.cross(yI, zI)
            xI /= xI.norm()
            zI = torch.linalg.cross(xI, yI)

            # Rows are model basis vectors expressed in the CoreMotion frame.
            RMI = torch.stack((xI, yI, zI), dim=0)
            self.RMI[slot] = RMI
            self.RSB[slot] = (
                RMI @ RIS_N0[source]
            ).transpose(0, 1) @ NPOSE_RMB[slot]
        print("Apple IMU walking 6DoF calibration complete. Mocap starts now.")

    def calibration_state(self):
        frames = self.latest_frames()
        effective_offsets = self._frozen_clock_offset_s or self._clock_offset_s
        return {
            "method": self.calibration_method,
            "source_slots": dict(self.source_slots),
            "session_ids": {
                source: frames[source].session_id
                for source in self.required_sources
                if source in frames
            },
            "synchronization": {
                "strategy": "slowest_stream_watermark_nearest_native_sample",
                "interpolation": False,
                "fixed_delay_s": None,
            },
            "clock_offset_s": dict(effective_offsets),
            "observed_clock_offset_s": dict(self._clock_offset_s),
            "clock_offsets_frozen": self._frozen_clock_offset_s is not None,
            "windows": list(self.calibration_windows),
            "RMI": self.RMI.clone(),
            "RSB": self.RSB.clone(),
        }
