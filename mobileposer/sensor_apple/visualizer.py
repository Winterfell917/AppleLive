"""Live time-series visualization for one Apple IMU source and modality."""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Tuple

import numpy as np

from .sensor import AppleSensorFrame


MODALITIES = ("acceleration", "aI", "angular_velocity", "orientation")
MODALITY_METADATA = {
    "acceleration": (("x", "y", "z"), "Acceleration (m/s²)"),
    "aI": (("xI", "yI", "zI"), "Inertial acceleration aI (m/s²)"),
    "angular_velocity": (("x", "y", "z"), "Angular velocity (rad/s)"),
    "orientation": (("roll", "pitch", "yaw"), "Angle (deg)"),
}


def _orientation_to_euler_degrees(frame: AppleSensorFrame) -> np.ndarray:
    """Convert sensor-to-inertial rotation to ZYX roll, pitch, yaw angles."""
    rotation = frame.RIS.detach().cpu().numpy()
    horizontal = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    if horizontal > 1e-6:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        pitch = math.atan2(-float(rotation[2, 0]), horizontal)
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        pitch = math.atan2(-float(rotation[2, 0]), horizontal)
        yaw = 0.0
    return np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))


def modality_values(frame: AppleSensorFrame, modality: str) -> Tuple[np.ndarray, tuple, str]:
    """Extract display values, axis labels, and units from an Apple frame."""
    if modality == "acceleration":
        values = frame.aS.detach().cpu().numpy()
        return values, ("x", "y", "z"), "Acceleration (m/s²)"
    if modality == "aI":
        values = (frame.RIS @ frame.aS).detach().cpu().numpy()
        return values, ("xI", "yI", "zI"), "Inertial acceleration aI (m/s²)"
    if modality == "angular_velocity":
        values = frame.gyroS.detach().cpu().numpy()
        return values, ("x", "y", "z"), "Angular velocity (rad/s)"
    if modality == "orientation":
        return _orientation_to_euler_degrees(frame), ("roll", "pitch", "yaw"), "Angle (deg)"
    raise ValueError(f"Unsupported Apple IMU modality: {modality}")


class AppleIMUPlotter:
    """Interactive rolling plot for one source and one three-axis modality."""

    def __init__(
        self,
        source: str,
        modality: str,
        window_s: float = 10.0,
        refresh_hz: float = 15.0,
    ) -> None:
        if modality not in MODALITIES:
            raise ValueError(f"Unsupported modality {modality!r}; choose from {MODALITIES}")
        if window_s <= 0:
            raise ValueError("Visualization window must be positive")

        import matplotlib.pyplot as plt

        self.source = source
        self.modality = modality
        self.window_s = float(window_s)
        self.refresh_period_s = 1.0 / refresh_hz
        self._plt = plt
        self._timestamps: Deque[float] = deque()
        self._values: Deque[np.ndarray] = deque()
        self._first_timestamp_s = None
        self._last_sequence = None
        self._last_draw_s = 0.0
        self._closed = False

        labels, y_label = MODALITY_METADATA[modality]
        plt.ion()
        self.figure, self.axes = plt.subplots(num=f"Apple IMU: {source} / {modality}")
        self.lines = [self.axes.plot([], [], label=label)[0] for label in labels]
        self.axes.set_title(f"{source} · {modality}")
        self.axes.set_xlabel("Time (s)")
        self.axes.set_ylabel(y_label)
        self.axes.grid(True, alpha=0.25)
        self.axes.legend(loc="upper right")
        self.axes.set_xlim(0.0, self.window_s)
        self.figure.tight_layout()
        plt.show(block=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def update(self, frame: AppleSensorFrame) -> None:
        if self._closed or frame.sequence_number == self._last_sequence:
            return
        if not self._plt.fignum_exists(self.figure.number):
            self._closed = True
            return

        self._last_sequence = frame.sequence_number
        if self._first_timestamp_s is None:
            self._first_timestamp_s = frame.timestamp_s
        relative_time_s = frame.timestamp_s - self._first_timestamp_s
        values, _, _ = modality_values(frame, self.modality)
        self._timestamps.append(relative_time_s)
        self._values.append(np.asarray(values, dtype=np.float64))

        cutoff_s = relative_time_s - self.window_s
        while self._timestamps and self._timestamps[0] < cutoff_s:
            self._timestamps.popleft()
            self._values.popleft()

        now = time.monotonic()
        if now - self._last_draw_s < self.refresh_period_s:
            return
        self._last_draw_s = now
        self._draw(relative_time_s)

    def _draw(self, current_time_s: float) -> None:
        timestamps = np.asarray(self._timestamps, dtype=np.float64)
        values = np.stack(self._values)
        for index, line in enumerate(self.lines):
            line.set_data(timestamps, values[:, index])

        x_right = max(self.window_s, current_time_s)
        self.axes.set_xlim(max(0.0, x_right - self.window_s), x_right)
        y_min = float(np.nanmin(values))
        y_max = float(np.nanmax(values))
        span = y_max - y_min
        padding = max(0.1 * span, 0.1 * max(abs(y_min), abs(y_max)), 0.1)
        self.axes.set_ylim(y_min - padding, y_max + padding)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def close(self) -> None:
        if not self._closed:
            self._plt.close(self.figure)
            self._closed = True
