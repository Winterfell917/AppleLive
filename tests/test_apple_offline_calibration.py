"""Synthetic protocol tests plus a direct regression against the live formula.

Run from repo root: python -m unittest discover -s tests -v
No sockets, device changes, pretrained models, or real recordings are needed.
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mobileposer"))
from sensor_apple.offline import (C, GRAVITY, NPOSE, CalibrationError, Config, Motion,
                                angle_deg, apply_calibration, auto_calibrate, calibration,
                                compare, load_motion, paired_indices, replay_reference,
                                rotation_from_forward)
from sensor_apple.sensor import AppleMocapSensor, AppleSensorFrame
from apple_offline_calibration import main


def synthetic(second_step=False):
    slots = {"apple_watch": 0, "iphone": 3}
    motions, expected = {}, {}
    for source, hz, yaw, mount in (("apple_watch", 50, -.7, [.2, .4, -.6]),
                                    ("iphone", 100, .3, [-.3, .2, 1.1])):
        t = np.arange(0, 14 if second_step else 9, 1 / hz)
        forward = np.array([np.cos(yaw), np.sin(yaw), 0])
        acceleration = np.zeros(len(t))
        for start in ([3, 8] if second_step else [3]):
            active = (t >= start) & (t < start + 2)
            acceleration[active] = np.sin(np.pi * (t[active] - start))
        ris = Rotation.from_euler("xyz", mount).as_matrix()
        aI = acceleration[:, None] * forward
        aS = aI @ ris
        # Different device boot times, same approximate wall-clock epoch.
        mono = t + (7000 if source == "iphone" else 3000)
        ns = 1_788_000_000_000_000_000 + np.round(t * 1e9).astype(np.int64)
        motions[source] = Motion(source, "session-test", ns, mono, np.arange(len(t)),
                                  np.repeat(ris[None], len(t), 0), aS, np.zeros_like(aS))
        rmi = rotation_from_forward(forward)
        expected[source] = (rmi, (rmi @ ris).T @ NPOSE[slots[source]])
    return motions, slots, expected


def live_recompute(motions, slots, reference):
    """Run the actual live calibration method, injecting samples not a socket."""
    windows = reference["windows"][:2]
    selections = [paired_indices(motions, (w["start_timestamp_s"], w["end_timestamp_s"]))[0] for w in windows]
    live = AppleMocapSensor.__new__(AppleMocapSensor)
    live.source_slots = slots
    live.RMI = torch.eye(3).repeat(7, 1, 1)
    live.RSB = torch.eye(3).repeat(7, 1, 1)
    static = {s: [torch.tensor(m.RIS[i], dtype=torch.float32) for i in selections[0][s]] for s, m in motions.items()}
    walk = []
    for n in range(len(next(iter(selections[1].values())))):
        frames = {}
        for s, m in motions.items():
            i = selections[1][s][n]
            frames[s] = AppleSensorFrame(s, m.session_id, int(m.sequence[i]), float(m.time[i]), 0.,
                                         torch.tensor(m.RIS[i], dtype=torch.float32),
                                         torch.tensor(m.aS[i], dtype=torch.float32),
                                         torch.tensor(m.gyroS[i], dtype=torch.float32))
        walk.append(frames)
    live._collect_static_RIS = lambda duration: static
    live._collect_matched_window = lambda **kw: (walk, {})
    with patch("builtins.input", return_value=""), patch("sensor_apple.sensor.beep"), contextlib.redirect_stdout(io.StringIO()):
        live.calibrate_walking_6dof()
    return live.RMI.numpy(), live.RSB.numpy()


class OfflineTests(unittest.TestCase):
    def test_auto_recovers_known_two_device_rotations(self):
        motions, slots, expected = synthetic()
        cal, quality, detection = auto_calibrate(motions, slots, Config())
        self.assertEqual(detection["accepted_count"], 1)
        for s, slot in slots.items():
            self.assertLess(angle_deg(np.array(cal["RMI"][slot]), expected[s][0]), 1e-6)
            self.assertLess(angle_deg(np.array(cal["RSB"][slot]), expected[s][1]), 1e-6)
            self.assertGreater(quality[s]["horizontal_displacement_m"], .5)
        self.assertIn("sample_selection", cal["windows"][0])

    def test_apply_keeps_native_rate_and_timestamps(self):
        motions, slots, _ = synthetic()
        cal, _, _ = auto_calibrate(motions, slots, Config())
        with tempfile.TemporaryDirectory() as tmp:
            apply_calibration(motions, cal, tmp, .15)
            for s, m in motions.items():
                x = np.load(Path(tmp) / f"{s}-calibrated.npz")
                keep = m.time >= cal["apply_from_timestamp_s"]
                np.testing.assert_array_equal(x["timestamp_unix_ns"], m.unix_ns[keep])
                np.testing.assert_allclose(x["ori_MB"], np.repeat(NPOSE[slots[s]][None], sum(keep), 0), atol=1e-10)
                self.assertFalse(x["gap_before"].any())

    def test_no_motion_fails(self):
        motions, slots, _ = synthetic()
        motions = {s: replace(m, aS=m.aS * 0) for s, m in motions.items()}
        with self.assertRaises(CalibrationError):
            auto_calibrate(motions, slots, Config())

    def test_ambiguous_two_steps_fail(self):
        motions, slots, _ = synthetic(second_step=True)
        with self.assertRaises(CalibrationError) as cm:
            auto_calibrate(motions, slots, Config())
        self.assertEqual(cm.exception.details["accepted_count"], 2)

    def test_gap_is_not_interpolated_for_integration(self):
        motions, slots, _ = synthetic()
        m = motions["iphone"]
        keep = ~((m.time - m.time[0] > 3.5) & (m.time - m.time[0] < 4))
        motions["iphone"] = replace(m, **{key: getattr(m, key)[keep] for key in
                                           ("unix_ns", "monotonic_s", "sequence", "RIS", "aS", "gyroS")})
        with self.assertRaises(CalibrationError):
            auto_calibrate(motions, slots, Config())

    def test_exact_live_formula_on_same_samples(self):
        motions, slots, _ = synthetic()
        start = motions["iphone"].time[0]
        reference = {"windows": [{"kind": "npose_static", "start_timestamp_s": start + .5, "end_timestamp_s": start + 2.5},
                                 {"kind": "forward_walk", "start_timestamp_s": start + 2.5, "end_timestamp_s": start + 5.5}]}
        cal, _, _ = replay_reference(motions, slots, reference, Config())
        rmi, rsb = live_recompute(motions, slots, reference)
        np.testing.assert_allclose(cal["RMI"], rmi, atol=2e-6)
        np.testing.assert_allclose(cal["RSB"], rsb, atol=2e-6)

    def test_reference_must_be_same_session_and_slot(self):
        motions, slots, _ = synthetic()
        cal, _, _ = auto_calibrate(motions, slots, Config())
        ref = json.loads(json.dumps(cal))
        ref["session_ids"]["iphone"] = "wrong-session"
        with self.assertRaises(CalibrationError):
            compare(cal, ref)
        ref = json.loads(json.dumps(cal))
        ref["source_slots"]["iphone"] = 2
        with self.assertRaises(CalibrationError):
            compare(cal, ref)

    def test_final_pose_change_rejected(self):
        motions, slots, _ = synthetic()
        for s, m in motions.items():
            ris = m.RIS.copy()
            ris[m.time - m.time[0] >= 5] = Rotation.from_euler("z", 60, degrees=True).as_matrix() @ ris[m.time - m.time[0] >= 5]
            motions[s] = replace(m, RIS=ris)
        with self.assertRaises(CalibrationError):
            auto_calibrate(motions, slots, Config())

    def test_short_manual_static_window_rejected(self):
        motions, slots, _ = synthetic()
        t = motions["iphone"].time[0]
        with self.assertRaisesRegex(CalibrationError, "too short"):
            calibration(motions, slots, (t + 1, t + 1.1), (t + 2.5, t + 5.5), Config(), (t + 6, t + 8))

    def test_split_ndjson_and_duplicate_handling(self):
        motions, slots, _ = synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for s, m in motions.items():
                path = Path(tmp) / f"{s}.ndjson"
                events = []
                for i in range(10):
                    matrix = (C @ m.RIS[i] @ C).T
                    a, g = C @ m.aS[i] / GRAVITY, -C @ m.gyroS[i]
                    v = {f"rotation_m{r+1}{c+1}": matrix[r, c] for r in range(3) for c in range(3)}
                    v.update({f"user_accel_{k}_g": a[j] for j, k in enumerate("xyz")})
                    v.update({f"rotation_{k}_rad_s": g[j] for j, k in enumerate("xyz")})
                    events.append(json.dumps(dict(source=s, sensor="device_motion", sessionID=m.session_id,
                                                  timestampUnixNs=str(m.unix_ns[i]), timestampMonotonicS=m.monotonic_s[i],
                                                  sequenceNumber=int(m.sequence[i]), values=v)))
                path.write_text("\n".join(events + events[:1]) + "\n")
                paths.append(path)
            result, stats = load_motion(paths, slots)
            self.assertEqual(stats["duplicate_samples"], 2)
            for s in slots:
                np.testing.assert_allclose(result[s].RIS, motions[s].RIS[:10], atol=1e-12)
                np.testing.assert_array_equal(result[s].unix_ns, motions[s].unix_ns[:10])
            bad = Path(tmp) / "bad.ndjson"
            e = json.loads(paths[0].read_text().splitlines()[0])
            e["sessionID"] = "different-session"
            bad.write_text(json.dumps(e))
            with self.assertRaises(CalibrationError):
                load_motion(paths + [bad], slots)

    def test_cli_does_not_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "calibration.json"
            marker.write_text("original")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["missing.ndjson", "--output", tmp])
            self.assertEqual(marker.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
