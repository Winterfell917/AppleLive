"""Clock, resampling, UWB provenance and full split-NDJSON CLI regressions."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mobileposer"))
from sensor_apple.alignment import (Clock, Stream, brackets, build_aligned, calibrated_arrays,
                                    clock_models, grid_motions, hold_uwb, load_auxiliary,
                                    lowpass, rotation_resample, vector_resample)
from sensor_apple.offline import C, GRAVITY, CalibrationError, Config, auto_calibrate
from apple_offline_calibration import main
from test_apple_offline_calibration import synthetic


def write_split_fixture(folder, stationary=False):
    motions, slots, _ = synthetic()
    if stationary:
        motions = {s: replace(m, aS=m.aS * 0) for s, m in motions.items()}
    for s, m in motions.items():
        rows = []
        for i in range(len(m.unix_ns)):
            matrix = (C @ m.RIS[i] @ C).T
            a, g = C @ m.aS[i] / GRAVITY, -C @ m.gyroS[i]
            v = {f"rotation_m{r+1}{c+1}": matrix[r, c] for r in range(3) for c in range(3)}
            v.update({f"user_accel_{k}_g": a[j] for j, k in enumerate("xyz")})
            v.update({f"rotation_{k}_rad_s": g[j] for j, k in enumerate("xyz")})
            e = dict(source=s, sensor="device_motion", sessionID=m.session_id, timestampUnixNs=str(m.unix_ns[i]),
                     timestampMonotonicS=m.monotonic_s[i], sequenceNumber=int(m.sequence[i]), values=v)
            rows.append(e)
            if i % 10 == 0:
                rows.append({**e, "sensor": "uwb_ranging", "sequenceNumber": i // 10, "values": {"distance_m": 1 + i / 10000}})
            rows.append({**e, "sensor": "accelerometer", "values": {f"{k}_g": a[j] for j, k in enumerate("xyz")}})
        (folder / f"{s}.ndjson").write_text("\n".join(json.dumps(e) for e in rows) + "\n")
    return motions, slots


class AlignmentTests(unittest.TestCase):
    def test_brackets_exact_samples_survive_gaps_and_no_extrapolation(self):
        t = np.array([0., .1, 1., 1.1])
        target = np.array([-.1, 0, .05, .1, .5, 1, 1.05, 1.1, 1.2])
        *_, valid, exact = brackets(t, target, .15)
        np.testing.assert_array_equal(valid, [False, True, True, True, False, True, True, True, False])
        self.assertTrue(exact[3])

    def test_vector_interpolation_and_invalid_values(self):
        t = np.array([0., .1, .2])
        values = np.array([[0., 0, 0], [1., 2, 3], [2., 4, 6]])
        out, valid, _, _ = vector_resample(t, values, np.array([.05, .15]), .15, 50, False)
        np.testing.assert_allclose(out, [[.5, 1, 1.5], [1.5, 3, 4.5]])
        self.assertTrue(valid.all())
        values[1] = np.nan
        _, valid, _, _ = vector_resample(t, values, np.array([.05, .15]), .15, 50, False)
        self.assertFalse(valid.any())

    def test_slerp_wraparound_and_rotation_validity(self):
        rot = Rotation.from_euler("z", [179, -179], degrees=True).as_matrix()
        r, valid = rotation_resample(np.array([0., .1]), rot, np.array([.05, .2]), .15)
        self.assertTrue(valid[0])
        self.assertFalse(valid[1])
        self.assertAlmostEqual(abs(Rotation.from_matrix(r[0]).as_rotvec()[2]), np.pi)
        np.testing.assert_allclose(r[0] @ r[0].T, np.eye(3), atol=1e-12)

    def test_uwb_hold_is_causal_expires_and_has_provenance(self):
        origin = 1788000000000000000
        stream = Stream(origin + np.array([70_000_000, 270_000_000]), np.array([.07, .27]),
                        np.array([10, 11]), np.array([[1., np.nan, np.nan, np.nan], [2., 1, 0, 0]]))
        out = hold_uwb(stream, Clock(origin), np.arange(6) * .1, .15)
        np.testing.assert_array_equal(out["uwb_valid"], [False, True, True, True, True, False])
        np.testing.assert_array_equal(out["uwb_updated"], [False, True, False, True, False, False])
        self.assertEqual(out["uwb_measurement_unix_ns"][1], origin + 70_000_000)
        self.assertEqual(out["uwb_measurement_unix_ns"][0], -1)
        self.assertAlmostEqual(out["uwb_age_s"][2], .13)
        self.assertTrue(np.isnan(out["uwb_distance_m"][-1]))
        self.assertFalse(out["uwb_direction_valid"][1])

    def test_invalid_uwb_update_does_not_extend_previous_distance(self):
        for invalid in (np.nan, -1.):
            stream = Stream(np.array([0, 100_000_000]), np.array([0, .1]), np.array([0, 1]),
                            np.array([[1., 0, 0, 0], [invalid, 0, 0, 0]]))
            result = hold_uwb(stream, Clock(0), np.array([.05, .15]), .3)
            np.testing.assert_array_equal(result["uwb_valid"], [True, False])
            np.testing.assert_array_equal(stream.values[:, 0], np.array([1., invalid]))

    def test_clock_anchors_fit_offset_and_drift(self):
        motions, _, _ = synthetic()
        origin = int(motions["iphone"].unix_ns[0])
        points = [{"source_unix_ns": str(origin + int(t * 1e9)),
                   "reference_unix_ns": str(origin + int(round((.1 + t * 1.00002) * 1e9)))} for t in (0., 10., 100.)]
        clocks, report = clock_models(motions, anchors={"apple_watch": points})
        self.assertAlmostEqual(clocks["apple_watch"].offset_s, .1, places=8)
        self.assertAlmostEqual(clocks["apple_watch"].drift_ppm, 20., places=5)
        self.assertEqual(report["iphone"]["method"], "unix_assumed")
        with self.assertRaises(CalibrationError):
            clock_models(motions, offsets={"apple_watch": .1}, anchors={"apple_watch": points})
        with self.assertRaises(CalibrationError):
            clock_models(motions, drifts={"apple_watch": 20000})

    def test_single_anchor_does_not_invent_drift(self):
        motions, _, _ = synthetic()
        origin = int(motions["iphone"].unix_ns[0])
        clocks, _ = clock_models(motions, anchors={"apple_watch": [{"source_unix_ns": str(origin), "reference_unix_ns": str(origin + 200000000)}]})
        self.assertEqual(clocks["apple_watch"].drift_ppm, 0)
        self.assertAlmostEqual(clocks["apple_watch"].offset_s, .2)

    def test_common_grid_missing_raw_gyro_and_rsi(self):
        motions, slots, _ = synthetic()
        original_ns = motions["iphone"].unix_ns.copy()
        clocks, _ = clock_models(motions)
        x, report = build_aligned(motions, {}, slots, clocks, fps=50)
        self.assertEqual(x["RIS"].shape[1:], (2, 3, 3))
        self.assertFalse(x["raw_gyro_apple_rad_s_valid"].any())
        self.assertTrue(np.isnan(x["raw_gyro_apple_rad_s"]).all())
        self.assertTrue(x["motion_valid"].all())
        self.assertFalse(report["clock_sync_accuracy_verified"])
        np.testing.assert_allclose(x["RSI"], x["RIS"].swapaxes(-1, -2))
        np.testing.assert_array_equal(np.diff(x["timestamp_unix_ns"]), 20_000_000)
        np.testing.assert_array_equal(motions["iphone"].unix_ns, original_ns)
        cal, _, _ = auto_calibrate(grid_motions(x, motions), slots, Config())
        y = calibrated_arrays(x, cal)
        self.assertEqual(y["ori_MB"].shape, x["RIS"].shape)
        self.assertTrue(y["after_calibration"].any())
        self.assertFalse(y["after_calibration"].all())

    def test_lowpass_reduces_high_frequency_alias(self):
        t = np.arange(0, 4, .005)
        v = np.repeat(np.sin(2 * np.pi * 80 * t)[:, None], 3, axis=1)
        filtered, report = lowpass(t, v, 50, .15, True)
        self.assertLess(np.std(filtered[100:-100]), .01)
        self.assertTrue(report["enabled"])

    def test_two_device_time_offset_moves_all_modalities_consistently(self):
        motions, slots, _ = synthetic()
        m = motions["apple_watch"]
        motions["apple_watch"] = replace(m, unix_ns=m.unix_ns + 100_000_000)
        clock, _ = clock_models(motions, offsets={"apple_watch": -.1})
        stream = Stream(motions["apple_watch"].unix_ns[:1], m.monotonic_s[:1], m.sequence[:1], np.array([[1., 0, 0, 0]]))
        x, _ = build_aligned(motions, {("apple_watch", "uwb_ranging"): stream}, slots, clock)
        self.assertEqual(x["timestamp_unix_ns"][0], motions["iphone"].unix_ns[0])
        self.assertEqual(x["uwb_measurement_unix_ns"][0, 0], m.unix_ns[0] + 100_000_000)
        self.assertEqual(x["uwb_measurement_aligned_unix_ns"][0, 0], m.unix_ns[0])
        self.assertAlmostEqual(x["uwb_age_s"][0, 0], 0)

    def test_cli_end_to_end_split_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "input"
            inputs.mkdir()
            native, _ = write_split_fixture(inputs)
            output = root / "out"
            with contextlib.redirect_stdout(io.StringIO()):
                result = main([str(inputs), "--output", str(output)])
            self.assertEqual(result, 0)
            x = np.load(output / "aligned_calibrated.npz", allow_pickle=False)
            c = json.loads((output / "calibration.json").read_text())
            self.assertEqual(c["calibration_sampling"], "aligned")
            self.assertTrue(x["uwb_valid"].any())
            self.assertFalse(x["raw_gyro_apple_rad_s_valid"].any())
            self.assertEqual(x["acc_M_m_s2"].shape[:2], x["uwb_distance_m"].shape)
            archive = np.load(output / "native_modalities.npz", allow_pickle=False)
            np.testing.assert_array_equal(archive["iphone__motion__timestamp_unix_ns"], native["iphone"].unix_ns)
            selection = c["windows"][0]["sample_selection"]["iphone"]
            self.assertIn("grid_indices", selection)
            self.assertNotIn("sequence_numbers", selection)

    def test_alignment_available_even_when_calibration_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "input"
            inputs.mkdir()
            write_split_fixture(inputs, stationary=True)
            for mode in ([], ["--alignment-only"]):
                output = root / ("only" if mode else "failed")
                with contextlib.redirect_stdout(io.StringIO()):
                    status = main([str(inputs), "--output", str(output)] + mode)
                self.assertEqual(status, 0 if mode else 2)
                self.assertTrue((output / "aligned_sensors.npz").exists())
                self.assertFalse((output / "aligned_calibrated.npz").exists())
                self.assertFalse((output / "calibration.json").exists())


if __name__ == "__main__":
    unittest.main()
