"""Per-device calibration for all 2 Watch x 4 phone placement combinations."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mobileposer"))
from apple_offline_calibration import main
from sensor_apple.alignment import build_aligned, calibrated_arrays, clock_models
from sensor_apple.offline import (CalibrationError, Config, NPOSE, apply_calibration,
                                 auto_calibrate, compare, replay_reference)
from sensor_apple.placements import (PHONE_PLACEMENTS, WATCH_PLACEMENTS, describe_placements,
                                    resolve_placements, source_rotations)
from test_apple_alignment import write_split_fixture
from test_apple_offline_calibration import synthetic


class PlacementTests(unittest.TestCase):
    def test_all_eight_combinations_have_independent_correct_rotations(self):
        motions, _, expected = synthetic()
        clocks, _ = clock_models(motions)
        for watch in WATCH_PLACEMENTS:
            for phone in PHONE_PLACEMENTS:
                with self.subTest(watch=watch, phone=phone), tempfile.TemporaryDirectory() as tmp:
                    slots, info = resolve_placements(watch=watch, phone=phone)
                    cal, _, _ = auto_calibrate(motions, slots, Config(), info)
                    arrays, _ = build_aligned(motions, {}, slots, clocks)
                    result = calibrated_arrays(arrays, cal)
                    collision = len(set(slots.values())) != 2
                    self.assertEqual(cal["placement"]["legacy_slot_export_available"], not collision)
                    self.assertEqual(cal["RMI"] is None, collision)
                    self.assertEqual(cal["RSB"] is None, collision)
                    # Same-slot devices still have distinct mounting/reference matrices.
                    self.assertFalse(np.allclose(*[source_rotations(cal, s)[1] for s in slots]))
                    apply_calibration(motions, cal, tmp, .15)
                    for col, (s, slot) in enumerate(slots.items()):
                        rmi, rsb = source_rotations(cal, s)
                        np.testing.assert_allclose(rmi, expected[s][0], atol=1e-12)
                        np.testing.assert_allclose(rsb, (rmi @ motions[s].RIS[0]).T @ NPOSE[slot], atol=1e-12)
                        np.testing.assert_allclose(result["RMI"][col], rmi, atol=1e-12)
                        np.testing.assert_allclose(result["RSB"][col], rsb, atol=1e-12)
                        np.testing.assert_allclose(result["ori_MB"][:, col], np.tile(NPOSE[slot], (len(result["RIS"]), 1, 1)), atol=1e-12)
                        np.testing.assert_allclose(result["acc_M_m_s2"][:, col],
                                                   (rmi @ np.einsum("nij,nj->ni", arrays["RIS"][:, col], arrays["user_acc_S_m_s2"][:, col]).T).T,
                                                   atol=1e-12)
                        with np.load(Path(tmp) / f"{s}-calibrated.npz", allow_pickle=False) as native:
                            np.testing.assert_allclose(native["RMI"], rmi)
                            np.testing.assert_allclose(native["RSB"], rsb)
                            self.assertEqual(native["placement"].item(), info["source_placements"][s])
                            self.assertEqual(native["target_frame"].item(), info["source_target_frames"][s])

    def test_right_wrist_uses_right_npose_not_left(self):
        motions, _, _ = synthetic()
        outputs = []
        for watch in WATCH_PLACEMENTS:
            slots, info = resolve_placements(watch=watch, phone="left_thigh_pocket")
            cal, _, _ = auto_calibrate(motions, slots, Config(), info)
            outputs.append(source_rotations(cal, "apple_watch"))
        np.testing.assert_allclose(outputs[0][0], outputs[1][0])
        self.assertFalse(np.allclose(outputs[0][1], outputs[1][1]))

    def test_cli_labels_all_outputs_and_keeps_uwb_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            inputs.mkdir()
            write_split_fixture(inputs)
            previous = None
            for i, (watch, phone) in enumerate((("left_wrist", "right_thigh_pocket"),
                                               ("right_wrist", "left_thigh_pocket"),
                                               ("left_wrist", "left_hand"))):
                out = root / str(i)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    status = main([str(inputs), "--output", str(out), "--watch-placement", watch,
                                   "--phone-placement", phone])
                self.assertEqual(status, 0)
                cal = json.loads((out / "calibration.json").read_text())
                self.assertEqual(cal["schema_version"], 2)
                self.assertEqual(cal["placement"]["source_placements"], {"apple_watch": watch, "iphone": phone})
                for file in ("alignment.json", "calibration_quality.json"):
                    self.assertEqual(json.loads((out / file).read_text())["placement"], cal["placement"])
                for file in ("aligned_sensors.npz", "aligned_calibrated.npz", "native_modalities.npz"):
                    with np.load(out / file, allow_pickle=False) as x:
                        self.assertEqual(x["source_placements"].tolist(), [watch, phone])
                        self.assertEqual(x["source_slots"].tolist(), list(cal["source_slots"].values()))
                        self.assertEqual(x["placement_selection"].item(), "explicit")
                        for key in x.files:
                            self.assertFalse(x[key].dtype.hasobject)
                with np.load(out / "aligned_sensors.npz") as x:
                    unchanged = {k: x[k] for k in ("timestamp_unix_ns", "RIS", "RSI", "user_acc_S_m_s2", "gyro_S_rad_s",
                                                   "raw_acc_apple_g", "uwb_distance_m", "uwb_valid", "uwb_measurement_unix_ns")}
                    if previous:
                        for k in unchanged:
                            np.testing.assert_array_equal(unchanged[k], previous[k])
                    previous = unchanged

    def test_legacy_defaults_and_numeric_overrides_remain_explicitly_unverified(self):
        slots, info = resolve_placements()
        self.assertEqual(slots, {"apple_watch": 0, "iphone": 3})
        self.assertEqual(info["selection"], "legacy_defaults_assumed")
        self.assertFalse(info["downstream_model_compatibility_verified"])
        slots, info = resolve_placements({"apple_watch": 1, "iphone": 2, "airpods": 4})
        self.assertEqual(set(info["source_placements"].values()), {"unspecified"})

    def test_invalid_or_ambiguous_placement_arguments_fail_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            for flags in (["--watch-placement", "right_wrist"],
                          ["--phone-placement", "right_hand"],
                          ["--watch-placement", "left_hand", "--phone-placement", "right_hand"],
                          ["--source-slot", "iphone:3", "--watch-placement", "left_wrist", "--phone-placement", "right_hand"],
                          ["--source-slot", "watch:0", "--source-slot", "apple_watch:1"],
                          ["--source-slot", "iphone:7"]):
                with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(["missing.ndjson", "--output", str(out)] + flags)
                self.assertFalse(out.exists())

    def test_same_slot_references_are_compared_per_source(self):
        motions, _, _ = synthetic()
        slots, info = resolve_placements(watch="left_wrist", phone="left_hand")
        cal, _, _ = auto_calibrate(motions, slots, Config(), info)
        comparison = compare(cal, cal)
        for values in comparison["sources"].values():
            self.assertLess(max(values.values()), 2e-6)
        ref = json.loads(json.dumps(cal))
        del ref["source_calibration"]
        with self.assertRaisesRegex(ValueError, "cannot represent"):
            source_rotations(ref, "iphone")

    def test_legacy_reference_fallback_and_named_replay(self):
        motions, _, _ = synthetic()
        slots, info = resolve_placements(watch="right_wrist", phone="left_hand")
        cal, _, _ = auto_calibrate(motions, slots, Config(), info)
        old = json.loads(json.dumps(cal))
        del old["source_calibration"], old["placement"]
        result = compare(cal, old)
        self.assertEqual(len(result["warnings"]), 2)
        for s in slots:
            for a, b in zip(source_rotations(cal, s), source_rotations(old, s)):
                np.testing.assert_array_equal(a, b)
        replay, _, _ = replay_reference(motions, slots, cal, Config())
        self.assertEqual(replay["placement"], info)

    def test_reject_mismatched_source_and_placement_metadata(self):
        with self.assertRaises(ValueError):
            describe_placements({"iphone": 0}, {"iphone": "right_hand"}, "explicit")
        motions, slots, _ = synthetic()
        clocks, _ = clock_models(motions)
        arrays, _ = build_aligned(motions, {}, slots, clocks)
        cal, _, _ = auto_calibrate(motions, slots, Config())
        arrays["source_slots"][0] = 1
        with self.assertRaises(CalibrationError):
            calibrated_arrays(arrays, cal)


if __name__ == "__main__":
    unittest.main()
