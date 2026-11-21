"""Static-pose math, validation, NDJSON loading, and portable CLI regressions."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mobileposer"))
from apple_static_rmi import main
from sensor_apple.offline import C, CalibrationError
from sensor_apple.static_calibration import (
    calculate_static_rmi, load_static_recording, read_saved_rmi, select_static_window,
)

CONFIG = Path(__file__).resolve().parents[1] / "mobileposer/configs/apple_static_rmi_pose.json"
TARGET = np.asarray(json.loads(CONFIG.read_text())["RMS_targets"]["iphone"], float)


def fixture(rmi=None, raw_gravity=None):
    rmi = np.eye(3) if rmi is None else rmi
    ris = rmi.T @ TARGET
    t = np.arange(0, 16, .02)
    raw_gravity = C @ TARGET.T @ np.array([0., -1., 0.]) if raw_gravity is None else raw_gravity
    return {"time": t, "RIS": np.repeat(ris[None], len(t), axis=0),
            "gyro": np.zeros((len(t), 3)), "acc": np.zeros((len(t), 3)),
            "gravity": np.tile(raw_gravity, (len(t), 1))}


def write_recording(folder, bad_gravity=False, change_frame=False):
    data = fixture(raw_gravity=[0., 0., 1.] if bad_gravity else None)
    for source in ("iphone", "apple_watch"):
        rows = []
        for i, t in enumerate(data["time"]):
            q = (C @ data["RIS"][i] @ C).T
            v = {f"rotation_m{r+1}{c+1}": float(q[r, c]) for r in range(3) for c in range(3)}
            v.update({f"rotation_{k}_rad_s": 0. for k in "xyz"})
            v.update({f"user_accel_{k}_g": 0. for k in "xyz"})
            v.update({f"gravity_{k}": float(data["gravity"][i, j]) for j, k in enumerate("xyz")})
            v.update(attitude_reference_frame=2 if change_frame and i == 100 else 4, magnetic_accuracy=3)
            rows.append(json.dumps({"source": source, "sensor": "device_motion", "sessionID": "fixture",
                "timestampUnixNs": str(1790000000000000000 + round(t * 1e9)), "timestampMonotonicS": t,
                "sequenceNumber": i, "values": v}))
        (folder / f"{source}.ndjson").write_text("\n".join(rows) + "\n")


class StaticRMITests(unittest.TestCase):
    def test_recovers_known_rmi(self):
        rmi = Rotation.from_euler("xyz", [15, 20, -40], degrees=True).as_matrix()
        result = calculate_static_rmi(fixture(rmi), [3, 11], TARGET)
        np.testing.assert_allclose(result["candidate_RMI"], rmi, atol=1e-12)
        self.assertFalse(result["validation_issues"])

    def test_gravity_rejects_historical_sketch_mismatch(self):
        result = calculate_static_rmi(fixture(raw_gravity=[0, 0, 1]), [3, 11], TARGET)
        self.assertEqual(result["validation_issues"], ["gravity_does_not_point_model_minus_y"])
        self.assertAlmostEqual(result["gravity_error_to_down_p95_deg"], 180)

    def test_handling_avoided(self):
        data = fixture()
        data["gyro"][(data["time"] < 4) | (data["time"] > 13)] = 10
        a, b = select_static_window({"iphone": data, "apple_watch": data})
        self.assertGreaterEqual(a, 3.75)
        self.assertLessEqual(b, 13.25)
        self.assertEqual(b - a, 8)

    def test_all_moving_rejected(self):
        data = fixture()
        data["gyro"][:] = 2
        with self.assertRaises(CalibrationError):
            select_static_window({"iphone": data})

    def test_manual_window_checked_not_blindly_accepted(self):
        for bounds in ([1, 2], [np.nan, 10], [10, 18]):
            with self.assertRaises(CalibrationError):
                select_static_window({"iphone": fixture()}, requested=bounds)
        self.assertEqual(select_static_window({"iphone": fixture()}, requested=[3, 11]), [3., 11.])

    def test_changing_reference_frame_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_recording(folder, change_frame=True)
            with self.assertRaisesRegex(CalibrationError, "reference frame changed"):
                load_static_recording(sorted(folder.glob("*.ndjson")))

    def test_integer_timestamp_origin_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_recording(folder)
            sid, origin, streams, _ = load_static_recording(sorted(folder.glob("*.ndjson")))
            self.assertEqual(origin, 1790000000000000000)
            self.assertEqual(sid, "fixture")
            self.assertEqual(streams["iphone"]["frames"], {"4": 800})
            self.assertAlmostEqual(streams["iphone"]["time"][1], .02, places=12)

    def test_cli_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_recording(folder)
            out = folder / "result"
            args = [str(folder), "--pose-config", str(CONFIG), "--output", str(out)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            artifact = out / "calibration.json"
            original = artifact.read_bytes()
            matrices, _, reasons = read_saved_rmi(artifact, ["iphone", "apple_watch"], "fixture")
            self.assertFalse(reasons)
            np.testing.assert_allclose(matrices["iphone"], np.eye(3), atol=1e-12)
            self.assertTrue((out / "diagnostic.png").is_file())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(artifact.read_bytes(), original)

    def test_cli_blocked_artifact_keeps_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_recording(folder, bad_gravity=True)
            out = folder / "result"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main([str(folder), "--pose-config", str(CONFIG), "--output", str(out)])
            self.assertEqual(code, 2)
            artifact = out / "calibration.json"
            self.assertEqual(json.loads(artifact.read_text())["status"], "blocked_coordinate_validation")
            with self.assertRaises(CalibrationError):
                read_saved_rmi(artifact, ["iphone"], "fixture")
            # Contradictory eligibility cannot bypass the status check.
            report = json.loads(artifact.read_text())
            report["eligible_for_mocap_reuse"] = True
            artifact.write_text(json.dumps(report))
            with self.assertRaises(CalibrationError):
                read_saved_rmi(artifact, ["iphone"], "fixture")


if __name__ == "__main__":
    unittest.main()
