import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mobileposer"))
from apple_static_mocap import DEFAULT_CONFIG, transform_model_inputs
from sensor_apple.offline import C, CalibrationError, Motion, NPOSE
from sensor_apple.static_calibration import checked_rotation, choose_npose, fit_rsb, read_saved_rmi


def rot(x, y, z):
    return Rotation.from_euler("xyz", [x, y, z], degrees=True).as_matrix()


class StaticMocapTests(unittest.TestCase):
    def test_fit_rsb_recovers_mounting_and_closes_pose(self):
        rmi, rsb = rot(10, 20, 30), rot(-25, 12, 140)
        ris = rmi.T @ NPOSE[0] @ rsb.T
        recovered, mean = fit_rsb(rmi, np.repeat(ris[None], 30, axis=0), NPOSE[0])
        np.testing.assert_allclose(recovered, rsb, atol=1e-12)
        np.testing.assert_allclose(rmi @ mean @ recovered, NPOSE[0], atol=1e-12)


    def test_transform_orders_source_slots_and_zeroes_missing_slots(self):
        # Opposite source order from the default mapping catches source/slot mixups.
        sources = ["iphone", "apple_watch"]
        slots = [3, 0]
        ris = np.array([[rot(1, 2, 3), rot(7, 8, 9)]])
        acc = np.array([[[1., 2., 3.], [4., 5., 6.]]])
        arrays = {"timestamp_unix_ns": np.array([1790000000000000000], np.int64),
                  "source_names": np.array(sources), "source_slots": np.array(slots),
                  "RIS": ris, "user_acc_S_m_s2": acc}
        cal = {"source_calibration": {s: {"slot": slot, "RMI": rot(10, 20, 30).tolist(),
                 "RSB": rot(40, 50, 60).tolist()} for s, slot in zip(sources, slots)}}
        out = transform_model_inputs(arrays, cal)
        assert out["aM"].shape == (1, 5, 3)
        assert out["RMB"].shape == (1, 5, 3, 3)
        assert not out["aM"][:, [1, 2, 4]].any()
        assert not out["RMB"][:, [1, 2, 4]].any()
        for col, slot in enumerate(slots):
            np.testing.assert_allclose(out["aM"][0, slot], rot(10, 20, 30) @ ris[0, col] @ acc[0, col], atol=1e-6)
            np.testing.assert_allclose(out["RMB"][0, slot], rot(10, 20, 30) @ ris[0, col] @ rot(40, 50, 60), atol=1e-6)
            # World acceleration agrees with bone acceleration via RMB; no extra RSB is applied to aM.
            np.testing.assert_allclose(out["RMB"][0, slot] @ out["aB_source"][0, col], out["aM"][0, slot], atol=2e-6)


    def test_saved_candidate_requires_override_and_is_never_modified(self):
        artifact = {"schema": "static_rmi_v1", "coordinates": {"C": C.tolist()}, "sessionID": "old",
                    "eligible_for_mocap_reuse": False, "sources": {"iphone": {"candidate_RMI": np.eye(3).tolist()}}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calibration.json"
            path.write_text(json.dumps(artifact))
            original = path.read_bytes()
            with self.assertRaisesRegex(CalibrationError, "not verified"):
                read_saved_rmi(path, ["iphone"], "new")
            matrices, _, reasons = read_saved_rmi(path, ["iphone"], "new", True)
            assert len(reasons) == 2
            np.testing.assert_array_equal(matrices["iphone"], np.eye(3))
            assert path.read_bytes() == original


    def test_reflection_is_not_a_valid_rotation(self):
        with self.assertRaisesRegex(CalibrationError, "proper rotation"):
            checked_rotation(np.diag([1, 1, -1]), "reflection")


    def test_opening_search_does_not_use_a_later_pause(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        t = np.arange(0, 15, .02)
        gyro = np.zeros((len(t), 3))
        gyro[t < 9, 0] = 1
        m = Motion("iphone", "test", 1790000000000000000 + np.rint(t * 1e9).astype(np.int64),
                   t, np.arange(len(t)), np.repeat(np.eye(3)[None], len(t), axis=0),
                   np.zeros((len(t), 3)), gyro)
        with self.assertRaisesRegex(CalibrationError, "opening N-pose"):
            choose_npose({"iphone": m}, config)
        _, bounds, quality = choose_npose({"iphone": m}, config, [10., 12.])
        assert bounds == [10., 12.]
        assert not quality["iphone"]["issues"]


    def test_nonfinite_input_is_rejected(self):
        arrays = {"timestamp_unix_ns": np.array([0]), "source_names": np.array(["iphone"]),
                  "source_slots": np.array([3]), "RIS": np.full((1, 1, 3, 3), np.nan),
                  "user_acc_S_m_s2": np.zeros((1, 1, 3))}
        cal = {"source_calibration": {"iphone": {"slot": 3, "RMI": np.eye(3), "RSB": np.eye(3)}}}
        with self.assertRaisesRegex(CalibrationError, "Non-finite"):
            transform_model_inputs(arrays, cal)


if __name__ == "__main__":
    unittest.main()
