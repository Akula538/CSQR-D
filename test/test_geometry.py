"""Numerical checks for the Python geometry implementation (no GUI or model)."""
import sys
import unittest
import contextlib
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test/deps"))
sys.path.insert(0, str(ROOT / "src/python"))
import numpy as np
from scipy.ndimage import map_coordinates
from TPS_correct import TPS
from tools import cut_image
import cv2
from decoder import decode_qr_grid
from QRscanner import detectAndDecode
from vectorfield_correct import DirectionField
from interpolation_correct import recover_pattern_if_need


class GeometryTests(unittest.TestCase):
    def test_tps_warp_and_composition(self):
        for fractional in (False, True):
            src = np.array([[0, 0], [25, 0], [25, 25], [0, 25], [12, 12]], dtype=float)
            if fractional:
                src[-1, 0] += .123
            dst = np.array([[20, 20], [170, 25], [165, 170], [25, 165], [90, 105.]])
            tps = TPS()
            tps.fit(src, dst)
            y, x = np.mgrid[:200, :200]
            image = (x + 2 * y).astype(float)
            yy, xx = np.mgrid[:174, :174]
            grid = (np.column_stack([xx.ravel(), yy.ravel()]) - 12) / 6
            coords = tps.tps_transform(grid)
            expected = map_coordinates(image, coords[:, ::-1].T, order=1, cval=255).reshape(174, 174)
            np.testing.assert_allclose(tps.warp_image_tps(image, (174, 174), 6, 12),
                                       expected, atol=1e-8)
            before = tps.tps_transform([[9, 11]])
            tps.affine(2, np.array([3, 4]))
            tps.affine(.5, np.array([-2, 1]))
            np.testing.assert_allclose(tps.tps_transform([[9, 11]]), before + [-.5, 3])
            tps.fit(src, dst)
            np.testing.assert_allclose(tps.tps_transform([[9, 11]]), before)

    def test_crop_uses_integer_origin(self):
        qr = np.array([[20.7, 22.4], [70.3, 22.4], [70.3, 75.2], [20.7, 75.2]])
        original = qr.copy()
        _, qr, _, shift = cut_image(np.zeros((100, 100)), qr, [], return_shift=True)
        np.testing.assert_array_equal(shift, shift.astype(int))
        np.testing.assert_allclose(qr + shift, original)

    def test_grid_decoder_with_missing_finders(self):
        for version in (1, 2, 5):
            params = cv2.QRCodeEncoder_Params()
            params.version = version
            params.correction_level = cv2.QRCodeEncoder_CORRECT_LEVEL_L
            symbol = cv2.QRCodeEncoder_create(params).encode("CSQR-D")
            dimension = 17 + 4 * version
            image = cv2.resize(symbol, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
            self.assertEqual(decode_qr_grid(image, dimension), "CSQR-D")
            # Finder cells carry no payload; direct module decoding can read
            # the same symbol even if finder extraction would fail.
            image[12:54, 12:54] = 255
            self.assertEqual(decode_qr_grid(image, dimension), "CSQR-D")
        self.assertIsNone(decode_qr_grid(np.zeros((174, 174), np.uint8), 25))
        self.assertIsNone(decode_qr_grid(np.full((174, 174), 255, np.uint8), 25))
        self.assertIsNone(decode_qr_grid(np.zeros((174, 174), np.uint8), 26))

    def test_headless_scan_and_empty_direction_windows(self):
        params = cv2.QRCodeEncoder_Params()
        symbol = cv2.QRCodeEncoder_create(params).encode("CSQR-D")
        image = cv2.resize(symbol, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(detectAndDecode(image), "CSQR-D")
        self.assertEqual(stdout.getvalue(), "")
        field = DirectionField(image)
        self.assertEqual(field.get(-1000, -1000), (None, None))
        self.assertEqual(field.get(10000, 10000), (None, None))
        self.assertEqual(field.magnitude.dtype, np.float32)

    def test_pattern_recovery_does_not_mutate_search_lines(self):
        finder = np.array([[0, 0], [7, 0], [7, 7], [0, 7]], dtype=float)
        alignment = np.array([[17, 17], [20, 17], [20, 20], [17, 20]], dtype=float)
        qr = np.concatenate([finder, finder + [18, 0], alignment, finder + [0, 18]])
        search = [None] * 4
        first = recover_pattern_if_need(qr, 25, [None, None], search)
        self.assertEqual(search, [None] * 4)
        arrays = [np.asarray(first[0][:8]), np.asarray(first[0][-8:]),
                  np.asarray(first[1][:8]), np.asarray(first[1][-8:])]
        second = recover_pattern_if_need(qr, 25, [None, None], arrays)
        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()
