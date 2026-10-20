"""Headless Python implementation of the C++ scanner recovery pipeline."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from binarizer import binarize_zxing_cpp
from decoder import decode_qr_grid, decode_variants, decode_zxingcpp
from find_qr import (find, find_qr_zxing, recognize_distribution,
                     recognize_distribution_polyline, recognize_search_distribution, zxing_find)
from homographic_correct import HomographyMatrix
from interpolation_correct import recover_pattern_if_need
from TPS_correct import (TPS, fit_tps_alligment_center, fit_tps_full_qr,
                         fit_tps_no_alligment, tps_correct)
from tools import cut_image
from vectorfield_correct import DirectionField, constructPolylines4recdist, vectorfield_nodes


def _gray(image):
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    array = np.asarray(image)
    if array.ndim == 2:
        return array.copy()
    return cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY if array.shape[2] == 4 else cv2.COLOR_BGR2GRAY)


def _decode_rectified(image, size):
    for decoder in (lambda value: decode_qr_grid(value, size), decode_zxingcpp, decode_variants):
        result = decoder(image)
        if result is not None:
            return result
    return None


def _scan_once(gray, detection_side, supplied=None, debug=None):
    height, width = gray.shape
    if supplied is None:
        scale = detection_side / max(width, height)
        small = cv2.resize(gray, (max(1, int(width * scale)), max(1, int(height * scale))),
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    else:
        small = _gray(supplied)
        scale = max(small.shape) / max(gray.shape)
        if abs(width * scale - small.shape[1]) > 1 or abs(height * scale - small.shape[0]) > 1:
            raise ValueError("The supplied image must preserve the original aspect ratio")
    result = decode_zxingcpp(small)
    if result is not None:
        return result
    patterns = zxing_find(small)

    def original_variants():
        return decode_variants(small) if len(patterns) >= 2 else None

    try:
        qr, binary = find_qr_zxing(small, patterns)
    except (ValueError, IndexError, RuntimeError, cv2.error):
        try:
            qr, binary = find_qr_zxing(cv2.bitwise_not(small))
            gray = cv2.bitwise_not(gray)
        except (ValueError, IndexError, RuntimeError, cv2.error):
            try:
                binary = binarize_zxing_cpp(small)
                qr = find(binary)
            except (ValueError, IndexError, RuntimeError, cv2.error):
                return original_variants() or "CTF"
    qr = np.asarray(qr, dtype=float)
    try:
        size, pattern = recognize_distribution(binary, qr)
        if not 21 <= size <= 177:
            raise ValueError("Invalid QR dimension")
        search_pattern = recognize_search_distribution(binary, qr)
        recovered = recover_pattern_if_need(qr, size, pattern, search_pattern)
    except (ValueError, IndexError, RuntimeError, cv2.error, ZeroDivisionError):
        return original_variants() or "CTF"

    tps = fit_tps_no_alligment(qr, size) if size <= 21 else fit_tps_full_qr(qr, size)

    def corrected_decode(transform, source, dimension, name):
        corrected = tps_correct(source, transform, dimension, square_size=6)
        if debug is not None:
            cv2.imwrite(str(debug / name), corrected)
        return _decode_rectified(corrected, dimension)

    result = corrected_decode(tps, small, size, "corrected_full_tps.png")
    if result is not None:
        return result
    result = original_variants()
    if result is not None:
        return result
    result = corrected_decode(fit_tps_alligment_center(qr, size), small, size, "corrected_alignment_tps.png")
    if result is not None:
        return result

    qr /= scale
    recovered = [[np.asarray(point) / scale for point in line] for line in recovered]
    crop, qr, recovered, shift = cut_image(gray, qr, recovered, size, return_shift=True)
    tps.affine(1 / scale, -shift)
    field = DirectionField(crop)
    try:
        lookup = HomographyMatrix(scale, shift * scale)
        polylines = constructPolylines4recdist(field, qr, 30)
        previous_size = size
        size, pattern = recognize_distribution_polyline(binary, qr, lookup, polylines)
        if not 21 <= size <= 177:
            raise ValueError("Invalid QR dimension")
        search_pattern = [None if line is None else np.asarray(line) / scale - shift
                          for line in search_pattern]
        recovered = recover_pattern_if_need(qr, size, pattern, search_pattern)
        if size != previous_size:
            tps = fit_tps_no_alligment(qr, size) if size <= 21 else fit_tps_full_qr(qr, size)
    except (ValueError, IndexError, RuntimeError, cv2.error, ZeroDivisionError):
        return "CTF"

    for spacing, steps in ((5, 20), (3, max(40, 2 * size))):
        target, source = vectorfield_nodes(crop, qr, pattern=recovered, QRsize=size, tps=tps,
                                           square_size=6, fract=spacing, lines_fract=steps, df=field)
        transform = TPS()
        transform.fit(target, source)
        result = corrected_decode(transform, crop, size, f"corrected_vector_{spacing}_tps.png")
        if result is not None:
            return result
    return None


def detectAndDecode(img, img_downscaled=None, can_homo=True, *, debug_output_dir=None):
    """Accept a PIL RGB image or an OpenCV BGR/grayscale uint8 array.

    Return a decoded string, "CTF" for failed geometry, or None when a
    recovered grid cannot be decoded. Debug images are written only on request.
    can_homo is retained for compatibility with existing callers.
    """
    gray = _gray(img)
    if gray.size == 0:
        return "CTF"
    if gray.dtype != np.uint8:
        raise ValueError("Expected a uint8 image")
    debug = Path(debug_output_dir) if debug_output_dir is not None else None
    if debug is not None:
        debug.mkdir(parents=True, exist_ok=True)
    for side in (1000, 1500):
        try:
            result = _scan_once(gray, side, img_downscaled if side == 1000 else None, debug)
        except (ValueError, IndexError, RuntimeError, cv2.error, ZeroDivisionError, np.linalg.LinAlgError):
            result = "CTF"
        if result != "CTF":
            return result
    return "CTF"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--debug-output", type=Path)
    arguments = parser.parse_args()
    with Image.open(arguments.image) as image:
        print(detectAndDecode(image, debug_output_dir=arguments.debug_output))
