from __future__ import annotations
import cv2
from PIL import Image
import numpy as np
from random import random
from time import time
import zxingcpp
from tools import *


from typing import Callable, Any, Iterable
import cv2
import numpy as np

_SHARPEN_KERNEL = np.array(
    [
        [-1.0, -1.0, -1.0],
        [-1.0,  9.0, -1.0],
        [-1.0, -1.0, -1.0],
    ],
    dtype=np.float32,
)

def qreader_style_global_decode(
    image: np.ndarray,
    decode: Callable[[np.ndarray], Any],
) -> Any | None:
    """
    Повторяет глобальные preprocessing-этапы из QReader и передает каждый
    вариант в вашу функцию decode(image).

    Ожидается RGB uint8 изображение HxWxC или HxW.
    decode(image) -> результат декодирования, либо None/False при неудаче.
    """

    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")

    # Handle different channel configurations
    if image.ndim == 3:
        if image.shape[2] == 1:
            # Single channel with explicit dimension
            image = image[:, :, 0]
        elif image.shape[2] == 4:
            # RGBA/BGRA -> convert to RGB by removing alpha channel
            image = image[:, :, :3]
        elif image.shape[2] != 3:
            raise ValueError(f"Expected 1, 3, or 4 channels, got {image.shape[2]} channels")

    def _try(img: np.ndarray) -> Any | None:
        result = decode(img)
        return result if result is not None and result is not False else None

    def _threshold_and_blur_decodings(gray_or_color: np.ndarray) -> Any | None:
        # 1) original
        result = _try(gray_or_color)
        if result is not None:
            return result

        # 2) Otsu binarization only for grayscale, exactly like QReader
        if gray_or_color.ndim == 2:
            _, binary_image = cv2.threshold(
                gray_or_color,
                thresh=0,
                maxval=255,
                type=cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            result = _try(binary_image)
            if result is not None:
                return result

        # 3) Gaussian blur variants
        for kernel_size in ((5, 5), (7, 7)):
            blurred = cv2.GaussianBlur(gray_or_color, kernel_size, sigmaX=0)
            result = _try(blurred)
            if result is not None:
                return result

        return None

    variants: list[np.ndarray] = [image]

    # Inverted image
    variants.append(np.array(255, dtype=np.uint8) - image)

    # Grayscale version, same channel assumption as QReader: RGB -> gray
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    variants.append(gray)

    # Sharpened grayscale
    if image.ndim == 3:
        sharpened = cv2.filter2D(src=image, ddepth=-1, kernel=_SHARPEN_KERNEL)
        sharpened_gray = cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY)
    else:
        sharpened_gray = cv2.filter2D(src=image, ddepth=-1, kernel=_SHARPEN_KERNEL)
    sharpened_gray = np.clip(sharpened_gray, 0, 255).astype(np.uint8)
    variants.append(sharpened_gray)

    # Apply the same progression as QReader: original -> inverted -> grayscale+filters -> sharpened+filters
    for idx, variant in enumerate(variants):
        if idx < 2:
            result = _try(variant)
            if result is not None:
                return result
        elif idx == 2:
            result = _threshold_and_blur_decodings(variant)
            if result is not None:
                return result
        else:
            result = _threshold_and_blur_decodings(variant)
            if result is not None:
                return result

    return None

def decode_pyzbar(img): #0.0012s
    from pyzbar.pyzbar import decode as pyzbar_decode
    results = pyzbar_decode(img)
    return results[0].data.decode('utf-8') if results else None


def decode_cv2(img): #0.007s
    img = np.array(img)
    
    if len(img.shape) >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    detector = cv2.QRCodeDetector()  
    data, points, _ = detector.detectAndDecode(img)
    
    if not data and points is not None:
        data, _ = detector.decodeCurved(img, points)
    
    return data or None

qreader = None
def decode_qreader(img): #0.46
    global qreader
    if qreader is None:
        from qreader import QReader
        qreader = QReader(model_size="n")
    img = np.array(img)
    try:
        return qreader.detect_and_decode(image=img)[0]
    except:
        return None

def decode_zxingcpp(img):
    result = zxingcpp.read_barcode(np.asarray(img), formats=zxingcpp.BarcodeFormat.QRCode,
                                  try_rotate=True, try_downscale=True, try_invert=True,
                                  text_mode=zxingcpp.TextMode.HRI)
    return result.text if result is not None and result.valid else None


def decode_variants(img):
    """The inexpensive grayscale preprocessing used by the C++ scanner."""
    gray = np.asarray(img)
    if gray.dtype != np.uint8 or gray.size == 0:
        return None
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    for index in range(2):
        image = gray if index == 0 else cv2.filter2D(gray, -1, _SHARPEN_KERNEL)
        if index:
            result = decode_zxingcpp(image)
            if result is not None:
                return result
        binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        result = decode_zxingcpp(binary)
        if result is not None:
            return result
        for kernel in (5, 7):
            result = decode_zxingcpp(cv2.GaussianBlur(image, (kernel, kernel), 0))
            if result is not None:
                return result
    return None


def _decode_modules(modules):
    # Python bindings expose the pure-image reader, not QRCode::Decode(bits).
    # Restore only fixed finder/separator/timing cells (never payload, format,
    # version or alignment cells) so pure extraction reads this exact grid.
    modules = modules.copy()
    n = len(modules)
    finder = np.full((7, 7), 0, dtype=np.uint8)
    finder[1:6, 1:6] = 255
    finder[2:5, 2:5] = 0
    for y, x in ((0, 0), (0, n - 8), (n - 8, 0)):
        modules[y:y + 8, x:x + 8] = 255
    modules[:7, :7] = finder
    modules[:7, n - 7:] = finder
    modules[n - 7:, :7] = finder
    timing = np.where(np.arange(8, n - 8) % 2 == 0, 0, 255).astype(np.uint8)
    modules[6, 8:n - 8] = timing
    modules[8:n - 8, 6] = timing
    image = np.repeat(np.repeat(np.pad(modules, 4, constant_values=255), 3, axis=0), 3, axis=1)
    result = zxingcpp.read_barcode(image, formats=zxingcpp.BarcodeFormat.QRCode,
                                  is_pure=True, try_rotate=False, try_downscale=False,
                                  try_invert=False, binarizer=zxingcpp.Binarizer.BoolCast,
                                  text_mode=zxingcpp.TextMode.HRI)
    return result.text if result is not None and result.valid else None


def decode_qr_grid(img, dimension, module_size=6):
    """Read known module centers, with the C++ thresholds and sampling phases."""
    gray = np.asarray(img)
    if (gray.dtype != np.uint8 or gray.ndim != 2 or dimension < 21 or dimension > 177
            or (dimension - 21) % 4 or module_size < 2
            or gray.shape != ((dimension + 4) * module_size,) * 2):
        return None
    smooth = cv2.blur(gray, (3, 3))
    centers = (np.arange(dimension) + 2) * module_size + module_size // 2
    for method in range(2):
        if method == 0:
            binary = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, module_size * 5 + 1, 5)
        else:
            binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        for dy in (0, -1, 1):
            for dx in (0, -1, 1):
                result = _decode_modules(binary[np.ix_(centers + dy, centers + dx)])
                if result is not None:
                    return result
    return None

# log = [0, 0, 0]
def decode(img):
    # print(log)
    # img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # img = Image.fromarray(img)
    result = decode_zxingcpp(img)
    if result is not None:
        # log[0] += 1
        return result
    
    return None
    
    result = decode_pyzbar(img)
    if result is not None:
        # log[1] += 1
        return result
    
    if random() < 0.02:
        result = decode_qreader(img)
        if result is not None:
            # log[2] += 1
            return result


if __name__ == "__main__":

    
    img = Image.open("debug/corrected_full_tps.png")
    
    # weight, height = img.size
    # scale = min(1000 / weight, 1000 / height)
    # img = img.resize((int(weight * scale), int(height * scale)))
    

    # start = time()
    # for i in range(100):
    #     decode_cv2(img)
    # print((time() - start) / 100)
    
    # show_points(img, [])
    print(decode_pyzbar(img))
    
    result = qreader_style_global_decode(np.array(img), decode_pyzbar)
    print(result)
    
    
    result = decode_zxingcpp(img)
    print(result)
    result = decode_qreader(img)
    print(result)
