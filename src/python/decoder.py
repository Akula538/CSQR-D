from __future__ import annotations
import cv2
from PIL import Image
import numpy as np
from pyzbar.pyzbar import decode as pyzbar_decode
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

from qreader import QReader
qreader = QReader(model_size="n")
def decode_qreader(img): #0.46
    img = np.array(img)
    try:
        return qreader.detect_and_decode(image=img)[0]
    except:
        return None

def decode_zxingcpp(img):
    try:
        result = zxingcpp.read_barcode(img).text
        return result
    except:
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