import cv2
from PIL import Image
import numpy as np
from pyzbar.pyzbar import decode as pyzbar_decode
from random import random
from time import time
import zxingcpp


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


def decode_qreader(img): #0.46
    from qreader import QReader
    qreader = QReader()
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


def decode(img):
    result = decode_zxingcpp(img)
    if result is not None:
        return result
    
    result = decode_pyzbar(img)
    if result is not None:
        return result
    
    if random() < 0.02:
        result = decode_qreader(img)
        if result is not None:
            return result


if __name__ == "__main__":
    img = Image.open("corrected.png")
    # img = Image.open("data/v2/H/small/img0039_v2_H_small_d1.png")
    # img = Image.open("C:/Users/user/Downloads/sample.png")

    # start = time()
    # for i in range(100):
    #     decode_cv2(img)
    # print((time() - start) / 100)
    
    result = decode_zxingcpp(img)
    result = decode_qreader(img)
    print(result)
    
    