from PIL import Image, ImageOps
import cv2
import numpy as np
import subprocess

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def binarize(orig):
    
    pixels = orig.load()

    weight, height = orig.size

    for x in range(weight):
        for y in range(height):
            pixel = pixels[x, y]
            
            if sum(pixel) / 3 < 115:
            # if min(pixel) < 80:
                pixels[x, y] = BLACK
            else:
                pixels[x, y] = WHITE
    
    return orig

def binarize_mean(orig, offset = 10, window_size = None):
    if window_size is None:
        window_size = min(orig.size) // 5
    
    img = np.array(orig)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(window_size,window_size))
    # gray = clahe.apply(gray)
    
    blur = cv2.blur(gray, (window_size, window_size))
    
    binary = np.where(gray > blur - offset, 255, 0).astype(np.uint8)
    # binary = gray - (blur - offset) + 128
    
    binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    binary = Image.fromarray(binary)
    binary = ImageOps.expand(binary, border=5, fill='white')
    
    return binary


def binarize_smart(orig, offset = 10, window_size = None):
    if window_size is None:
        window_size = min(orig.size) // 5
    
    img = np.array(orig)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(window_size,window_size))
    gray = clahe.apply(gray)

    # binary = cv2.adaptiveThreshold(
    #     gray, 
    #     255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY,
    #     blockSize=35,
    #     C=-10
    # )
    
    binary = gray
    
    binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    binary = Image.fromarray(binary)
    binary = ImageOps.expand(binary, border=5, fill='white')
    
    return binary


def binarize_sauvola(image, window=None, k=0.3, R=128):
    if window is None:
        window = min(image.size) // 5
    
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    mean = cv2.boxFilter(gray, cv2.CV_32F, (window, window))
    sqmean = cv2.boxFilter(gray * gray, cv2.CV_32F, (window, window))
    variance = sqmean - mean * mean
    std = np.sqrt(np.maximum(variance, 0))

    thresh = mean * (1 + k * ((std / R) - 1))
    binary = (gray > thresh).astype(np.uint8) * 255

    return Image.fromarray(binary)


def binarize_zxing_cpp(img):
    img.save("zxing-cpp_finder/input.png")
    subprocess.run('.\\"zxing-cpp_finder\\zxing_binarize.exe" "zxing-cpp_finder/input.png"', shell=True)
    output = Image.open("zxing-cpp_finder/output.png")
    return output.convert("RGB")


if __name__ == "__main__":
    
    # orig = Image.open("orig.png")
    orig = Image.open("corrected.png")
    # orig = Image.open("data/v2/H/small/img0039_v2_H_small_d1.png")
    # orig = Image.open("data/v2/H/big/img0017_v2_H_big_d0.png")
    orig = Image.open("C:/Users/user/Downloads/sample.png")
    
    orig = binarize_zxing_cpp(orig)
    orig.save("corrected.png")