from PIL import Image
from find_qr import find, recognize_distribution, find_qr_zxing
from interpolation_correct import interpolation_correct_homo as correct
from binarizer import binarize_mean as binarize
from decoder import decode, decode_qreader, decode_zxingcpp
from homographic_correct import homographic_correct_perspective, find_homo
from tools import *


def detectAndDecode(img, can_homo = True):    
    weight, height = img.size
    scale = min(1000 / weight, 1000 / height)
    img = img.resize((int(weight * scale), int(height * scale)))
    
    # img = binarize(img)
    
    result = decode(img)
    if result is not None:
        return result
    
        
    try:
        qr, img_binarized = find_qr_zxing(img)
    except:
        print("zxing CTF")
        try:
            img_binarized = binarize(img)
            qr = find(img_binarized)
        except:
            return "CTF"

            
    try:
        size, pattern = recognize_distribution(img_binarized, qr)
    except Exception as e:
        return "CTF"
    
    # print(pattern)

    corrected = correct(img, qr, pattern, QRsize=size, square_size=7, fract=1)
    result = decode(corrected)
    
    if result is None:
        result = decode_qreader(corrected)
    # if result is None:
    #     corrected = correct(img, qr, QRsize=size, square_size=7, fract=1)
    #     result = decode(corrected)
    # if result is None:
    #     result = decode_qreader(corrected)
    if can_homo and result is None:
        qr_H, img_H = homographic_correct_perspective(img_binarized, qr)
        # show_points(img_H)
        # img_H.save("corrected.png")
        result = detectAndDecode(img_H, can_homo=False)
        if result == "CTF":
            result = None
        if result is not None:
            return result
    
    corrected.save("corrected.png")
    
    return result


if __name__ == "__main__":
    img = Image.open("orig_bynarized1.png")
    # img = Image.open("corrected.png")
    # img = Image.open("images/image6.png")
    # img = Image.open("data/v2/L/big/img0052_v2_L_big_d2.png")
    
    print(detectAndDecode(img))
    
    # corrected.save("corrected.png")
