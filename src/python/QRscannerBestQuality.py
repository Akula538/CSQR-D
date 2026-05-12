from find_qr import find, recognize_distribution, find_qr_zxing, recognize_search_distribution
from interpolation_correct import interpolation_correct_homo as correct
from vectorfield_correct import vectorfield_correct_homo as vector_correct
from vectorfield_correct import vectorfield_nodes
from binarizer import binarize_zxing_cpp as binarize
from decoder import decode, decode_qreader, decode_zxingcpp
from homographic_correct import homographic_correct_perspective, find_homo
from tools import *
import interpolation_correct
import TPS_correct
import cv2
from PIL import Image

def detectAndDecode(img, img_downscaled = None, can_homo = True):
    # img = np.array(img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # height, weight, _ = img.shape
    # scale_ = min(2000 / weight, 2000 / height)
    # img = cv2.resize(img, (int(weight * scale_), int(height * scale_)))
    
    # height, weight, _ = img.shape
    # scale = min(1000 / weight, 1000 / height)
    # img_downscaled = cv2.resize(img, (int(weight * scale), int(height * scale)))
    
    # weight, height = img.size
    # scale_ = min(2000 / weight, 2000 / height)
    # img = img.resize((int(weight * scale_), int(height * scale_)))
    
    
    if img_downscaled is None:
        weight, height = img.size
        scale = min(1000 / weight, 1000 / height)
        img_downscaled = img.resize((int(weight * scale), int(height * scale)))
    else:
        scale = 1/2

    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_downscaled = np.array(img_downscaled)
    img_downscaled = cv2.cvtColor(img_downscaled, cv2.COLOR_RGB2BGR)
    
    # show_points(img_downscaled)
    
    
    
    # img = binarize(img)
    # print("fsfsd")
    # show_points(img)
    
    result = decode(img_downscaled)
    if result is not None:
        return result
    result = decode_qreader(img_downscaled)
    if result is not None:
        return result
        
    try:
        qr, img_binarized = find_qr_zxing(img_downscaled)
    # try:pass
    except:
        print("zxing CTF")
        try:
            img_binarized = binarize(img_downscaled)
            qr = find(img_binarized)
        except:
            return "CTF"
        
    # show_points(img_binarized, qr)

            
    try:
        size, pattern = recognize_distribution(img_binarized, qr)
        
        search_pattern = recognize_search_distribution(img_binarized, qr)
        
        pattern_recovered = interpolation_correct.recover_pattern_if_need(qr, size, pattern, search_pattern)
    # try: pass
    except Exception as e:
        print(e)
        return "CTF"
    
    
    tps = TPS_correct.fit_tps_full_qr(qr, size)
    
    corrected = TPS_correct.tps_correct(img_downscaled, tps, size, square_size=6)
    
    cv2.imwrite("corrected.png", corrected)
    
    result = decode_zxingcpp(corrected)
    if result: return result
    result = decode_qreader(corrected)
    if result: return result
    
    tps1 = TPS_correct.fit_tps_alligment_center(qr, size)
    
    corrected = TPS_correct.tps_correct(img_downscaled, tps1, size, square_size=6)
    
    cv2.imwrite("corrected.png", corrected)
    
    result = decode_zxingcpp(corrected)
    if result: return result
    result = decode_qreader(corrected)
    if result: return result
    
    # corrected = correct(img_downscaled, qr, pattern, QRsize=size, square_size=6, fract=5) # 6 5

    # return result
    
    # show_points(img_binarized, list(qr) + list(pattern_recovered[0]) + list(pattern_recovered[1]), 1, (0, 0, 255))
        
    for i in range(len(qr)):
        qr[i] = qr[i] / scale
    
    for i in range(len(pattern_recovered)):
        for j in range(len(pattern_recovered[i])):
            pattern_recovered[i][j] = pattern_recovered[i][j] / scale
    tps.affine(1 / scale, 0)
    # show_points(img, list(qr) + list(pattern_recovered[0]) + list(pattern_recovered[1]), 4, (0, 0, 255))
    
    # corrected = vector_correct(img, qr, pattern=pattern_recovered, QRsize=size, tps=tps, square_size=6, fract=5, lines_fract=20) # 6 5 9

    # img, qr, pattern_recovered = cut_image(img, qr, pattern_recovered)
    # show_points(img, list(qr) + list(pattern_recovered[0]) + list(pattern_recovered[1]), 4, (0, 0, 255))
    
    target, scr = vectorfield_nodes(img, qr, pattern=pattern_recovered, QRsize=size, tps=tps, square_size=6, fract=5, lines_fract=20) # 6 5 9
    
    tps = TPS_correct.TPS()
    tps.fit(target, scr)
    
    corrected = TPS_correct.tps_correct(img, tps, size, square_size=6)
    
    cv2.imwrite("corrected.png", corrected)
    
    result = decode_zxingcpp(corrected)
    if result: return result
    result = decode_qreader(corrected)
    if result: return result
        
        
    return result


if __name__ == "__main__":
    # img = Image.open("data/v2/M/big/img0102_v2_M_big_d2.png")
    # img = Image.open("data/v2/M/big/img0114_v2_M_big_d2.png")
    # img = Image.open("data/v2/L/big/img0046_v2_L_big_d3.png")
    # img = Image.open("data_orig/v2/M/big/img0102_v2_M_big_d2.png")
    # img = Image.open("data/v2/H/small/img0039_v2_H_small_d1.png")
    # img = Image.open("data_orig/v2/M/big/img0090_v2_M_big_d2.png")
    # img = Image.open("data_orig/v2/M/small/img0120_v2_M_small_d2.png")
    img = Image.open("data_orig/v2/Q/small/img0164_v2_Q_small_d2.png")
    
    # img = Image.open("corrected.png")
    
    
    
    print(detectAndDecode(img, can_homo=False))
    # print(detectAndDecode_TPS(img))
    
    # corrected.save("corrected.png")
