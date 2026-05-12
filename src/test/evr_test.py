from pathlib import Path
from QRscanner import detectAndDecode
from decoder import decode
from find_qr import zxing_check
import decoder
import cv2
from PIL import Image
import time
import numpy as np

path_list = [
    "data2/train/83.png",
    "data2/train/70.png",
    "data2/test/150.png",
    "data2/train/94.png",
    "data2/train/86.png",
    "data2/test/144.png"
]

def test(data_path):
    paths = list(Path(data_path).rglob("*.png")) + list(Path(data_path).rglob("*.jpg"))
    paths.sort()
    
    # paths = [Path(p) for p in path_list]
    
    total = len(paths)
    passed = 0
    
    with open("evr_test_log.txt", "w") as file:
        file.write("")
    
    with open("evr_test_log.txt", "a", buffering=1, encoding="utf-8") as file:
        
        for path in paths:
            print(path)
            # with open("ts.txt", 'a') as f: f.write(f"{path}\n")
            
            # img = cv2.imread(str(path))
            img = Image.open(path)
            img_downscaled = None
            
            # path_str = str(path)
            # path_str = path_str.replace("2000", "1000")
            # # print(path_str)
            # img_downscaled = Image.open(path_str)
            
            start = time.time()
            
            # res = detectAndDecode(img, img_downscaled)
            # res = detectAndDecode_TPS(img)
            res1 = decoder.decode_qreader(img)
            # res = zxing_check(img)
            res = decoder.qreader_style_global_decode(np.array(img), decoder.decode_zxingcpp)
            
            fin = time.time() - start
            
            if res is None:
                res = "None"
            elif res[:5] == "GANYU":
                passed += 1
                # with open("ts.txt", 'a') as f: f.write(f"OK\n") 
                
                
            if res1 is None:
                res1 = "None"
            res = res.replace("\n", " ")
            res1 = res1.replace("\n", " ")
            
            if (res1 == "None" or res == "None") and res != res1:
                file.write(f"{path.name + ' ' * (30 - len(path.name))} -> {res} | {res1}\n")
            # file.write(f"{path.name + ' ' * (30 - len(path.name))} -> {res} -> {fin}\n")
            # file.flush()

    print(f"passed {passed}/{total}")
