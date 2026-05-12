from pathlib import Path
from QRscanner import detectAndDecode
from decoder import decode
from find_qr import zxing_check
import decoder
import cv2
from PIL import Image
import time
import subprocess

def test(data_path):
    paths = list(Path(data_path).rglob("*.png")) + list(Path(data_path).rglob("*.jpg"))
    paths.sort()
    
    total = len(paths)
    passed = 0
    
    total_time = 0
    
    with open("evr_test_log.txt", "w") as file:
        file.write("")
    
    with open("evr_test_log.txt", "a", buffering=1) as file:
        i = 0
        for path in paths:
            i += 1
            print(path, f"{i}/{total}")
            # with open("ts.txt", 'a') as f: f.write(f"{path}\n")
            
            # img = cv2.imread(str(path))

            start = time.time()
            
            run = subprocess.run(f'.\\"build\\Release\\qrscanner_cli.exe" {path} --seed 1', shell=True, capture_output=True, text=True)
            res = run.stdout.strip()
                        
            fin = time.time() - start
            
            res = res.replace("\n", " ")

            print(res)
            
            if res is None:
                res = "None"
            elif res != "None" and not "CTF" in res:
                passed += 1
                # with open("ts.txt", 'a') as f: f.write(f"OK\n") 
                
            
            file.write(f"{path.name + ' ' * (30 - len(path.name))} -> {res} -> {fin}\n")
            file.flush()
            total_time += fin

    print(f"passed {passed}/{total}")
    print(f"time: {total_time/total}")
