from PIL import Image
from pathlib import Path
from QRscanner import detectAndDecode
from decoder import decode
from find_qr import zxing_check

def test(data_path):
    paths = list(Path(data_path).rglob("*.png"))
    paths.sort()
    
    total = len(paths)
    passed = 0
    
    with open("evr_test_log.txt", "w") as file:
        file.write("")
    
    with open("evr_test_log.txt", "a", buffering=1) as file:
        
        for path in paths:
            print(path)
            
            img = Image.open(path)
            res = detectAndDecode(img)
            # res = decode(img)
            # res = zxing_check(img)
            
            if res is None:
                res = "None"
            elif res[:5] == "GANYU":
                passed += 1
            
            file.write(f"{path.name + ' ' * (30 - len(path.name))} -> {res}\n")
            file.flush()

    print(f"passed {passed}/{total}")

if __name__ == "__main__":
    
    test("C:/Users/user/Documents/Projects/QRscanner/data orig")