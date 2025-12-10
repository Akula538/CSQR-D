import cv2
import numpy as np

def show_points(img, points=[], size = 5):
    rgb_array = np.array(img)
    paint = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    for i, (x, y) in enumerate(points):
        cv2.circle(paint, (int(x), int(y)), size, (0, 255, 0), -1)
    
    cv2.imshow('Points', paint)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def print_as_point(p):
    print(f"({p[0]},{p[1]})")
    

def center(v):
    s = np.array([0, 0])
    for p in v:
        s = s + p
    return s / len(v)

def cross(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]