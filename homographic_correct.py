from PIL import Image
import numpy as np
from tools import *

RED = np.array([255, 0, 0, 255])
BLUE = np.array([0, 255, 0, 255])
GREEN = np.array([0, 0, 255, 255])


class HomographyMatrix():
    
    def __init__(self, fr, to):
        A = []
        B = []
        
        for (x, y), (x1, y1) in zip(fr, to):
                    
            A.append([x, y, 1, 0, 0, 0, -x * x1, -y * x1])
            A.append([0, 0, 0, x, y, 1, -x * y1, -y * y1])
            
            B.append(x1)
            B.append(y1)
        
        A = np.array(A)
        B = np.array(B)
        
        try:
            h = np.linalg.inv(A) @ B
        except:
            h = np.array([1, 0, 0, 0, 1, 0, 0, 0])
        
        self.H = np.array([[h[0], h[1], h[2]],
                           [h[3], h[4], h[5]],
                           [h[6], h[7],    1]])

    def predict(self, point):
        A = np.array([point[0], point[1], 1])
    
        B = self.H @ A
        if B[2] == 0:
            return np.array((0, 0))
        B /= B[2]
        
        return np.array([B[0], B[1]])


def find_H(current, expected):
    
    A = []
    B = []
    
    for (x, y), (x1, y1) in zip(current, expected):
                
        A.append([x, y, 1, 0, 0, 0, -x * x1, -y * x1])
        A.append([0, 0, 0, x, y, 1, -x * y1, -y * y1])
        
        B.append(x1)
        B.append(y1)
    
    A = np.array(A)
    B = np.array(B)
    
    try:
        h = np.linalg.inv(A) @ B
    except:
        h = np.array([1, 0, 0, 0, 1, 0, 0, 0])
    
    H = np.array([[h[0], h[1], h[2]],
                  [h[3], h[4], h[5]],
                  [h[6], h[7],    1]])
    
    return H


def F(point, H):
    A = np.array([point[0], point[1], 1])
    
    B = H @ A
    B /= B[2]
    
    return (B[0], B[1])


def dist(p1, p2):
    if p1 == p2:
        return 1e-10
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 2

def h(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    a = p1 - p2
    b = p3 - p2
    return -(a[0] * b[1] - a[1] * b[0]) / np.linalg.norm(b)

def h1(p1, p2, p3):
    x0, y0 = p1
    x1, y1 = p2
    x2, y2 = p3
    
    A_x = y2 - y1
    B_x = -(x2 - x1)
    C_x = x2 * y1 - x1 * y2
    
    numerator = abs(A_x * x0 + B_x * y0 + C_x)
    denominator = np.sqrt(A_x**2 + B_x**2)
    
    height = numerator / denominator
    return height


def baric(a, b, c, x):
    A = np.array([[a[0], b[0], c[0]],
                  [a[1], b[1], c[1]],
                  [   1,    1,    1]])
    b = np.array([x[0], x[1], 1])
    try:
        return np.linalg.inv(A) @ b
    except:
        return np.array([1, 1, 1])


def predict(point, H1, p1, H2, p2, H3, p3):
    
    # d1 = dist(point, p1)
    # d2 = dist(point, p2)
    # d3 = dist(point, p3)
    
    # w1 = 1 / d1
    # w2 = 1 / d2
    # w3 = 1 / d3
    
    # w2 = point[0] - p1[0]
    # w1 = -point[0] + p2[0]
    # w3 = 0
    
    # w1 = h(point, p2, p3) ** 2
    # w2 = h(point, p3, p1) ** 2
    # w3 = h(point, p1, p2) ** 2
    
    w1, w2, w3 = baric(p1, p2, p3, point)
    
    H = (H1 * w1 + H2 * w2 + H3 * w3) / (w1 + w2 + w3)
    
    return F(point, H)

def col(point, p1, p2, p3):
    
    w1 = h(point, p2, p3)
    w2 = h(point, p3, p1)
    w3 = h(point, p1, p2)
    
    w1, w2, w3 = baric(p1, p2, p3, point)
    
    if point == (1023, 1023):
        print(w1, w2, w3)
    
    # if min(abs(w1), abs(w2), abs(w3)) < 1:
    #     return (0, 0, 0, 255)
    
    c = (RED * w1 + BLUE * w2 + GREEN * w3) / (w1 + w2 + w3)
    # c /= 2
    # c += [128, 128, 128, 128]
    c = map(int, c)
    
    return tuple(c)
    

def correct(orig):
    
    current = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])
    current *= 175
    # current += [20, 20]
    
    expected1 = [(86, 36), (220, 30), (211, 158), (70, 168)]
    H1 = find_H(current, expected1)
    
    current += [1023 - 175, 0]
    expected2 = [[725, 47], [846, 58], [870, 187], [746, 172]]
    H2 = find_H(current, expected2)
    
    current += [-(1023 - 175), 1023 - 175]
    expected3 = [[21, 723], [171, 709], [172, 860], [23, 874]]
    H3 = find_H(current, expected3)
    
    
    orig = orig.convert("RGBA")
    image = Image.new("RGBA", (1200, 1200))
    orig_pixels = orig.load()
    pixels = image.load()

    weight, height = image.size

    for x in range(weight):
        for y in range(height):
            
            # pixels[x, y] = col((x, y), (0, 0), (1023, 0), (0, 1023))
            # continue
            
            try:
                pixels[x, y] = orig_pixels[*predict((x - 20, y - 20), H1, (0, 0), H2, (1023, 0), H3, (0, 1023))]
                # pixels[x, y] = orig_pixels[*F((x, y), H3)]
            except:
                pixels[x, y] = (0, 0, 0, 0)
    
    return image


def homographic_correct_perspective(img, points, size = 300):
    
    fr = [(points[0] + points[1] * 13 + points[2] * 169 + points[3] * 13) / 196,
          (points[5] + points[6] * 13 + points[7] * 169 + points[4] * 13) / 196,
          center([points[8], points[9], points[10], points[11]]),
          (points[15] + points[12] * 13 + points[13] * 169 + points[14] * 13) / 196]
    
    # show_points(img, fr)
    
    to = [[size / 3, size / 3],
          [size * 2 / 3, size / 3],
          [size * 2 / 3, size * 2 / 3],
          [size / 3, size * 2 / 3]]
    
    H = HomographyMatrix(fr, to)
    H_ = HomographyMatrix(to, fr)
    
    corrected = Image.new("RGB", (size, size))
    
    pixels = corrected.load()
    orig_pixels = img.load()
    
    for x in range(size):
        for y in range(size):
            try:
                pixels[x, y] = orig_pixels[*H_.predict((x, y))]
            except:
                pixels[x, y] = (0, 0, 0)
    
    points_corrected = [np.array(H.predict(i)) for i in points]
    
    return points_corrected, corrected


def find_homo(points, size = 2000):
    fr = [(points[0] + points[1] * 13 + points[2] * 169 + points[3] * 13) / 196,
          (points[5] + points[6] * 13 + points[7] * 169 + points[4] * 13) / 196,
          center([points[8], points[9], points[10], points[11]]),
          (points[15] + points[12] * 13 + points[13] * 169 + points[14] * 13) / 196]
    
    # show_points(img, fr)
    
    to = [[size / 3, size / 3],
          [size * 2 / 3, size / 3],
          [size * 2 / 3, size * 2 / 3],
          [size / 3, size * 2 / 3]]
    
    H = HomographyMatrix(fr, to)
    H_ = HomographyMatrix(to, fr)
    
    points_corrected = [np.array(H.predict(i)) for i in points]
    
    return points_corrected, H_


if __name__ == "__main__":
    
    # current = np.array([(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
    # expected = [[725, 47], [846, 58], [870, 187], [746, 172]]
    # H = find_H(current, expected)
    # print(H)
    # for i in range(4):
    #     current[i] = F(current[i], H)
    # print(current)
    
    # quit()
    
    orig = Image.open("orig_bynarized.png")
    # orig = Image.open("grid.png")
    orig = correct(orig)
    orig.save("corrected.png")