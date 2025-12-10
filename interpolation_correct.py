import numpy as np
from PIL import Image
import data
from collections import deque
from homographic_correct import HomographyMatrix
import copy
import time
from tools import *


class Spline():
    
    def __init__(self, x1, y1, d1, x2, y2, d2):
        
        A = np.array([[x1 ** 3, x1 ** 2, x1, 1],
                    [x2 ** 3, x2 ** 2, x2, 1],
                    [3 * x1 ** 2, 2 * x1, 1, 0],
                    [3 * x2 ** 2, 2 * x2, 1, 0]])
        
        b = np.array([y1, y2, d1, d2])
        
        try:
            self.s = np.linalg.inv(A) @ b
        except:
            self.s = np.zeros(4)

    def predict(self, x):
        return self.s[0] * x**3 + self.s[1] * x**2 + self.s[2] * x + self.s[3]


class Cubic():
    
    def __init__(self, x1, y1, x2, y2, x3, y3, x4, y4):
        
        A = np.array([[x1 ** 3, x1 ** 2, x1, 1],
                      [x2 ** 3, x2 ** 2, x2, 1],
                      [x3 ** 3, x3 ** 2, x3, 1],
                      [x4 ** 3, x4 ** 2, x4, 1]])
        
        b = np.array([y1, y2, y3, y4])
        
        try:
            self.s = np.linalg.inv(A) @ b
        except:
            self.s = np.zeros(4)

    def predict(self, x):
        return self.s[0] * x**3 + self.s[1] * x**2 + self.s[2] * x + self.s[3]


class Conic():
    
    def __init__(self, x1, y1, x2, y2, x3, y3):
        
        A = np.array([[x1 ** 2, x1, 1],
                      [x2 ** 2, x2, 1],
                      [x3 ** 2, x3, 1]])
        
        b = np.array([y1, y2, y3])
        
        try:
            self.s = np.linalg.inv(A) @ b
        except:
            self.s = np.zeros(3)
            
        self.mx = max(x1, x2, x3)

    def predict(self, x, line_con = True):
        # return np.sign(self.s[2])
        if x > self.mx and line_con:
            k = 2 * self.s[0] * self.mx + self.s[1]
            return x * k + self.predict(self.mx) - self.mx * k
        return self.s[0] * x**2 + self.s[1] * x + self.s[2]
    
    def __str__(self):
        return f"{self.s[0]}x^2+{self.s[1]}x+{self.s[2]}"


class Linear():
    
    def __init__(self, x1, y1, x2, y2):
        
        A = np.array([[x1, 1],
                      [x2, 1]])
        
        b = np.array([y1, y2])
        
        try:
            self.s = np.linalg.inv(A) @ b
        except:
            self.s = np.zeros(2)

        self.mx = max(x1, x2)

    def predict(self, x):
        # if x > self.mx:
        #     return self.predict(self.mx)
        return self.s[0] * x + self.s[1]
    
    def __str__(self):
        return f"{self.s[0]}x+{self.s[1]}"

class StretchDist():
    
    def __init__(self, dist, x1, y1, x2, y2, x3, y3):
        self.dist = dist
        
        self.con = Conic(x1, y1 / dist[x1], x2, y2 / dist[x2], x3, y3 / dist[x3])
    
    def predict(self, x):
        return self.dist[x] * self.con.predict(x, False)


class StretchDistLine():
    
    def __init__(self, dist, x1, y1, x2, y2):
        self.dist = dist
        
        self.con = Linear(x1, y1 / self.line_dist__(dist, x1), x2, y2 / self.line_dist__(dist, x2))
    
    def line_dist__(self, dist, x):
        x1 = int(x)
        x2 = x1 + 1
        
        if x2 >= len(dist): return dist[-1]
        
        return dist[x1] * (x2 - x) + dist[x2] * (x - x1)
    
    def predict(self, x):
        return self.dist[x] * self.con.predict(x)


class Curve():
    
    def __init__(self, p1, p2, p3):
        if self.__colleniar(p1, p2, p3):
            self._type = "line"
            
            self.P0 = p1
            
            self.N = (p2 - p1) / np.linalg.norm(p2 - p1)
        else:
            self._type = "circle"
            
            A = np.array([[2 * (p1[0] - p2[0]), 2 * (p1[1] - p2[1])],
                          [2 * (p1[0] - p3[0]), 2 * (p1[1] - p3[1])]])
            
            b = np.array([p1[0] ** 2 - p2[0] ** 2 + p1[1] ** 2 - p2[1] ** 2,
                          p1[0] ** 2 - p3[0] ** 2 + p1[1] ** 2 - p3[1] ** 2])

            self.O = np.linalg.inv(A) @ b
            
            self.R = np.linalg.norm(self.O - p1)
            
      
    def __colleniar(self, p1, p2, p3):
        area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        return area < 1e-10
    
    
    def dist(self, p1, p2):
        if self._type == "line":
            
            p1 = np.dot(p1 - self.P0, self.N)
            p2 = np.dot(p2 - self.P0, self.N)
            
            return p1 - p2
        
        elif self._type == "circle":
            
            p1 = (p1 - self.O) / np.linalg.norm(p1 - self.O)
            p2 = (p2 - self.O) / np.linalg.norm(p2 - self.O)

            p1 = np.arctan2(p1[1], p1[0])
            p2 = np.arctan2(p2[1], p2[0])
            
            a = (p1 - p2) % (2 * np.pi)

            if a > np.pi: a -= 2 * np.pi
            return a * self.R

    
    def move(self, p, d):
        if self._type == "line":
            
            p = np.dot(p - self.P0, self.N)
            
            return self.P0 + self.N * (p + d)
        
        elif self._type == "circle":
            d /= self.R
            
            p = (p - self.O) / np.linalg.norm(p - self.O)
            p = np.arctan2(p[1], p[0])
            p += d
            
            return np.array([np.cos(p), np.sin(p)]) * self.R + self.O
    
    def __str__(self):
        if self._type == "line":
            return f"(x-{self.P0[0]})*{self.N[1]}=(y-{self.P0[1]})*{self.N[0]}"
        elif self._type == "circle":
            return f"(x-{self.O[0]})^2+(y-{self.O[1]})^2={self.R}^2"
            

def CurveIntersection(cur1, cur2):
    if cur1._type == "circle" and cur2._type == "circle":
        p1 = cur1.O
        p2 = cur2.O
        r1 = cur1.R
        r2 = cur2.R
        
        v = p2 - p1
        d = np.linalg.norm(v)
        
        if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
            return []
        
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h = np.sqrt(r1**2 - a**2)
        
        p0 = p1 + a * v / d
        v_perp = np.array([-v[1], v[0]]) * h / d
        
        return [p0 + v_perp, p0 - v_perp]

    if cur1._type == "line" and cur2._type == "circle":
        P0 = cur1.P0
        N = cur1.N
        O = cur2.O
        R = cur2.R
    
        v = O - P0
    
        proj = np.dot(v, N)
        
        perp = v - proj * N
        dist_to_line = np.linalg.norm(perp)
        
        if dist_to_line > R:
            return []
        else:
            half_chord = np.sqrt(R**2 - dist_to_line**2)
            
            closest_point = P0 + proj * N
            
            p1 = closest_point - half_chord * N
            p2 = closest_point + half_chord * N
            
            return [p1, p2]
    
    if cur1._type == "circle" and cur2._type == "line":
        return CurveIntersection(cur2, cur1)

    if cur1._type == "line" and cur2._type == "line":
        if abs(np.dot(cur1.N, cur2.N)) > 1 - 1e-10:
            return []
        
        A = np.array([cur1.N, -cur2.N]).T
        b = cur2.P0 - cur1.P0
        t = np.linalg.solve(A, b)[0]
        return [cur1.P0 + t * cur1.N]


def constructCubic(curve, p0, points, ns):
    
    cub = Cubic(ns[0], curve.dist(points[0], p0),
                ns[1], curve.dist(points[1], p0),
                ns[2], curve.dist(points[2], p0),
                ns[3], curve.dist(points[3], p0))

    return cub


def constructSpline(curve, p0, points, ns):
    # return constructCubic(curve, p0, points, ns)
    
    cub = Spline((ns[0] + ns[1]) / 2, (curve.dist(points[0], p0) + curve.dist(points[1], p0)) / 2, (curve.dist(points[0], p0) - curve.dist(points[1], p0)) / (ns[0] - ns[1]),
                 (ns[2] + ns[3]) / 2, (curve.dist(points[2], p0) + curve.dist(points[3], p0)) / 2, (curve.dist(points[2], p0) - curve.dist(points[3], p0)) / (ns[2] - ns[3]))

    return cub

def constructConic(curve, p0, points, ns):
    
    con = Conic(ns[0], curve.dist(points[0], p0),
                ns[1], curve.dist(points[1], p0),
                ns[2], curve.dist(points[2], p0))
    return con

def constructDist(curve, p0, points, pattern):

    dist_pattren = [curve.dist(i, p0) for i in pattern]
    d1 = [(curve.dist(points[0], p0) * (7 - i) + curve.dist(points[1], p0) * i) / 7 for i in range(8)]
    d2 = [(curve.dist(points[2], p0) * (7 - i) + curve.dist(points[3], p0) * i) / 7 for i in range(8)]
    
    return d1 + dist_pattren + d2

def constructStretchLine(curve, dist, p0, points, ns):
    
    con = StretchDistLine(dist, ns[1], curve.dist(points[1], p0),
                                ns[2], curve.dist(points[2], p0))
    return con


def constructStretch(curve, dist, p0, points, ns):    
    con = StretchDist(dist, ns[1], curve.dist(points[1], p0),
                            ns[2], curve.dist(points[2], p0),
                            ns[3], curve.dist(points[3], p0))
    return con

def containnedInPol(pol, p):
    def ray(p1, p2, p):
        if (p[0] - p1[0]) * (p[0] - p2[0]) >= 0:
            return 0
        
        k = (p1[1] - p2[1]) / (p1[0] - p2[0])
        b = p1[1] - p1[0] * k
        if k * p[0] + b > p[1]:
            return 1
        return 0

    s = 0
    for i in range(len(pol)):
        s += ray(pol[i], pol[(i + 1) % len(pol)], p)
    
    return bool(s % 2)


def averageCol(img, pol):
    
    pixels = img.load()
    
    s = (pol[0] + pol[1] + pol[2] + pol[3]) / 4
    s = (int(s[0]), int(s[1]))
    q = deque()
    q.append(s)
    used = set()
    
    dir = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    a = []
    
    while q:
        x, y = q.popleft()
        
        a.append(pixels[x, y])
        
        for dx, dy in dir:
            p = (x + dx, y + dy)
            if containnedInPol(pol, p) and not p in used:
                used.add(p)
                q.append(p)
    
    
    sm = (0, 0, 0)
    for p in a:
        sm = (sm[0] + p[0], sm[1] + p[1], sm[2] + p[2])

    return (sm[0] // len(a), sm[1] // len(a), sm[2] // len(a))



def interpolation_correct_homo(img, points, pattern = [None, None], QRsize = 45, square_size = 10, fract = 1, persp = None):
    
    start_time = time.time()
    
    def range_(c, f = fract):
        a = [i for i in range(0, c - 1, f)]
        a.append(c - 1)
        return a

    
    # points = data.predsetP2sq4_better
    # points = data.predsetP1sq4
    
    points_mark = list(points[:])
    points_show = []

    points3 = [(points[12] * 13 + points[15]) / 14,
               (points[13] * 13 + points[14]) / 14,
               center([points[8], points[9], points[10], points[11]])]
    
                
    cur1 = Curve(points[3], points[2], points[6])
    cur2 = Curve(points[0], points[1], points[5])
    # cur3 = Curve(points[12], points[13], points[9] * 2/3 + points[10] / 3)
    cur3 = Curve(*points3)
    
    
    if pattern[0] is not None:
        try:
            p = [(points[0] + points[3] * 13) / 14,
                (points[1] + points[2] * 13) / 14,
                (points[4] + points[7] * 13) / 14,
                (points[5] + points[6] * 13) / 14]
            dist = constructDist(Curve(p[0], p[1], p[3]), p[0], p, pattern[0])
            cub1 = constructStretch(cur1, dist, points[3], [points[3], points[2], points[7], points[6]], [0, 7, QRsize - 7, QRsize])
            cub2 = constructStretch(cur2, dist, points[0], [points[0], points[1], points[4], points[5]], [0, 7, QRsize - 7, QRsize])
            # cub3 = constructStretchLine(cur3, dist, points[12], [points[12], points[13], points[8] * 2/3 + points[11] / 3, points[9] * 2/3 + points[10] / 3], [0, 7, QRsize - 8, QRsize - 5])
            cub3 = constructStretchLine(cur3, dist, points3[0], points3, [0, 7, QRsize - 6.5])
        except:
            pattern[0] = None
    
    if pattern[0] is None:
        cub1 = constructSpline(cur1, points[3], [points[3], points[2], points[7], points[6]], [0, 7, QRsize - 7, QRsize])
        cub2 = constructSpline(cur2, points[0], [points[0], points[1], points[4], points[5]], [0, 7, QRsize - 7, QRsize])
        # cub3 = constructSpline(cur3, points[12], [points[12], points[13], points[8] * 2/3 + points[11] / 3, points[9] * 2/3 + points[10] / 3], [0, 7, QRsize - 8, QRsize - 5])
        # cub3 = constructConic(cur3, points[12], [points[12], points[13], points[8] * 2/3 + points[11] / 3, points[9] * 2/3 + points[10] / 3], [0, 7, QRsize - 8, QRsize - 5])
        cub3 = constructConic(cur3, points3[0], points3, [0, 7, QRsize - 6.5])
    
    points6 = [(points[4] * 13 + points[5]) / 14,
               (points[7] * 13 + points[6]) / 14,
               center([points[8], points[9], points[10], points[11]])]
    
    
    cur4 = Curve(points[1], points[2], points[14])
    cur5 = Curve(points[0], points[3], points[15])
    # cur6 = Curve(points[4], points[7], points[11] * 2/3 + points[10] / 3)
    cur6 = Curve(*points6)
    
    if pattern[1] is not None:
        try:
            p = [(points[0] + points[1] * 13) / 14,
                (points[3] + points[2] * 13) / 14,
                (points[12] + points[13] * 13) / 14,
                (points[15] + points[14] * 13) / 14]
            dist = constructDist(Curve(p[0], p[1], p[3]), p[0], p, pattern[1])
            cub4 = constructStretch(cur4, dist, points[1], [points[1], points[2], points[13], points[14]], [0, 7, QRsize - 7, QRsize])
            cub5 = constructStretch(cur5, dist, points[0], [points[0], points[3], points[12], points[15]], [0, 7, QRsize - 7, QRsize])
            # cub6 = constructStretchLine(cur6, dist, points[4], [points[4], points[7], points[8] * 2/3 + points[9] / 3, points[11] * 2/3 + points[10] / 3], [0, 7, QRsize - 8, QRsize - 5])
            cub6 = constructStretchLine(cur6, dist, points6[0], points6, [0, 7, QRsize - 6.5])
        except:
            pattern[1] = None
    
    if pattern[1] is None:
        cub4 = constructSpline(cur4, points[1], [points[1], points[2], points[13], points[14]], [0, 7, QRsize - 7, QRsize])
        cub5 = constructSpline(cur5, points[0], [points[0], points[3], points[12], points[15]], [0, 7, QRsize - 7, QRsize])
        # cub6 = constructSpline(cur6, points[4], [points[4], points[7], points[8] * 2/3 + points[9] / 3, points[11] * 2/3 + points[10] / 3], [0, 7, QRsize - 8, QRsize - 5])
        # cub6 = constructConic(cur6, points[4], [points[4], points[7], points[8] * 2/3 + points[9] / 3, points[11] * 2/3 + points[10] / 3], [0, 7, QRsize - 8, QRsize - 5])        
        cub6 = constructConic(cur6, points6[0], points6, [0, 7, QRsize - 6.5])        

    # print(cub3.con)
    # print(cub6.con)

    grade = [[], []]

    for i in range_(QRsize + 1):

        grade[0].append(Curve(cur1.move(points[3], cub1.predict(i)),
                              cur2.move(points[0], cub2.predict(i)),
                              cur3.move(points[12], cub3.predict(i))))
        
        grade[1].append(Curve(cur4.move(points[1], cub4.predict(i)),
                              cur5.move(points[0], cub5.predict(i)),
                              cur6.move(points[4], cub6.predict(i))))
        
        points_show.append(cur1.move(points[3], cub1.predict(i)))
        points_show.append(cur2.move(points[0], cub2.predict(i)))
        points_show.append(cur3.move(points[12], cub3.predict(i)))
        points_show.append(cur4.move(points[1], cub4.predict(i)))
        points_show.append(cur5.move(points[0], cub5.predict(i)))
        points_show.append(cur6.move(points[4], cub6.predict(i)))
    
    # show_points(img, points_show, 1)
    # print(pattern[0])
    # print(pattern[1])
    
    rg = range_(QRsize + 1)
    
    nodes = [[None for _ in range_(QRsize + 1)] for _ in range_(QRsize + 1)]
    
    for x in range(len(rg)):
        for y in range(len(rg)):
            if x == 0 and y == 0:
                last = points[0]
            elif x == 0:
                last = nodes[x][y - 1]
            else:
                last = nodes[x - 1][y]
            
            p = CurveIntersection(grade[0][x], grade[1][y])
            
            if p == []:
                nodes[x][y] = (0, 0)
                continue
            if len(p) == 1: p.append(p[0])
            # print(x, y, last, p)
            
            if np.linalg.norm(last - p[0]) < np.linalg.norm(last - p[1]):
                nodes[x][y] = p[0]
            else:
                nodes[x][y] = p[1]
            # print_as_point(nodes[x][y])
    
    
    transform = [[None for _ in range(len(rg) - 1)] for _ in range(len(rg) - 1)]
    
    for x in range(len(rg) - 1):
        for y in range(len(rg) - 1):
            fr = [(rg[x] * square_size, rg[y] * square_size),
                  (rg[x + 1] * square_size, rg[y] * square_size),
                  (rg[x] * square_size, rg[y + 1] * square_size),
                  (rg[x + 1] * square_size, rg[y + 1] * square_size)]
            to = [nodes[x][y],
                  nodes[x + 1][y],
                  nodes[x][y + 1],
                  nodes[x + 1][y + 1]]
            
            transform[x][y] = HomographyMatrix(fr, to)

    
    margins = square_size * 2
    
    corrected = Image.new("RGB", (square_size * QRsize + 2 * margins, square_size * QRsize + 2 * margins))
    
    pixels = corrected.load()
    orig_pixels = img.load()
    
    weight, height = corrected.size
    
    def comp(a, mn = 0, mx = len(rg) - 2):
        if a < mn: a = mn
        if a > mx: a = mx
        return a

    for x in range(weight):
        for y in range(height):
            cur_x = x - margins
            cur_y = y - margins
            # if persp is not None:
            #     cur_x, cur_y = persp.predict((cur_x, cur_y))
            try:
                pixels[x, y] = orig_pixels[*transform[comp(cur_x // square_size // fract)][comp(cur_y // square_size // fract)].predict((cur_x, cur_y))]
            except:
                pixels[x, y] = (0, 0, 0)

    print("ready in", time.time() - start_time)
    
    img = img.copy()
    
    pixels = img.load()
    
    for x in range(QRsize + 1):
        for y in range(QRsize + 1):
            try:
                pixels[int(nodes[x][y][0] + 0.5), int(nodes[x][y][1]) + 0.5] = (255, 0, 0)
            except:
                pass
        
    for x, y in points_mark:
        try:
            pixels[x, y] = (0, 255, 0)
        except:
            pass
    for x, y in points_show:
        try:
            pixels[x, y] = (0, 0, 255)
        except:
            pass
        
    img.save("marked.png")
    
    return corrected


if __name__ == "__main__":
    
    c = Conic(0, 0, 1, 2, 2, 3)
    
    for i in range(-20, 60):
        print_as_point([i / 10, c.predict(i / 10)])