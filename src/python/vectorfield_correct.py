import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import convolve1d
from PIL import Image, ImageDraw
from tools import*
import time
from homographic_correct import HomographyMatrix
from TPS_correct import TPS
from shapely.geometry import LineString


def upsample(gray, scale=2):
    gray = gray.astype(np.float32)
    if gray.max() > 1.5:
        gray /= 255.0

    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.7)
    
    gray_up = cv2.resize(
        gray, None,
        fx=scale, fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    gray_up = cv2.GaussianBlur(gray_up, (0, 0), sigmaX=0.5)

    return gray_up

class DirectionField():
    def __init__(self, img, show_flag = False):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = img.copy()
        
        gray = cv2.GaussianBlur(gray, (1, 1), 0)
        # show_points(gray, [])
            
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=7)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=7)
        
        self.magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        self.orientation = np.arctan2(sobel_y, sobel_x)
        
        self.show_flag = show_flag
        
        self.cache = {}
        self.lucky = []
        self.dislucky = []
        
        # if gray.dtype == np.uint8:
        #     gray_f = gray.astype(np.float32) / 255.0
        # else:
        #     gray_f = gray.astype(np.float32)
            
        # sobel_kernel = 3
        # Ix = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
        # Iy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=sobel_kernel)

        # Jxx = Ix * Ix
        # Jyy = Iy * Iy
        # Jxy = Ix * Iy

        # tensor_sigma = 4
        # Jxx = cv2.GaussianBlur(Jxx, ksize=(0, 0), sigmaX=tensor_sigma)
        # Jyy = cv2.GaussianBlur(Jyy, ksize=(0, 0), sigmaX=tensor_sigma)
        # Jxy = cv2.GaussianBlur(Jxy, ksize=(0, 0), sigmaX=tensor_sigma)

        # self.tensor_orientation = 0.5 * np.arctan2(2.0 * Jxy, (Jxx - Jyy + 1e-12))  # radians in (-pi/2, pi/2)

        # numer = np.sqrt((Jxx - Jyy)**2 + 4.0 * (Jxy**2))
        # denom = (Jxx + Jyy) + 1e-12
        # self.coherence = (numer / denom).astype(np.float32)  # in [0,1] roughly
    
    def initGrid(self, kernel = 35):
        # kernel //= 1
        ym, xm = self.magnitude.shape
        
        self.grid = [[[] for i in range(int(ym // kernel + 2))] for j in range(int(xm // kernel + 2))]
        
        for x in range(0, int(xm // kernel + 1)):
            for y in range(0, int(ym // kernel + 1)):
                dist1, dist2 = self.get(min(xm, x * kernel), min(ym, y * kernel), kernel)
                if dist1 is not None:
                    if dist2 is not None:
                        self.grid[x][y] = [dist1, dist2]
                    else:
                        self.grid[x][y] = [dist1]
        self.kernel = kernel
    
    
    def get(self, x, y, kernel = 35):
        x = int(x)
        y = int(y)
        kernel = int(kernel)
        
        if (x, y, kernel) in self.cache:
            return self.cache[(x, y, kernel)]
        
        ym, xm = self.magnitude.shape
        
        x1 = max(0, x - kernel // 2)
        y1 = max(0, y - kernel // 2)
        x2 = min(xm - 1, x - kernel // 2 + kernel)
        y2 = min(ym - 1, y - kernel // 2 + kernel)
        
        roi_magnitude = self.magnitude[y1:y2, x1:x2]
        roi_orientation = self.orientation[y1:y2, x1:x2]
        
        # roi_magnitude = self.coherence[y1:y2, x1:x2]
        # roi_orientation = self.tensor_orientation[y1:y2, x1:x2]
        
        threshold = np.percentile(roi_magnitude, 85)
        # threshold = 0.3
        
        mask = roi_magnitude > threshold
        
        strong_orientations = roi_orientation[mask]
        strong_magnitudes = roi_magnitude[mask]
        
        line_orientations = (np.degrees(strong_orientations) + 45) % 180 - 45
        
        hist_counts, bins = np.histogram(line_orientations, bins=180, range=(-45, 135), weights=strong_magnitudes)
        bins_centers = bins[:-1]
        
        sigma = 2.0
        kernel_size = int(2 * 3 * sigma) + 1
        xx = np.arange(-kernel_size//2, kernel_size//2 + 1)
        gaussian_kernel = np.exp(-xx**2 / (2 * sigma**2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        smoothed_counts = convolve1d(hist_counts, gaussian_kernel, mode='wrap')
        
        # print(smoothed_counts)
        
        #========================================
        if self.show_flag:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(14, 6))

            # Исходная гистограмма
            # plt.subplot(1, 2, 1)
            plt.bar(bins_centers, hist_counts, width=1, alpha=0.7, edgecolor='black')
            plt.xlabel('Направление градиента (градусы)')
            # plt.ylabel('Количество точек')
            # plt.title('Исходная гистограмма направлений')
            plt.grid(True, alpha=0.3)
            plt.xlim(-45, 135)

            # Сглаженная гистограмма
            # plt.subplot(1, 2, 2)
            # plt.bar(bins_centers, smoothed_counts, width=1, alpha=0.7, color='red', edgecolor='black')
            # plt.plot(bins_centers, smoothed_counts, 'k-', linewidth=2, label='Сглаженная кривая')
            # plt.xlabel('Направление градиента (градусы)')
            # plt.ylabel('Количество точек (сглаженное)')
            # plt.title(f'Сглаженная гистограмма (σ={sigma})')
            # plt.grid(True, alpha=0.3)
            # plt.xlim(0, 180)
            # plt.legend()

            plt.tight_layout()
            plt.show()
        
        #========================================
        
        pad = 45
        extended = np.concatenate([smoothed_counts[-pad:], smoothed_counts, smoothed_counts[:pad]])
        
        peaks_ext, props = find_peaks(extended,
                               height=0.1*np.max(smoothed_counts),
                               distance=25)
        N = len(smoothed_counts)
        peaks_in_center = [p - pad for p in peaks_ext if pad <= p < pad + N]
        
        if len(peaks_in_center) < 1:
            return None, None
        
        peaks = [(smoothed_counts[i], bins_centers[i]) for i in peaks_in_center]

        peaks.sort()
        
        def circ_dist(a, b):
            d = abs(a - b) % 180.0
            return min(d, 180.0 - d)

        if len(peaks) >= 2:
            h1, a1 = peaks[-1]
            h2, a2 = peaks[-2]
            if circ_dist(a1, a2) < 25:
                peaks.pop()
        
        # print(peaks)
        if len(peaks) == 1:
            self.cache[(x, y, kernel)] = (peaks[-1][1], None)
            return peaks[-1][1], None
        else:
            self.cache[(x, y, kernel)] = (peaks[-1][1], peaks[-2][1])
            return peaks[-1][1], peaks[-2][1]
    
    def getBiliniar(self, x, y):
        def circ_dist(a, b):
            d = abs(a - b) % 180.0
            return min(d, 180.0 - d)
        
        ym, xm = self.magnitude.shape
        
        x = min(xm - 1, x)
        x = max(0, x)
        y = min(ym - 1, y)
        y = max(0, y)
        
        dirs1 = []
        dirs2 = []
        
        for dx in range(2):
            for dy in range(2):
                # print("size", len(self.grid), len(self.grid[0]))
                # print(int(x // self.kernel + dx), int(y // self.kernel + dy))
                d = self.grid[int(x // self.kernel + dx)][int(y // self.kernel + dy)]
                if len(d) < 1: continue
                
                thrashold = 30
                if dirs1 == [] or max(circ_dist(d[0], i) for i in dirs1) < thrashold:
                    dirs1.append(d[0])
                elif dirs2 == [] or max(circ_dist(d[0], i) for i in dirs2) < thrashold:
                    dirs2.append(d[0])
                
                if len(d) < 2: continue
                
                if dirs2 == [] or max(circ_dist(d[1], i) for i in dirs2) < thrashold:
                    dirs2.append(d[1])
                elif dirs1 == [] or max(circ_dist(d[1], i) for i in dirs1) < thrashold:
                    dirs1.append(d[1])
        
        if len(dirs1) < 4 and len(dirs2) < 4:
            return None, None
            self.dislucky.append((x, y))
            return self.get(x, y, self.kernel)
        # print(x, y, dirs1, dirs2)
        
        if dirs2 != [] and max(dirs1) - min(dirs1) > 45:
            for i in range(len(dirs1)):
                if dirs1[i] > 90: dirs1[i] -= 180
        
        if dirs2 != [] and max(dirs2) - min(dirs2) > 45:
            for i in range(len(dirs2)):
                if dirs2[i] > 90: dirs2[i] -= 180
                
        x_ = (x / self.kernel) % 1.0
        y_ = (y / self.kernel) % 1.0
        
        dir1 = None
        dir2 = None
        if len(dirs1) == 4:
            dir1 = dirs1[0] * x_ * y_ + dirs1[1] * x_ * (1 - y_) + dirs1[2] * (1 - x_) * y_ + dirs1[3] * (1 - x_) * (1 - y_)
        if len(dirs2) == 4:
            dir2 = dirs2[0] * x_ * y_ + dirs2[1] * x_ * (1 - y_) + dirs2[2] * (1 - x_) * y_ + dirs2[3] * (1 - x_) * (1 - y_)

        if dir1 is None:
            dir1, dir2 = dir2, dir1
        
        self.lucky.append((x, y))
        
        return dir1, dir2

    
def calc_integral_line(DF : DirectionField, p, dir0, line_len, n, kernel = 35, RK = 1): # RK = 4
    h = line_len / n
    dir0 = dir0 / np.linalg.norm(dir0)
    
    def get_vec(p, v = dir0):
        v = v / np.linalg.norm(v)
        dir1, dir2 = DF.get(p[0], p[1], kernel)
        # dir1, dir2 = DF.getBiliniar(p[0], p[1])
        if dir1 is None:
            return v
        if dir2 is None:
            dir1 += 90
            dir1 = np.radians(dir1)
            v1 = np.array([np.cos(dir1), np.sin(dir1)])
            if abs(v1 @ v) >= 0.85:
                return v1 * np.sign(v1 @ v)
            return v

        dir1 += 90
        dir2 += 90
        dir1 = np.radians(dir1)
        dir2 = np.radians(dir2)
        v1 = np.array([np.cos(dir1), np.sin(dir1)])
        v2 = np.array([np.cos(dir2), np.sin(dir2)])
        
        # if abs(p[0] - 604) < 1 and abs(p[1] - 326) < 1:
        #     print(dir1, dir2, v1, v2)
        #     print(v, abs(v @ v1), abs(v @ v2))
        
        if abs(v1 @ v) > abs(v2 @ v):
            if abs(v1 @ v) >= 0:
                return v1 * np.sign(v1 @ v)
            return v
        if abs(v2 @ v) >= 0:
            return v2 * np.sign(v2 @ v)
        return v
    
    p = np.array(p)

    points = [p]
    v = dir0
    prev_v = v
    for _ in range(n):
        # print(p, dir0, get_vec(p, dir0))
        try:
            if v @ dir0 < -0.5: break
            
            if v @ dir0 > -0.26 and abs(prev_v @ dir0) < 0.99:
                predicted_v = v * 2 - prev_v
            else:
                predicted_v = v
            
            if RK == 4:
                k1 = get_vec(p, predicted_v)
                k2 = get_vec(p + (h/2)*k1, predicted_v)
                k3 = get_vec(p + (h/2)*k2, predicted_v)
                k4 = get_vec(p + h*k3, predicted_v)
                k5 = (1/6) * (k1 + 2*k2 + 2*k3 + k4)
            if RK == 3:
                k1 = get_vec(p, predicted_v)
                k2 = get_vec(p + (h/2)*k1, predicted_v)
                k3 = get_vec(p + h*k2, predicted_v)
                k5 = (1/6) * (k1 + 4*k2 + k3)
            if RK == 2:
                k1 = get_vec(p, predicted_v)
                k2 = get_vec(p + (h/2)*k1, predicted_v)
                k5 = k2
            if RK == 1:
                k1 = get_vec(p, predicted_v)
                k5 = k1
            
            # if abs(p[0] - 600) < 6 and abs(p[1] - 325) < 6:
            #     print(p, k1, k2, k3, k4, k5, v / np.linalg.norm(v))
                # print(p + (h/2)*k1)
            
            p_ = p + h * k5 / np.linalg.norm(k5)
            points.append(p_)
            prev_v = v
            v = p_ - p
            v = v / np.linalg.norm(v)
            p = p_
        # try:pass
        except:
            p_ = p + h * dir0 / np.linalg.norm(dir0)
            points.append(p_)
            break
        
    return points


def complete_pattern(patt, p1, p2, p3, p4):
    d1 = [(p1 * (7 - i) + p2 * i) / 7 for i in range(8)]
    d2 = [(p3 * (7 - i) + p4 * i) / 7 for i in range(8)]
    
    return d1 + patt + d2

class Polyline():
    def __init__(self, points):
        self.points = points
        self._type = "polyline"
        
        self.LineString = LineString(points)
        

def PolylineIntersection(cur1 : Polyline, cur2 : Polyline):
    
    def seg_intersection(p1, p2, p3, p4, eps=1e-9):
        p1, p2, p3, p4 = map(lambda a: np.asarray(a, dtype=float), (p1, p2, p3, p4))
        ab, cd = p2 - p1, p4 - p3
        denom = np.cross(ab, cd)

        if abs(denom) > eps:
            t = np.cross(p3 - p1, cd) / denom
            u = np.cross(p3 - p1, ab) / denom
            if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
                return p1 + t * ab
            return None

        if abs(np.cross(p3 - p1, ab)) > eps:
            return None

        axis = 0 if abs(ab[0]) >= abs(ab[1]) else 1
        a0, a1 = p1[axis], p2[axis]
        b0, b1 = p3[axis], p4[axis]
        a_lo, a_hi = min(a0, a1), max(a0, a1)
        b_lo, b_hi = min(b0, b1), max(b0, b1)
        lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)

        if lo > hi + eps:
            return None

        def point_from_proj(pa, pb, qa, qb, val):
            if abs(qb - qa) <= eps:
                return pa
            t = (val - qa) / (qb - qa)
            return pa + t * (pb - pa)

        if abs(hi - lo) <= eps:
            if a_lo - eps <= lo <= a_hi + eps:
                return point_from_proj(p1, p2, a0, a1, lo)
            return point_from_proj(p3, p4, b0, b1, lo)

        if a_lo + eps >= b_lo and a_hi <= b_hi + eps:
            return 0.5 * (p1 + p2)
        if b_lo + eps >= a_lo and b_hi <= a_hi + eps:
            return 0.5 * (p3 + p4)

        P1 = point_from_proj(p1, p2, a0, a1, lo) if a_lo <= lo <= a_hi else point_from_proj(p3, p4, b0, b1, lo)
        P2 = point_from_proj(p1, p2, a0, a1, hi) if a_lo <= hi <= a_hi else point_from_proj(p3, p4, b0, b1, hi)
        return 0.5 * (P1 + P2)
    
    if cur1._type == cur2._type == "polyline":
        
        return fucking_geom(cur1.LineString.intersection(cur2.LineString))
        
        ans = []
        for i in range(len(cur1.points) - 1):
            for j in range(len(cur2.points) - 1):
                inter = seg_intersection(cur1.points[i], cur1.points[i + 1], cur2.points[j], cur2.points[j + 1])
                if inter is not None:
                    ans.append(inter)
        # print(f"cur1 = {cur1.points}\ncur2 = {cur2.points}\nans = {ans}")
        return ans


def constructPolyline(df, p, dir, QRsize, line_len, n, kernel = 35):
    pol1 = calc_integral_line(df, p, -dir, line_len * (6 / QRsize + 0.5), n * 6 // QRsize + n // 2, kernel)
    pol2 = calc_integral_line(df, p, dir, line_len * ((QRsize - 6) / QRsize + 0.5), n * (QRsize - 6) // QRsize + n // 2, kernel)
    # print(line_len, line_len * ((QRsize - 6) / QRsize + 0.5))
    
    pol1.reverse()
    pol1.pop()
    
    return Polyline(pol1 + pol2)
    

def vectorfield_correct_homo(img, points, pattern = [None, None], QRsize = 25, square_size = 10, fract = 1, lines_fract = 5, tps = None):
    
    start_time = time.time()
    
    def range_(c, f = fract):
        a = [i for i in range(0, c - 1, f)]
        a.append(c - 1)
        return a

    paint = np.array(img)
    paint = cv2.cvtColor(paint, cv2.COLOR_RGB2BGR)

    characteristic_size = max(np.linalg.norm(points[6] - points[3]),
                              np.linalg.norm(points[14] - points[1]))
    kernel_size = max(characteristic_size * 45 // 300, 35)
    if kernel_size % 2 == 0: kernel_size += 1
    print(characteristic_size, kernel_size)
    
    
    # points = data.predsetP2sq4_better
    # points = data.predsetP1sq4
    
    points_mark = list(points[:])
    if pattern[0] is not None: points_mark += list(pattern[0])
    if pattern[1] is not None: points_mark += list(pattern[1])
    points_show = []

    # pattern_comp = [complete_pattern(pattern[0], points[3], points[2], points[7], points[6]),
                    # complete_pattern(pattern[1], points[1], points[2], points[13], points[14])]
    
    pattern_comp = pattern
    
    # show_points(img, list(pattern[0]) + list(pattern[1]))

    df = DirectionField(img)
    
    grade = [[], []]

    for i in range_(QRsize + 1):

        if np.linalg.norm(pattern_comp[1][i] - points[2]) <\
           np.linalg.norm(pattern_comp[1][i] - points[13]):
            dist1 = points[2] - points[3]
        else:
            dist1 = points[13] - points[12]
            
        grade[1].append(constructPolyline(df, pattern_comp[1][i], dist1,
            QRsize, np.linalg.norm(points[6] - points[3]), lines_fract, kernel_size))
        
        if np.linalg.norm(pattern_comp[0][i] - points[2]) <\
           np.linalg.norm(pattern_comp[0][i] - points[7]):
            dist2 = points[2] - points[1]
        else:
            dist2 = points[7] - points[4]
        
        grade[0].append(constructPolyline(df, pattern_comp[0][i], dist2,
            QRsize, np.linalg.norm(points[14] - points[1]), lines_fract, kernel_size))
        
        # points_show += grade[0][-1].points
        # points_show += grade[1][-1].points
        
        zp = grade[0][-1].points
        zp = np.array(zp, dtype=np.int32)
        zp = zp.reshape((-1, 1, 2))
        cv2.polylines(paint, [zp], False, (255, 0, 0))
        zp = grade[1][-1].points
        zp = np.array(zp, dtype=np.int32)
        zp = zp.reshape((-1, 1, 2))
        cv2.polylines(paint, [zp], False, (255, 0, 0))

    # cv2.imshow('Points', paint)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("marked.png", paint)
    
    # show_points(img, points_show, 3)
    # print(pattern[0])
    # print(pattern[1])
    
    rg = range_(QRsize + 1)
    
    nodes = [[None for y in range_(QRsize + 1)] for x in range_(QRsize + 1)]
    predicted_last = [[None for y in range_(QRsize + 1)] for x in range_(QRsize + 1)]
    if tps is not None:
        pts = []
        for x in range_(QRsize + 1):
            for y in range_(QRsize + 1):
                pts.append([x, y])
        pts = tps.tps_transform(pts)
        for x in range(len(rg)):
            for y in range(len(rg)):
                predicted_last[x][y] = pts[x * len(rg) + y]
        
        # show_points(img, pts)
    
    for x in range(len(rg)):
        for y in range(len(rg)):
            if predicted_last[x][y] is not None:
                last = predicted_last[x][y]
            elif x == 0 and y == 0:
                last = points[0]
            elif x == 0:
                last = nodes[x][y - 1]
            else:
                last = nodes[x - 1][y]
            
            p = PolylineIntersection(grade[0][x], grade[1][y])
            
            if p == []:
                nodes[x][y] = last
                continue
            
            nodes[x][y] = p[0]
            for i in p:
                if np.linalg.norm(last - i) < np.linalg.norm(last - nodes[x][y]):
                    nodes[x][y] = i

    # nodes[0][0] = points[0]
    # nodes[-1][0] = points[5]
    # nodes[0][-1] = points[15]
    
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
    
    nodes_show = []
    for x in range(QRsize + 1):
        for y in range(QRsize + 1):
            try:
                nodes_show.append(nodes[x][y])
                pixels[int(nodes[x][y][0] + 0.5), int(nodes[x][y][1]) + 0.5] = (255, 0, 0)
            except:
                pass
            
    # show_points(img, nodes_show, 2, (0, 0, 255))
        
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
        
    # img.save("marked.png")
    
    return corrected


def vectorfield_nodes(img, points, pattern = [None, None], QRsize = 25, square_size = 10, fract = 1, lines_fract = 5, tps = None):
    
    start_time = time.time()
    
    def range_(c, f = fract):
        a = [i for i in range(0, c - 1, f)]
        a.append(c - 1)
        return a

    paint = np.array(img)

    characteristic_size = max(np.linalg.norm(points[6] - points[3]),
                              np.linalg.norm(points[14] - points[1]))
    kernel_size = max(characteristic_size * 30 // 300, 35)
    if kernel_size % 2 == 0: kernel_size += 1
    print(characteristic_size, kernel_size)
    
    
    # points = data.predsetP2sq4_better
    # points = data.predsetP1sq4
    
    points_mark = list(points[:])
    if pattern[0] is not None: points_mark += list(pattern[0])
    if pattern[1] is not None: points_mark += list(pattern[1])
    points_show = []

    # pattern_comp = [complete_pattern(pattern[0], points[3], points[2], points[7], points[6]),
                    # complete_pattern(pattern[1], points[1], points[2], points[13], points[14])]
    
    pattern_comp = pattern
    
    # show_points(img, list(pattern[0]) + list(pattern[1]))

    df = DirectionField(img)
    # df.initGrid(kernel_size)
    
    grade = [[], []]

    for i in range_(QRsize + 1):

        if np.linalg.norm(pattern_comp[1][i] - points[2]) <\
           np.linalg.norm(pattern_comp[1][i] - points[13]):
            dist1 = points[2] - points[3]
        else:
            dist1 = points[13] - points[12]
            
        grade[1].append(constructPolyline(df, pattern_comp[1][i], dist1,
            QRsize, np.linalg.norm(points[6] - points[3]), lines_fract, kernel_size))
        
        if np.linalg.norm(pattern_comp[0][i] - points[2]) <\
           np.linalg.norm(pattern_comp[0][i] - points[7]):
            dist2 = points[2] - points[1]
        else:
            dist2 = points[7] - points[4]
        
        grade[0].append(constructPolyline(df, pattern_comp[0][i], dist2,
            QRsize, np.linalg.norm(points[14] - points[1]), lines_fract, kernel_size))
        
        # points_show += grade[0][-1].points
        # points_show += grade[1][-1].points
        
        zp = grade[0][-1].points
        zp = np.array(zp, dtype=np.int32)
        zp = zp.reshape((-1, 1, 2))
        cv2.polylines(paint, [zp], False, (255, 0, 0), 5)
        zp = grade[1][-1].points
        zp = np.array(zp, dtype=np.int32)
        zp = zp.reshape((-1, 1, 2))
        cv2.polylines(paint, [zp], False, (255, 0, 0), 5)

    # cv2.imshow('Points', paint)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # show_points(img, df.lucky)
    # show_points(img, df.dislucky)
    cv2.imwrite("marked.png", paint)
    
    # show_points(img, points_show, 3)
    # print(pattern[0])
    # print(pattern[1])
    
    rg = range_(QRsize + 1)
    
    nodes = [[None for y in range_(QRsize + 1)] for x in range_(QRsize + 1)]
    predicted_last = [[None for y in range_(QRsize + 1)] for x in range_(QRsize + 1)]
    if tps is not None:
        pts = []
        for x in range_(QRsize + 1):
            for y in range_(QRsize + 1):
                pts.append([x, y])
        pts = tps.tps_transform(pts)
        for x in range(len(rg)):
            for y in range(len(rg)):
                predicted_last[x][y] = pts[x * len(rg) + y]
        
        # show_points(img, pts)
    
    for x in range(len(rg)):
        for y in range(len(rg)):
            if predicted_last[x][y] is not None:
                last = predicted_last[x][y]
            elif x == 0 and y == 0:
                last = points[0]
            elif x == 0:
                last = nodes[x][y - 1]
            else:
                last = nodes[x - 1][y]
            
            p = PolylineIntersection(grade[0][x], grade[1][y])
            
            if p == []:
                nodes[x][y] = last
                continue
            
            nodes[x][y] = p[0]
            for i in p:
                if np.linalg.norm(last - i) < np.linalg.norm(last - nodes[x][y]):
                    nodes[x][y] = i

    print("ready in", time.time() - start_time)

    pts = []
    for x in range_(QRsize + 1):
        for y in range_(QRsize + 1):
            pts.append([x, y])
    nodes = np.asarray(nodes)
    nodes = np.concatenate(nodes)
    
    return pts, nodes





if __name__ == "__main__":
    # img = Image.open("data/v2/H/small/img0032_v2_H_small_d2.png")
    # img = Image.open("data/v2/H/big/img0003_v2_H_big_d2.png")
    # img = Image.open("data_orig/v2/M/small/img0124_v2_M_small_d2.png")
    img = Image.open("corrected.png")
    img = np.array(img)


    # weight, height = img.size
    # scale_ = min(2000 / weight, 2000 / height)
    # img = img.resize((int(weight * scale_), int(height * scale_)))
    
    df = DirectionField(img, show_flag=True)

    kernel = 35
    x = 1110
    y = 465
    print(df.get(0, 0, 1000001))
    
    x1 = max(0, x - kernel // 2)
    y1 = max(0, y - kernel // 2)
    x2 = x - kernel // 2 + kernel
    y2 = y - kernel // 2 + kernel
    arr = np.array(img)
    cv2.imwrite("corrected.png", arr[y1:y2, x1:x2])
    # show_points(img, [(x, y), (x1, y1), (x2, y2)])
    
    
if __name__ == "__main__" and False:
    img = Image.open("data/v2/Q/small/img0164_v2_Q_small_d2.png")
    
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    gray = upsample(gray, 2)
    gray *= 255
    
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    cv2.imwrite("corrected.png", gray)
