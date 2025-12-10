import cv2
import numpy as np
from shapely.geometry import *
from PIL import Image
from shapely.validation import explain_validity
import time
from interpolation_correct import print_as_point
import sys
import itertools
from tools import *
from homographic_correct import HomographyMatrix
import subprocess
from binarizer import binarize_mean
sys.setrecursionlimit(500000)

def approxQuadr(contour):
    l = 0.1
    r = cv2.arcLength(contour, True)
    while r - l > 0.1:
        m = (l + r) / 2
        
        approx = cv2.approxPolyDP(contour, m, True)
        if len(approx) <= 4:
            r = m
        else:
            l = m
    
    return cv2.approxPolyDP(contour, r, True)

def contour_iou_rect(contour1, contour2):
    rect1 = cv2.boundingRect(contour1)
    rect2 = cv2.boundingRect(contour2)
    
    x = min(rect1[0], rect2[0])
    y = min(rect1[1], rect2[1])
    w = max(rect1[0] + rect1[2], rect2[0] + rect2[2]) - x
    h = max(rect1[1] + rect1[3], rect2[1] + rect2[3]) - y
    
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)
    
    contour1_offset = contour1 - [x, y]
    contour2_offset = contour2 - [x, y]
    
    cv2.fillPoly(mask1, [contour1_offset], 1)
    cv2.fillPoly(mask2, [contour2_offset], 1)
    
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    iou = np.sum(intersection) / np.sum(union)
    return iou


def contour_iou(contour1, contour2):
    try:
        poly1 = Polygon(contour1.reshape(-1, 2))
        poly2 = Polygon(contour2.reshape(-1, 2))
        
        if not poly1.is_valid: poly1 = poly1.buffer(0)
        if not poly2.is_valid: poly2 = poly2.buffer(0)
        
        if not poly1.is_valid or not poly2.is_valid:
            # print("=============")
            # print(poly1)
            # print(poly2)
            # print("=============")
            print(poly1)
            print(explain_validity(poly1))
            print(poly2)
            print(explain_validity(poly2))
            return 0.0
        
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        
        if union == 0:
            return 0.0
        
        return intersection / union
    except Exception as e:
        # print("ex is ", e)
        # print(poly1)
        # print(poly2)
        return 0.0

def get_colors(contours, hierarchy):
    colors = [0 for _ in range(len(contours))]
    def dfs(v, c):
        if v == -1: return
        colors[v] = c
        dfs(hierarchy[0][v][0], c)
        dfs(hierarchy[0][v][2], 1 - c)
    
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            dfs(i, 0)
    
    return colors
        

def isQuadr(contour):
    if cv2.contourArea(contour) == 0: return False
    return contour_iou(approxQuadr(contour), contour) >= 0.7


def find_search(contours, hierarchy, colors, lvl = 0):
    if lvl == 0:
        ans = []
        for i in range(len(contours)):
            if colors[i] != 1: continue
            
            j = hierarchy[0][i][2]
            while j != -1:
                k = hierarchy[0][j][2]
                while k != -1:
                    if isQuadr(contours[k]) and isQuadr(contours[j]) and isQuadr(contours[i]):
                        ans.append(i)
                        break
                    k = hierarchy[0][k][0]
                else:
                    j = hierarchy[0][j][0]
                    continue
                break
            
        return ans
    if lvl == 1:
        ans = []
        for i in range(len(contours)):
            if colors[i] != 0: continue
            
            j = hierarchy[0][i][2]
            while j != -1:
                if isQuadr(contours[j]) and isQuadr(contours[i]):
                    ans.append(i)
                    break
                j = hierarchy[0][j][0]
            
        return ans
    if lvl == 2:
        ans = []
        for i in range(len(contours)):
            if colors[i] != 1: continue
            
            if isQuadr(contours[i]):
                ans.append(i)
            
        return ans

def find_corrective(contours, hierarchy, colors, search_area):
    ans = []
    for i in range(len(contours)):
        if colors[i] != 0 or not isQuadr(contours[i]): continue
        
        if hierarchy[0][i][2] != -1 and cv2.contourArea(contours[i]) < search_area:
            ans.append(i)
    
    return ans


def clockwise(contour):
    if len(contour) < 3:
        return contour
    
    area = cv2.contourArea(contour, oriented=True)
    
    if area < 0:
        return contour[::-1]
    else:
        return contour

def cyclic_shift(arr, start_index):
    return arr[start_index:] + arr[:start_index]


def sort_search(search):
    def calculate_angle(p1, p2, p3):
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 180
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))
        angle_rad = np.acos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    all_vertices = np.vstack(search)
    
    global img
    
    search = [[tuple(map(int, j)) for j in i] for i in search]
        
    multi_point = MultiPoint(all_vertices)
    convex_hull = multi_point.convex_hull
    
    hull_vertices = list(convex_hull.exterior.coords)[:-1]
    hull_vertices = [tuple(map(int, vertex)) for vertex in hull_vertices]
    
    # show_points(img, hull_vertices)
        
    angles = []
    n = len(hull_vertices)
    
    for i in range(n):
        p1 = hull_vertices[(i-1) % n]
        p2 = hull_vertices[i]
        p3 = hull_vertices[(i+1) % n]
        
        angle = calculate_angle(p1, p2, p3)
        angles.append(angle)
    
    
    angle_indices = list(zip(angles, range(n)))
    angle_indices.sort()
    selected_indices = [idx for angle, idx in angle_indices[:5]]
    selected_indices.sort()
    
    pentagon = [hull_vertices[i] for i in selected_indices]
    pentagon.reverse()
    # print(pentagon)
    
    # show_points(img, pentagon)
    
    
    new_search = []
    
    zero_point = None
    for i in range(3):
        points_in = []
        for point in search[i]:
            if tuple(point) in pentagon:
                points_in.append(tuple(point))
        if len(points_in) == 1:
            zero_point = pentagon.index(points_in[0])            
            search[i] = cyclic_shift(search[i], search[i].index(points_in[0]))
            new_search.append(search[i])
            break
    
    if zero_point is None:
        raise ValueError("Не найден четырехугольник с одной вершиной в выпуклой оболочке")
    
    
    point = pentagon[(zero_point + 1) % 5]
    for i in range(3):
        if point in search[i]:
            search[i] = cyclic_shift(search[i], (search[i].index(point) - 1) % 4)
            new_search.append(search[i])
            break
    point = pentagon[(zero_point - 1) % 5]
    for i in range(3):
        if point in search[i]:
            search[i] = cyclic_shift(search[i], (search[i].index(point) + 1) % 4)
            new_search.append(search[i])
            break
    
    new_search = [[np.array(j) for j in i] for i in new_search]
    
    return new_search

def center(v):
    s = np.array([0, 0])
    for p in v:
        s = s + p
    return s / len(v)

def dist_center(a, b):
    return np.linalg.norm(center(a) - center(b))


def sort_search_dist(search):
    new_search = [None, None, None]
    
    mx = 0
    for i in range(3):
        dist = dist_center(search[(i + 1) % 3], search[(i + 2) % 3])
        if dist > mx:
            mx = dist
            new_search = cyclic_shift(search, i)
    
    if cross(center(new_search[2]) - center(new_search[0]),
                center(new_search[1]) - center(new_search[0])) > 0:
        new_search[1], new_search[2] = new_search[2], new_search[1]
    
    c = (center(new_search[1]) + center(new_search[2])) / 2
    
    def shift_search(ind, target):
        best = 0
        for i in range(4):
            if np.linalg.norm(new_search[ind][best] - c) >\
               np.linalg.norm(new_search[ind][i] - c):
                best = i
        # print(new_search[ind], (best - target) % 4)
        new_search[ind] = cyclic_shift(list(new_search[ind]), (best - target) % 4)
    
    shift_search(0, 2)
    shift_search(1, 3)
    shift_search(2, 1)
    
    return new_search

def sort_search_dir(search):
    new_search = [None, None, None]
    
    def abs_mod(x):
        x %= np.pi
        return min(x, np.pi - x)
    
    def mean_mod(a, b):
        x = (a + b) / 2
        y = (x + np.pi / 2) % np.pi
        if abs_mod(x - a) < abs_mod(y - a): return x
        return y
    
    dirs = [[np.atan2(*reversed(s[(i + 1) % 4] - s[i])) % np.pi for i in range(4)] for s in search]
    dirs = [[mean_mod(d[0], d[2]), mean_mod(d[1], d[3])] for d in dirs]
    
    mx = 0
    best = 0
    for i in range(3):
        angle = abs_mod(dirs[i][0] - dirs[i][1])
        if angle > mx:
            mx = angle
            best = i
    
    dir1, dir2 = dirs[best]
    
    if abs_mod(dir1 - dirs[(best + 1) % 3][0]) >\
        abs_mod(dir1 - dirs[(best + 1) % 3][1]):
        dirs[(best + 1) % 3][0], dirs[(best + 1) % 3][1] = dirs[(best + 1) % 3][1], dirs[(best + 1) % 3][0]
    
    if abs_mod(dir1 - dirs[(best + 2) % 3][0]) >\
        abs_mod(dir1 - dirs[(best + 2) % 3][1]):
        dirs[(best + 2) % 3][0], dirs[(best + 2) % 3][1] = dirs[(best + 2) % 3][1], dirs[(best + 2) % 3][0]
    
    # print(dir1 / np.pi * 180, dir2 / np.pi * 180)
    # for i in range(3):
    #     print(dirs[(best + i) % 3][0] / np.pi * 180, dirs[(best + i) % 3][1] / np.pi * 180)
    
    dir1 = mean_mod(dir1, mean_mod(dirs[(best + 1) % 3][0], dirs[(best + 2) % 3][0]))
    dir2 = mean_mod(dir2, mean_mod(dirs[(best + 1) % 3][1], dirs[(best + 2) % 3][1]))
    # dir1 = (dir1 + np.pi / 2) % np.pi
    # dir2 = (dir2 + np.pi / 2) % np.pi
    
    dir1 = np.array([np.cos(dir1), np.sin(dir1)])
    dir2 = np.array([np.cos(dir2), np.sin(dir2)])
    
    # global img
    # show_points(img, [(dir1 + np.array([1, 0])) * 150, (dir2 + np.array([1, 0])) * 150, [150, 0]])
    
    M = np.array([[dir1[0], dir2[0]],
                  [dir1[1], dir2[1]]])
    
    M = np.linalg.inv(M)
    
    # for i in range(3):
    #     print(M @ center(search[i]))
    
    def worst_by_dir(d):
        mx = [0, 0]
        for i in range(3):
            # x = abs(np.dot(center(search[i]), d) - np.dot(center(search[(i + 1) % 3]), d)) +\
            # abs(np.dot(center(search[i]), d) - np.dot(center(search[(i + 2) % 3]), d))
            
            x = abs((M @ center(search[i]))[d] - (M @ center(search[(i + 1) % 3]))[d]) +\
            abs((M @ center(search[i]))[d] - (M @ center(search[(i + 2) % 3]))[d])
            mx = max(mx, [x, i])
        return mx[1]
    
    # print(dir1 / np.pi * 180, dir2 / np.pi * 180)
    
    left_top = [0, 1, 2]
    if worst_by_dir(0) == worst_by_dir(1):
        return sort_search(search)
    left_top.remove(worst_by_dir(0))
    left_top.remove(worst_by_dir(1))
    
    
    new_search = cyclic_shift(search, left_top[0])
    
    
    if cross(center(new_search[2]) - center(new_search[0]),
                center(new_search[1]) - center(new_search[0])) > 0:
        new_search[1], new_search[2] = new_search[2], new_search[1]
    
    c = (center(new_search[1]) + center(new_search[2])) / 2
    
    def shift_search(ind, target):
        best = 0
        for i in range(4):
            if np.linalg.norm(new_search[ind][best] - c) >\
               np.linalg.norm(new_search[ind][i] - c):
                best = i
        # print(new_search[ind], (best - target) % 4)
        new_search[ind] = cyclic_shift(list(new_search[ind]), (best - target) % 4)
    
    shift_search(0, 2)
    shift_search(1, 3)
    shift_search(2, 1)
    
    return new_search


def sort_corrective(correctives, search):
    corrective = correctives[0]
    predicted = search[1][3] + search[2][1] - search[0][2]
    for c in correctives:
        if np.linalg.norm(predicted - center(c)) <\
           np.linalg.norm(predicted - center(corrective)):
            corrective = c
                
    zero = 0
    for i in range(4):
         if np.linalg.norm(center(search[0]) - corrective[i]) <\
           np.linalg.norm(center(search[0]) - corrective[zero]):
            zero = i
    
    corrective = cyclic_shift(list(corrective), zero)
    corrective = [np.array(list(map(int, i))) for i in corrective]
    return corrective


def cov_evristic_sort(search, search_areas):

    def evr(qr):
        def cov(data):
            return np.std(data, ddof=1) / np.mean(data)
        
        dist_arr = [dist_center(search[qr[i]], search[qr[(i + 1) % 3]]) for i in range(3)]
        dist_arr.sort()
        dist_arr[2] /= np.sqrt(2)
        
        return cov([search_areas[j] for j in qr]) + cov(dist_arr)
    
    
    
    all_search_sort = list(itertools.combinations([i for i in range(len(search))], 3))
    all_search_sort = [[evr(i), i] for i in all_search_sort]
    all_search_sort.sort()
    all_search = [[search[j] for j in i[1]] for i in all_search_sort]
    
    return all_search

def disp_evristic_sort(search, search_areas):

    def evr(qr):
        def F_area(S1, S2, S3):
            mean_area = (S1 + S2 + S3) / 3
            variance = ((S1 - mean_area)**2 + (S2 - mean_area)**2 + (S3 - mean_area)**2) / 3
            
            epsilon = 1e-6
            return 1 / (1 + variance / (mean_area + epsilon))
        def F_shape(sides):
            sides = sorted(sides)
            # print(sides)
            leg1, leg2, hypo = sides[0], sides[1], sides[2]
            
            ideal_leg_ratio = 1.0      # leg1/leg2 = 1
            ideal_hypo_ratio = 2**0.5  # hypo/leg1 = √2
            
            actual_leg_ratio = leg1 / leg2 if leg2 > 0 else 0
            actual_hypo_ratio = hypo / leg1 if leg1 > 0 else 0
            
            leg_similarity = 1 - min(1, abs(actual_leg_ratio - ideal_leg_ratio))
            hypo_similarity = 1 - min(1, abs(actual_hypo_ratio - ideal_hypo_ratio) / ideal_hypo_ratio)
            
            angle_score = 0
            try:
                cos_angle = (leg1**2 + leg2**2 - hypo**2) / (2 * leg1 * leg2)
                # print (leg1, leg2)
                angle_error = abs(cos_angle)  # Для прямого угла cos ≈ 0
                angle_score = 1 - min(1, angle_error)
            except:
                angle_score = 0
            
            return (leg_similarity + hypo_similarity + angle_score) / 3
        
        
        return -(0.4 * F_area(search_areas[qr[0]], search_areas[qr[1]], search_areas[qr[2]]) +
                 0.6 * F_shape([dist_center(search[qr[i]], search[qr[(i + 1) % 3]]) for i in range(3)]))
    
    
    all_search_sort = list(itertools.combinations([i for i in range(len(search))], 3))
    # print(all_search_sort)
    all_search_sort = [[evr(i), i] for i in all_search_sort]
    all_search_sort.sort()
    all_search = [[search[j] for j in i[1]] for i in all_search_sort]
    indexes = [[j for j in i[1]] for i in all_search_sort]
    
    # print(all_search_sort)
    
    return all_search, indexes

def find(img):
    start_time = time.time()
    
    rgb_array = np.array(img)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(contours[0])
    
    # print(len(hierarchy[0]))
    
    colors = get_colors(contours, hierarchy)
    # print(time.time() - start_time)
    
    paint = bgr_array.copy()
    
    # start_time = time.time()
    # find_search(contours, hierarchy, colors)
    # print("find", time.time() - start_time)
    #===============================
    
    search_contours = [approxQuadr(contours[i]) for i in find_search(contours, hierarchy, colors)]
    # print(search_contours)
    search_contours = [clockwise(i) for i in search_contours]
    search_areas = [cv2.contourArea(i) for i in search_contours]
    search = [i.reshape(-1, 2) for i in search_contours]
    
    all_search, all_indexes = disp_evristic_sort(search, search_areas)
 
    search = all_search[0]
    
    search_area = sum(search_areas) / max(len(search_contours), 1)
    
    search = sort_search_dir(search)
        
    corrective_contours = [approxQuadr(contours[i]) for i in find_corrective(contours, hierarchy, colors, search_area)]
    corrective_contours = [clockwise(i) for i in corrective_contours]
    corrective = [i.reshape(-1, 2) for i in corrective_contours]
    corrective = sort_corrective(corrective, search)
    search.insert(2, corrective)
    search = np.array(search)
    search = search.reshape(-1, 2)
    
    # ===============================
    # print(search)
    # print(corrective)
    
    print("ready in", time.time() - start_time)
    
    # for con in find_search(contours, hierarchy, colors):
    #     cv2.drawContours(paint, [approxQuadr(contours[con])], -1, (0, 0, 255), 2)
    # for con in find_corrective(contours, hierarchy, colors, search_area):
    #     cv2.drawContours(paint, [approxQuadr(contours[con])], -1, (255, 0, 0), 2)
    # for i, (x, y) in enumerate(search):
    #     cv2.circle(paint, (x, y), 5, (0, 255, 0), -1)
    #     cv2.putText(paint, str(i), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)    
    
    # cv2.imshow('All Contours', paint)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return search
    

def recognize_distribution_(img, p1, p2, p3, H = None):
    def get_pixel(p):
        if H is not None:
            p = H.predict(p)
        pixel = img.getpixel((int(p[0]), int(p[1])))
        return int(sum(pixel) / len(pixel) > 127)
    
    from interpolation_correct import Curve
    
    cur = Curve(p1, p2, p3)
    
    dx = 0.5 * np.sign(cur.dist(p3, p2))
    
    points = [cur.move(p2, dx * i) for i in range(2 * int(abs(cur.dist(p2, p3))))]
    points = [np.array([int(i[0]), int(i[1])]) for i in points]
    
    # for p in points:
    #     print(p, get_pixel(p))
    
    while not get_pixel(points[-1]): points.pop(-1)
    while not get_pixel(points[0]): points.pop(0)
    
    bord = []
    for i in range(len(points) - 1):
        if get_pixel(points[i]) != get_pixel(points[i + 1]):
            bord.append(points[i])
    
    trashold = max(abs(cur.dist(bord[i + 1], bord[i])) for i in range(len(bord) - 1)) * 0.3
    
    i = 0
    while i + 1 < len(bord):
        if abs(cur.dist(bord[i + 1], bord[i])) < trashold:
            bord.pop(i)
            bord.pop(i)
        else:
            i += 1

    return bord

def recognize_distribution(img, search, H = None):
    pattern1 = recognize_distribution_(img, (search[0] + search[3] * 13) / 14,
                                            (search[1] + search[2] * 13) / 14,
                                            (search[4] + search[7] * 13) / 14, H)
    
    pattern2 = recognize_distribution_(img, (search[0] + search[1] * 13) / 14,
                                            (search[3] + search[2] * 13) / 14,
                                            (search[12] + search[13] * 13) / 14, H)
    
    # show_points(img, pattern1 + pattern2, 1)
    
    sz1 = (len(pattern1) // 4) * 4 + 17
    sz2 = (len(pattern2) // 4) * 4 + 17
    size = max(sz1, sz2)
    
    if len(pattern1) % 4 != 2: pattern1 = None
    if len(pattern2) % 4 != 2: pattern2 = None
    
    if pattern1 is None and pattern2 is not None: size = sz2
    if pattern1 is not None and pattern2 is None: size = sz1
    
    if size != sz1: pattern1 = None
    if size != sz2: pattern2 = None
    
    return size, [pattern1, pattern2]
    

def zxing_find(img):
    img.save("zxing-cpp_finder/input.png")
    run = subprocess.run('.\\"zxing-cpp_finder\\zxing_find.exe" "zxing-cpp_finder/input.png"', shell=True, capture_output=True, text=True)
    res = run.stdout.split('\n')
    res.pop()
    
    def point_from(s):
        return np.array(list(map(float, s.split(','))))
    
    res = [i.split() for i in res]
    # print(res)
    res = [[point_from(i[0]), int(i[1]), [point_from(j) for j in i[2:]]] for i in res]
    
    i = 0
    while i < len(res) - 1:
        if res[i][0][0] == res[i + 1][0][0] and res[i][0][1] == res[i + 1][0][1]:
            res.pop(i + 1)
        else:
            i += 1
    
    return res

def zxing_check(img):
    res = zxing_find(img)
    if len(res) < 3:
        return "CTFall"
    
    cnt = sum(int(len(i[2]) == 4) for i in res)
    
    if cnt < 3:
        return "CTF"

    return None


def findContoursWindow(binary, c, d):
    def relux(x):
        ym, xm = binary.shape
        if x < 0: return 0
        if x >= xm: return xm - 1
        return x
    def reluy(y):
        ym, xm = binary.shape
        if y < 0: return 0
        if y >= ym: return ym - 1
        return y

    d += 1
    x1 = relux(c[0] - d)
    x2 = relux(c[0] + d)
    y1 = reluy(c[1] - d)
    y2 = reluy(c[1] + d)
    
    roi = binary[y1:y2, x1:x2].copy()

    # show_points(roi, [])
    
    contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if np.any((pts[:,0] <= 1) | (pts[:,0] >= x2 - x1 - 1) |
                    (pts[:,1] <= 1) | (pts[:,1] >= y2 - y1 - 1)):
            cnt[:] = np.array([[[0, 0]]])
            # print(cv2.contourArea(cnt))
        # else:
        #     show_points(roi, pts, 1)
            
    
    for cnt in contours:
        cnt[:] = cnt + np.array([[x1, y1]])
        
    return contours, hierarchy


def find_qr_zxing(orig):

    prets = zxing_find(orig)
    img = Image.open("zxing-cpp_finder/output.png")
    start_time = time.time()
    
    
    rgb_array = np.array(img)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    paint = bgr_array.copy()
    

    search_contours = []
    for pret in prets:
        c = pret[0]
        c = np.array([int(c[0]), int(c[1])])
        if len(pret[2]):
            search_contours.append([])
            for p in pret[2]:
                search_contours[-1].append([(p - c) * 7/6 + c])
        else:
                        
            contours, hierarchy = findContoursWindow(binary, c, 1 * pret[1])
            colors = get_colors(contours, hierarchy)
            
            for lvl in range(0, 3):
                search_prets = find_search(contours, hierarchy, colors, lvl)
                if len(search_prets):
                    best = approxQuadr(contours[search_prets[0]])
                    for contour in search_prets:
                        cont = approxQuadr(contours[contour])
                        if np.linalg.norm(center(best.reshape(-1, 2)) - c) >\
                           np.linalg.norm(center(cont.reshape(-1, 2)) - c):
                               best = cont
                    
                    best = best.reshape(-1, 2)
                    search_contours.append([])
                    for p in best:
                        search_contours[-1].append([(p - c) * 7 / (7 - lvl * 2) + c])
                    break

    search_contours = [np.array(i, dtype="int32") for i in search_contours]
    search_contours = [clockwise(i) for i in search_contours]
    search_areas = [cv2.contourArea(i) for i in search_contours]
    search = [i.reshape(-1, 2) for i in search_contours]
    show = []
    for i in search:
        show += list(i)
    # show_points(img, show)
    
    all_search, all_indexes = disp_evristic_sort(search, [i[1] for i in prets])
 
    search = all_search[0]
    indexes = all_indexes[0]
    
    search_area = sum(search_areas[i] for i in indexes) / 3
    search_module = max(prets[i][1] for i in indexes)
    
    search = sort_search_dir(search)
    
    mask = np.zeros_like(binary)
    c = search[1][3] + search[2][1] - search[0][2]
    
    # search_module *= 100
    # show_points(mask, [c])

    contours, hierarchy = findContoursWindow(binary, c, 2 * search_module)
    colors = get_colors(contours, hierarchy)
    
    corrective_contours = [approxQuadr(contours[i]) for i in find_corrective(contours, hierarchy, colors, search_area)]
    corrective_contours = [clockwise(i) for i in corrective_contours]
    corrective = [i.reshape(-1, 2) for i in corrective_contours]
    corrective = sort_corrective(corrective, search)
    search.insert(2, corrective)
    search = np.array(search)
    search = search.reshape(-1, 2)
    
    # ===============================
    # print(search)
    # print(corrective)
    
    print("ready in", time.time() - start_time)
    
    # for con in find_search(contours, hierarchy, colors):
    #     cv2.drawContours(paint, [approxQuadr(contours[con])], -1, (0, 0, 255), 2)
    # for con in find_corrective(contours, hierarchy, colors, search_area):
    #     cv2.drawContours(paint, [approxQuadr(contours[con])], -1, (255, 0, 0), 2)
    
    
    # for i, (x, y) in enumerate(search):
    #     cv2.circle(paint, (x, y), 5, (0, 255, 0), -1)
    #     cv2.putText(paint, str(i), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # cv2.imshow('All Contours', paint)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return search, img.convert("RGB")
    


if __name__ == "__main__":
    # img = Image.open("orig_bynarized1.png")
    # img = Image.open("corrected.png")
    img = Image.open("data/v2/M/big/img0095_v2_M_big_d2.png")
    # img = Image.open("data/v2/Q/big/img0155_v2_Q_big_d1.png")
    img = Image.open("data/v2/H/big/img0001_v2_H_big_d2.png")
    
    
    weight, height = img.size
    scale = min(1000 / weight, 1000 / height)
    img = img.resize((int(weight * scale), int(height * scale)))
    # from binarizer import binorize_mean as binarize
    # img = binarize(img, 15)
    
    # img = binarize_mean(img)
    
    find_qr_zxing(img)
    # find(img)
    # show_points(img, find_qr_zxing(img))
    # print(*[i[1] for i in zxing_find(img)])