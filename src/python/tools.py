import cv2
import numpy as np

def show_points(img, points=[], size = 5, color = (0, 255, 0)):
    rgb_array = np.array(img)
    paint = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    for i, (x, y) in enumerate(points):
        cv2.circle(paint, (int(x), int(y)), size, color, -1)
    
    cv2.imwrite("marked.png", paint)
    
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

from shapely.geometry import (
    Point, MultiPoint, LineString, LinearRing,
    MultiLineString, Polygon, MultiPolygon, GeometryCollection
)
from typing import Union, Iterable

ShapelyGeom = Union[
    Point, MultiPoint, LineString, LinearRing,
    MultiLineString, Polygon, MultiPolygon, GeometryCollection
]

def fucking_geom(geom: ShapelyGeom,
                        round_decimals: int = 9,
                        max_points: int | None = None) -> np.ndarray:
    """
    Преобразует любую shapely-геометрию в массив точек np.array([[x,y], ...]).
    - round_decimals: количество знаков при округлении при дедупликации (по умолчанию 9).
    - max_points: если указано, вернёт не более этого числа точек (ранний выход).
    Возвращает np.ndarray формы (m,2), где m >= 0.
    """
    if geom is None:
        return []

    pts: list[tuple[float, float]] = []

    def add_xy(x: float, y: float):
        pts.append((float(x), float(y)))

    def collect(g):
        """рекурсивно собирает координаты из geometry"""
        if g is None or g.is_empty:
            return
        t = g.geom_type
        if t == 'Point':
            add_xy(g.x, g.y)
        elif t == 'MultiPoint':
            for p in g.geoms:
                add_xy(p.x, p.y)
        elif t in ('LineString', 'LinearRing'):
            for x, y in g.coords:
                add_xy(x, y)
        elif t == 'MultiLineString':
            for line in g.geoms:
                for x, y in line.coords:
                    add_xy(x, y)
        elif t == 'Polygon':
            # берем внешнюю границу + (опционально) внутренние кольца
            for x, y in g.exterior.coords:
                add_xy(x, y)
            for interior in g.interiors:
                for x, y in interior.coords:
                    add_xy(x, y)
        elif t == 'MultiPolygon':
            for poly in g.geoms:
                collect(poly)
        elif t == 'GeometryCollection':
            for sub in g.geoms:
                collect(sub)
        else:
            # на всякий случай: попытка взять coords (если есть)
            try:
                for x, y in g.coords:
                    add_xy(x, y)
            except Exception:
                pass

    collect(geom)

    if not pts:
        return []

    # дедупликация с помощью округления (чтобы избежать мелких FP-расхождений)
    seen = set()
    unique: list[list[float]] = []
    for x, y in pts:
        key = (round(x, round_decimals), round(y, round_decimals))
        if key not in seen:
            seen.add(key)
            unique.append([x, y])
            if max_points is not None and len(unique) >= max_points:
                break

    return unique

def cut_image(img, qr, pattern_recovered):

    x1 = min(np.column_stack(qr)[0])
    x2 = max(np.column_stack(qr)[0])
    y1 = min(np.column_stack(qr)[1])
    y2 = max(np.column_stack(qr)[1])
    
    x1 = max(x1 - 0.3 * (x2 - x1), 0)
    x2 = min(x2 + 0.3 * (x2 - x1), img.shape[1])
    y1 = max(y1 - 0.3 * (y2 - y1), 0)
    y2 = min(y2 + 0.3 * (y2 - y1), img.shape[0])
    
    for i in range(len(qr)):
        qr[i] = qr[i] - np.array([x1, y1])
        
    for i in range(len(pattern_recovered)):
        for j in range(len(pattern_recovered[i])):
            pattern_recovered[i][j] = pattern_recovered[i][j] - np.array([x1, y1])

    return img[int(y1):int(y2), int(x1):int(x2)], qr, pattern_recovered