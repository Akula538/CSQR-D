import cv2
import numpy as np

def simple_sobel_filter(image_path, output_path):
    """
    Простая версия с применением оператора Собеля
    """
    # Загрузка изображения
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Ошибка загрузки изображения")
        return
    
    # Ядра Собеля
    kernel_x = np.array([[-1, -2, -1],
                         [0,  0,  0],
                         [1,  2,  1]])
    
    kernel_y = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    
    # Применение свертки
    grad_x = cv2.filter2D(img.astype(np.float32), -1, kernel_x)
    grad_y = cv2.filter2D(img.astype(np.float32), -1, kernel_y)
    
    # Вычисление общего градиента
    grad = np.sqrt(grad_x**2 + grad_y**2)
    
    # Нормализация и сохранение
    grad_normalized = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
    result = grad_normalized.astype(np.uint8)
    
    cv2.imwrite(output_path, result)
    print(f"Результат сохранен: {output_path}")

# Использование
simple_sobel_filter("data/v2/H/small/img0039_v2_H_small_d1.png", "marked.png")
