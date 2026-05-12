import matplotlib.pyplot as plt
import numpy as np
import math

file = "evr_test_log.txt"
# file = "results/evr_test_log007.txt"

total_sum = 0.0
total_count = 0

# Чтение данных из файла
data = {}
with open(file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or '->' not in line:
            continue

        parts = [p.strip() for p in line.split('->')]
        # ожидаем: filename  -> result -> time
        if len(parts) < 3:
            continue

        filename = parts[0]
        result = parts[1]
        time_str = parts[2]

        # пропускаем строки с маркером CTF (как в оригинале)
        # if result == "CTF": continue

        # попытка преобразовать время в float (в конце строки находится дробное число)
        try:
            time_val = float(time_str)
        except ValueError:
            # если не получилось распарсить — пропускаем строку
            continue

        # Извлекаем тип кода и степень деформации из названия файла
        code_type = None
        deformation = None
        
        total_sum += time_val
        total_count += 1

        if '_H_' in filename:
            code_type = 'H'
        elif '_L_' in filename:
            code_type = 'L'
        elif '_M_' in filename:
            code_type = 'M'
        elif '_Q_' in filename:
            code_type = 'Q'

        # Извлекаем степень деформации (d0, d1, d2, d3)
        if '_d0.' in filename:
            deformation = 'd0'
        elif '_d1.' in filename:
            deformation = 'd1'
        elif '_d2.' in filename:
            deformation = 'd2'
        elif '_d3.' in filename:
            deformation = 'd3'

        if code_type and deformation:
            key = (code_type, deformation)
            if key not in data:
                data[key] = {'count': 0, 'sum_time': 0.0}
            data[key]['count'] += 1
            data[key]['sum_time'] += time_val

print(f"Total average time: {total_sum / total_count:.4f} seconds over {total_count} samples")

# Подготовка данных для тепловой карты
code_types = ['L', 'M', 'Q', 'H']
deformations = ['d0', 'd1', 'd2', 'd3']

# Создаем матрицу средних времён (в секундах)
avg_matrix = np.full((len(code_types), len(deformations)), np.nan, dtype=float)

for i, code_type in enumerate(code_types):
    for j, deformation in enumerate(deformations):
        key = (code_type, deformation)
        if key in data and data[key]['count'] > 0:
            avg = data[key]['sum_time'] / data[key]['count']
            avg_matrix[i, j] = avg
        else:
            avg_matrix[i, j] = np.nan  # отсутствующие значения

# Построение тепловой карты
fig, ax = plt.subplots(figsize=(8, 6))

# определим vmin/vmax по диапазону ненулевых значений, чтобы цветовая шкала была информативной
if np.all(np.isnan(avg_matrix)):
    vmin, vmax = 0, 1
else:
    vmin = np.nanmin(avg_matrix)
    vmax = np.nanmax(avg_matrix)
    # если разброс очень мал — добавить небольшой запас
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-6

# cmap можно поменять по вкусу
im = ax.imshow(avg_matrix, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')

# Настройка осей
ax.set_xticks(np.arange(len(deformations)))
ax.set_yticks(np.arange(len(code_types)))
ax.set_xticklabels(deformations)
ax.set_yticklabels(code_types)
ax.set_xlabel('Степень деформации')
ax.set_ylabel('Уровень коррекции ошибок')
ax.set_title('Среднее время обработки (с) по типам и степеням деформации')

# Добавление цветовой шкалы
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Среднее время (с)', rotation=-90, va="bottom")

# Добавление текста в ячейки (форматирование, пропуски показываем как '-')
for i in range(len(code_types)):
    for j in range(len(deformations)):
        val = avg_matrix[i, j]
        if np.isnan(val):
            text = '—'
        else:
            text = f'{val:.2f}'
        ax.text(j, i, text, ha="center", va="center", color="white", fontweight='bold')

plt.tight_layout()
plt.show()
