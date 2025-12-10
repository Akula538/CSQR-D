import matplotlib.pyplot as plt
import numpy as np

file = "evr_test_log.txt"
# file = "results\evr_test_log005.txt"

# Чтение данных из файла
data = {}
with open(file, 'r') as f:
    for line in f:
        if '->' in line:
            filename, result = line.split('->')
            filename = filename.strip()
            result = result.strip()
            if result == "CTF": continue
            
            # Извлекаем тип кода и степень деформации из названия файла
            code_type = None
            deformation = None
            
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
                    data[key] = {'total': 0, 'success': 0}
                
                data[key]['total'] += 1
                if result != 'None' and result != 'CTF':
                    data[key]['success'] += 1

# Подготовка данных для тепловой карты
code_types = ['L', 'M', 'Q', 'H']
deformations = ['d0', 'd1', 'd2', 'd3']

# Создаем матрицу точности
accuracy_matrix = np.zeros((len(code_types), len(deformations)))

for i, code_type in enumerate(code_types):
    for j, deformation in enumerate(deformations):
        key = (code_type, deformation)
        if key in data and data[key]['total'] > 0:
            accuracy = data[key]['success'] / data[key]['total']
            accuracy_matrix[i, j] = accuracy
        else:
            accuracy_matrix[i, j] = 0  # или np.nan для пропущенных значений

# Построение тепловой карты
fig, ax = plt.subplots(figsize=(8, 6))

# Создаем тепловую карту с цветовой гаммой от красного к зеленому
im = ax.imshow(accuracy_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

# Настройка осей
ax.set_xticks(np.arange(len(deformations)))
ax.set_yticks(np.arange(len(code_types)))
ax.set_xticklabels(deformations)
ax.set_yticklabels(code_types)
ax.set_xlabel('Степень деформации')
ax.set_ylabel('Уровень коррекции ошибок')
ax.set_title('Точность считывания кодов по типам и степеням деформации')

# Добавление цветовой шкалы
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Точность', rotation=-90, va="bottom")

# Добавление текста в ячейки
for i in range(len(code_types)):
    for j in range(len(deformations)):
        text = ax.text(j, i, f'{accuracy_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black", fontweight='bold')

plt.tight_layout()
plt.show()

# # Вывод подробной статистики
# print("Подробная статистика:")
# for code_type in code_types:
#     print(f"\n{code_type}:")
#     for deformation in deformations:
#         key = (code_type, deformation)
#         if key in data:
#             total = data[key]['total']
#             success = data[key]['success']
#             accuracy = success / total if total > 0 else 0
#             print(f"  {deformation}: {success}/{total} ({accuracy:.2%})")