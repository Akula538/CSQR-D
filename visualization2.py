import matplotlib.pyplot as plt
import numpy as np

def parse_file(filename):
    """Парсит файл и возвращает словарь с данными"""
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            if '->' in line:
                filename_str, result = line.split('->')
                filename_str = filename_str.strip()
                result = result.strip()
                
                # Извлекаем тип кода и степень деформации из названия файла
                code_type = None
                deformation = None
                
                if '_H_' in filename_str:
                    code_type = 'H'
                elif '_L_' in filename_str:
                    code_type = 'L'
                elif '_M_' in filename_str:
                    code_type = 'M'
                elif '_Q_' in filename_str:
                    code_type = 'Q'
                
                # Извлекаем степень деформации (d0, d1, d2, d3)
                if '_d0.' in filename_str:
                    deformation = 'd0'
                elif '_d1.' in filename_str:
                    deformation = 'd1'
                elif '_d2.' in filename_str:
                    deformation = 'd2'
                elif '_d3.' in filename_str:
                    deformation = 'd3'
                
                if code_type and deformation:
                    key = (code_type, deformation)
                    if key not in data:
                        data[key] = {'total': 0, 'success': 0}
                    
                    data[key]['total'] += 1
                    if result != 'None' and result != 'CTF':
                        data[key]['success'] += 1
    return data

def calculate_accuracy_matrix(data, code_types, deformations):
    """Вычисляет матрицу точности"""
    accuracy_matrix = np.zeros((len(code_types), len(deformations)))
    
    for i, code_type in enumerate(code_types):
        for j, deformation in enumerate(deformations):
            key = (code_type, deformation)
            if key in data and data[key]['total'] > 0:
                accuracy = data[key]['success'] / data[key]['total']
                accuracy_matrix[i, j] = accuracy
            else:
                accuracy_matrix[i, j] = 0
    return accuracy_matrix

# Парсим два файла
file1 = "results/evr_test_log005.txt"
file2 = "evr_test_log.txt"

data1 = parse_file(file1)
data2 = parse_file(file2)

# Подготовка данных для тепловой карты
code_types = ['L', 'M', 'Q', 'H']
deformations = ['d0', 'd1', 'd2', 'd3']

# Вычисляем матрицы точности для обоих файлов
accuracy_matrix1 = calculate_accuracy_matrix(data1, code_types, deformations)
accuracy_matrix2 = calculate_accuracy_matrix(data2, code_types, deformations)

# Вычисляем матрицу прироста точности (в процентах)
improvement_matrix = (accuracy_matrix2 - accuracy_matrix1) * 100  # в процентах

# Построение тепловой карты прироста точности
fig, ax = plt.subplots(figsize=(10, 8))

# Создаем тепловую карту с цветовой гаммой от красного к зеленому
# vmin=-100, vmax=100 чтобы охватить весь диапазон от -100% до +100%
im = ax.imshow(improvement_matrix, cmap='RdYlGn', vmin=-33, vmax=33, aspect='auto')

# Настройка осей
ax.set_xticks(np.arange(len(deformations)))
ax.set_yticks(np.arange(len(code_types)))
ax.set_xticklabels(deformations)
ax.set_yticklabels(code_types)
ax.set_xlabel('Степень деформации')
ax.set_ylabel('Тип кода')
ax.set_title(f'Прирост точности считывания\n({file2} vs {file1})')

# Добавление цветовой шкалы
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Прирост точности (%)', rotation=-90, va="bottom")

# Добавление текста в ячейки
for i in range(len(code_types)):
    for j in range(len(deformations)):
        improvement = improvement_matrix[i, j]
        text_color = "white" if abs(improvement) > 50 else "black"  # меняем цвет текста для контраста
        text = ax.text(j, i, f'{improvement:+.1f}%',
                      ha="center", va="center", color=text_color, 
                      fontweight='bold', fontsize=10)

# Добавляем сетку для лучшей читаемости
ax.set_xticks(np.arange(-0.5, len(deformations), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(code_types), 1), minor=True)
ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
ax.tick_params(which="minor", size=0)

plt.tight_layout()
plt.show()

# # Вывод подробной статистики
# print("\n" + "="*60)
# print("ПОДРОБНАЯ СТАТИСТИКА СРАВНЕНИЯ")
# print("="*60)

# for code_type in code_types:
#     print(f"\n{code_type}:")
#     for deformation in deformations:
#         key = (code_type, deformation)
        
#         acc1 = 0
#         if key in data1 and data1[key]['total'] > 0:
#             acc1 = data1[key]['success'] / data1[key]['total']
        
#         acc2 = 0
#         if key in data2 and data2[key]['total'] > 0:
#             acc2 = data2[key]['success'] / data2[key]['total']
        
#         improvement = (acc2 - acc1) * 100
        
#         print(f"  {deformation}: {acc1:.1%} → {acc2:.1%} ({improvement:+.1f}%)")

# # Сводная статистика
# print("\n" + "="*60)
# print("СВОДНАЯ СТАТИСТИКА")
# print("="*60)
# print(f"Средний прирост точности: {np.mean(improvement_matrix):.1f}%")
# print(f"Максимальный прирост: {np.max(improvement_matrix):.1f}%")
# print(f"Минимальный прирост: {np.min(improvement_matrix):.1f}%")