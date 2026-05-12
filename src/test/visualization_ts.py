import matplotlib.pyplot as plt
import numpy as np
import re

# Чтение файла
with open('ts.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Обработка строк для извлечения чисел
ok_numbers = []
not_ok_numbers = []
is_prev_number = False
prev_number = None

for i, line in enumerate(lines):
    line = line.strip()
    
    # Пропускаем пустые строки
    if not line:
        continue
    
    # Пытаемся извлечь число из строки
    match = re.search(r'\b\d+\.\d+\b', line)
    if match:
        number = float(match.group())
        prev_number = number
        is_prev_number = True
    elif line == "OK" and is_prev_number and prev_number is not None:
        ok_numbers.append(prev_number)
        is_prev_number = False
    elif not line.startswith("C:\\") and line != "OK":
        # Если строка не путь к файлу и не "OK", возможно это число без точки?
        try:
            number = float(line)
            prev_number = number
            is_prev_number = True
        except ValueError:
            is_prev_number = False
    else:
        # Если мы встретили новую строку с путем или другую строку, и у нас было число без OK
        if is_prev_number and prev_number is not None:
            not_ok_numbers.append(prev_number)
            is_prev_number = False
        is_prev_number = False

# Добавляем последнее число, если оно не было обработано
if is_prev_number and prev_number is not None:
    not_ok_numbers.append(prev_number)

# Настройка параметров гистограммы
bin_width = 0.01

# Создаем один большой график
plt.figure(figsize=(14, 8))

# Объединяем все числа для определения границ бинов
all_numbers = ok_numbers + not_ok_numbers
bins = np.arange(min(all_numbers), max(all_numbers) + bin_width, bin_width)

# Строим гистограммы с наложением (alpha для прозрачности)
plt.hist(ok_numbers, bins=bins, alpha=0.7, color='green', 
         label=f'OK ({len(ok_numbers)} значений)', edgecolor='black', linewidth=0.5)
plt.hist(not_ok_numbers, bins=bins, alpha=0.7, color='red', 
         label=f'Не OK ({len(not_ok_numbers)} значений)', edgecolor='black', linewidth=0.5)

# Добавляем вертикальные линии для анализа
ok_mean = np.mean(ok_numbers) if ok_numbers else 0
not_ok_mean = np.mean(not_ok_numbers) if not_ok_numbers else 0
ok_max = max(ok_numbers) if ok_numbers else 0
not_ok_min = min(not_ok_numbers) if not_ok_numbers else 0

plt.axvline(x=ok_mean, color='darkgreen', linestyle='--', linewidth=2, 
           label=f'Среднее OK: {ok_mean:.3f}')
plt.axvline(x=not_ok_mean, color='darkred', linestyle='--', linewidth=2, 
           label=f'Среднее Не OK: {not_ok_mean:.3f}')

# Находим максимальное значение гистограммы для лучшего масштабирования
max_count = max(
    max(np.histogram(ok_numbers, bins=bins)[0]) if ok_numbers else 0,
    max(np.histogram(not_ok_numbers, bins=bins)[0]) if not_ok_numbers else 0
)

# Увеличиваем немного верхнюю границу для текста
plt.ylim(0, max_count * 1.1)

# Добавляем статистическую информацию в виде текста
stats_text = f'''
Статистика:
OK: {len(ok_numbers)} значений
  Мин: {min(ok_numbers):.3f}  Макс: {max(ok_numbers):.3f}
  Среднее: {np.mean(ok_numbers):.3f}  Медиана: {np.median(ok_numbers):.3f}

Не OK: {len(not_ok_numbers)} значений
  Мин: {min(not_ok_numbers):.3f}  Макс: {max(not_ok_numbers):.3f}
  Среднее: {np.mean(not_ok_numbers):.3f}  Медиана: {np.median(not_ok_numbers):.3f}

Общее: {len(all_numbers)} значений
Процент OK: {len(ok_numbers)/len(all_numbers)*100:.1f}%
'''

plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
         verticalalignment='top', fontsize=10, family='monospace')

# Настройки графика
plt.title('Распределение значений: OK (зеленый) vs Не OK (красный)', fontsize=16, pad=20)
plt.xlabel('Значение', fontsize=14)
plt.ylabel('Количество', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Улучшаем читаемость оси X
plt.xticks(rotation=45)

plt.show()

# Дополнительный анализ границы между OK и не OK
print("="*60)
print("АНАЛИЗ ГРАНИЦЫ МЕЖДУ OK И НЕ OK")
print("="*60)

# Находим перекрывающийся диапазон
ok_min, ok_max = min(ok_numbers), max(ok_numbers)
not_ok_min, not_ok_max = min(not_ok_numbers), max(not_ok_numbers)

print(f"Диапазон OK: [{ok_min:.3f}, {ok_max:.3f}]")
print(f"Диапазон Не OK: [{not_ok_min:.3f}, {not_ok_max:.3f}]")

# Находим перекрытие диапазонов
overlap_min = max(ok_min, not_ok_min)
overlap_max = min(ok_max, not_ok_max)

if overlap_min <= overlap_max:
    print(f"Перекрывающийся диапазон: [{overlap_min:.3f}, {overlap_max:.3f}]")
    
    # Считаем значения в перекрывающемся диапазоне
    ok_in_overlap = [x for x in ok_numbers if overlap_min <= x <= overlap_max]
    not_ok_in_overlap = [x for x in not_ok_numbers if overlap_min <= x <= overlap_max]
    
    print(f"  OK в перекрытии: {len(ok_in_overlap)} значений")
    print(f"  Не OK в перекрытии: {len(not_ok_in_overlap)} значений")
    print(f"  Всего в перекрытии: {len(ok_in_overlap) + len(not_ok_in_overlap)} значений")
else:
    print("Диапазоны НЕ перекрываются - есть четкая граница!")

print(f"\nМаксимальное значение OK: {ok_max:.3f}")
print(f"Минимальное значение Не OK: {not_ok_min:.3f}")

if ok_max < not_ok_min:
    print(f"✓ Четкая граница: все OK < {ok_max:.3f}, все Не OK > {not_ok_min:.3f}")
    print(f"  Разрыв между диапазонами: {not_ok_min - ok_max:.3f}")
elif not_ok_max < ok_min:
    print(f"✓ Четкая граница: все Не OK < {not_ok_max:.3f}, все OK > {ok_min:.3f}")
    print(f"  Разрыв между диапазонами: {ok_min - not_ok_max:.3f}")
else:
    print("✗ Нет четкой границы - диапазоны перекрываются")
    
# Сортируем значения для анализа границы
sorted_ok = sorted(ok_numbers)
sorted_not_ok = sorted(not_ok_numbers)

print(f"\nТоп-5 самых больших значений OK:")
for val in sorted_ok[-5:]:
    print(f"  {val:.3f}")

print(f"\nТоп-5 самых маленьких значений Не OK:")
for val in sorted_not_ok[:5]:
    print(f"  {val:.3f}")