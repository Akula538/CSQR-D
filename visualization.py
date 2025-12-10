import matplotlib.pyplot as plt
import numpy as np

file = "evr_test_log.txt"
file = "results\evr_test_log005.txt"

# Чтение данных из файла
data = {}
with open(file, 'r') as f:
    for line in f:
        if '->' in line:
            filename, result = line.split('->')
            filename = filename.strip()
            result = result.strip()
            
            # Извлекаем тип кода из названия файла
            code_type = None
            if '_H_' in filename:
                code_type = 'H'
            elif '_L_' in filename:
                code_type = 'L'
            elif '_M_' in filename:
                code_type = 'M'
            elif '_Q_' in filename:
                code_type = 'Q'
            
            if code_type:
                if code_type not in data:
                    data[code_type] = {'total': 0, 'success': 0, 'none': 0, 'ctf': 0}
                
                data[code_type]['total'] += 1
                if result == 'None':
                    data[code_type]['none'] += 1
                elif result == 'CTF':
                    data[code_type]['ctf'] += 1
                else:
                    data[code_type]['success'] += 1

# Подготовка данных для диаграммы
code_types = ['L', 'M', 'Q', 'H']
success_rates = []
none_rates = []
ctf_rates = []

for code_type in code_types:
    total = data[code_type]['total']
    success_rates.append(data[code_type]['success'] / total)
    none_rates.append(data[code_type]['none'] / total)
    ctf_rates.append(data[code_type]['ctf'] / total)

# Построение диаграммы
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.8
x_pos = np.arange(len(code_types))

# Создаем stacked bar chart
bars_success = ax.bar(x_pos, success_rates, bar_width, label='Успешно считано', color='lime')
bars_none = ax.bar(x_pos, none_rates, bar_width, bottom=success_rates, label='None', color='red')
bars_ctf = ax.bar(x_pos, ctf_rates, bar_width, 
                  bottom=[success_rates[i] + none_rates[i] for i in range(len(code_types))], 
                  label='CTF', color='blue')

# Настройка внешнего вида
ax.set_xlabel('Уровень коррекции ошибок')
ax.set_ylabel('Доля')
ax.set_title('Результаты считывания кодов по типам')
ax.set_xticks(x_pos)
ax.set_xticklabels(code_types)
ax.legend()

# Добавление подписей с процентами
for i, (success, none, ctf) in enumerate(zip(success_rates, none_rates, ctf_rates)):
    ax.text(i, success/2, f'{success:.1%}', ha='center', va='center', color='white', fontweight='bold')
    ax.text(i, success + none/2, f'{none:.1%}', ha='center', va='center', color='white', fontweight='bold')
    ax.text(i, success + none + ctf/2, f'{ctf:.1%}', ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()

# # Вывод статистики
# print("Статистика по типам кодов:")
# for code_type in code_types:
#     total = data[code_type]['total']
#     print(f"\n{code_type}:")
#     print(f"  Всего тестов: {total}")
#     print(f"  Успешно: {data[code_type]['success']} ({data[code_type]['success']/total:.1%})")
#     print(f"  None: {data[code_type]['none']} ({data[code_type]['none']/total:.1%})")
#     print(f"  CTF: {data[code_type]['ctf']} ({data[code_type]['ctf']/total:.1%})")