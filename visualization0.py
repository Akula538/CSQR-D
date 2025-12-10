import matplotlib.pyplot as plt

# Чтение данных из файла и подсчет вердиктов
verdicts = {'Успешно': 0, 'None': 0, 'CTF': 0}

file = "evr_test_log.txt"
# file = "results\evr_test_log005.txt"


with open(file, 'r') as f:
    for line in f:
        if '->' in line:
            result = line.split('->')[1].strip()
            
            if result == 'None':
                verdicts['None'] += 1
            elif result == 'CTF':
                verdicts['CTF'] += 1
            else:
                verdicts['Успешно'] += 1

# Подготовка данных для диаграммы
labels = list(verdicts.keys())
sizes = list(verdicts.values())
colors = ['#4CAF50', '#FF5252', '#2196F3']  # зеленый, красный, синий

# Построение круговой диаграммы
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Соотношение вердиктов считывания кодов')
plt.axis('equal')  # Чтобы диаграмма была круглой

plt.tight_layout()
plt.show()