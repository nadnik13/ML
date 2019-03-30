import pandas as pd
import pylab as plt
import numpy as np
import string

with open('wikibig.text', 'r') as f:
    data_text = f.readlines() #считываем строки из файл
    data_text_word = [] #объявление списка
    for line in data_text: # обработка каждой из строк
        line = line.lower() # приведение к нижнему регистру
        line = "".join(c for c in line if not c.isdigit() and c not in string.punctuation) # удаление цифр и знаков припинания
        line = line.split() #преобразование строки к списку слов
        data_text_word.extend(line) # добавление элементов в список всех слов
    # конструирование словаря слово:число вхождений в текст
    dict_word_counts = {i: data_text_word.count(i) for i in list(set(data_text_word))}
    for i in dict_word_counts:
        print("%s" % (i.ljust(19)), dict_word_counts[i]) #ljust(n) - левоориентированный вывод n знаков

data = pd.read_csv('forestfires.csv')
#создание словарей
month_name = {month: i for i, month in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])}
day_name = {day: i for i, day in enumerate(['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'])}

data['month'] = data['month'].map(month_name)
data['day'] = data['day'].map(day_name)
# статистические признаки
print('Static signs \n')
print('x\n', data['X'].describe())
print('\ny\n', data['Y'].describe())
print('\nmonth\n', data['month'].describe())
print('\nday\n', data['day'].describe())
print('\nFFMC\n', data['FFMC'].describe())
print('\nDC\n', data['DC'].describe())
print('\nwind\n', data['wind'].describe())
#построение графиков
plt.figure('X')
values = data['X'].value_counts()
plt.figure('X')
plt.bar(values.index, values)

plt.figure('Y')
plt.figure('Y')
values = data['Y'].value_counts()
plt.figure('Y')
plt.bar(values.index, values)

values = data['month'].value_counts()
plt.figure('month')
plt.bar(values.index, values)

values = data['day'].value_counts()
plt.figure('day')
plt.bar(values.index, values)

plt.figure('wind')
plt.hist(data['wind'],bins='auto', density=True)

plt.figure('rain')
plt.hist(data['rain'],bins=200, density=True)

plt.figure('DC')
plt.hist(data['DC'],bins=100, density=True)

plt.figure('FFMC')
plt.hist(data['FFMC'],bins='auto', density=True)

plt.figure('RH')
plt.hist(data['RH'],bins='auto', density=True)

plt.figure('ISI')
plt.hist(data['ISI'],bins='auto', density=True)

plt.figure('Burned area')
plt.ylim([0, 0.06])
plt.hist(data['area'],bins='auto', density=True)

plt.figure('Space')
N = len(data['X'])
colors = np.random.rand(N)# point color
area = (data['FFMC'])**5/10e7  # point radius
print(area)
plt.scatter(data['X'], data['Y'], s=area, c=colors, alpha=1)

plt.show()

