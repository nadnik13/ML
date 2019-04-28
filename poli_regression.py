import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pylab as plt
import string
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_text(filename):
    with open(filename, 'r') as f:
        data_text = f.readlines()  # считываем строки из файл
        data_text_word = []  # объявление списка
        for line in data_text:  # обработка каждой из строк
            line = line.lower()  # приведение к нижнему регистру
            line = "".join(
                c for c in line if not c.isdigit() and c not in string.punctuation)  # удаление цифр и знаков припинания
            line = line.split()  # преобразование строки к списку слов
            data_text_word.extend(line)  # добавление элементов в список всех слов
        # конструирование словаря слово:число вхождений в текст
        dict_word_counts = {i: data_text_word.count(i) for i in list(set(data_text_word))}
        for i in dict_word_counts:
            print("%s" % (i.ljust(19)), dict_word_counts[i])  # ljust(n) - левоориентированный вывод n знаков


def load_dataset(filename):
    data = pd.read_csv(filename, index_col=False)
    data['DMC'] = data['DMC'] / np.linalg.norm(data['DMC'])
    X = data['X'] / np.linalg.norm(data['X'])
    y = data['Y'] / np.linalg.norm(data['Y'])
    return X, y


def f(a, x):
    value = 0
    for i in range(0, len(a)):
        value += a[i]*x**i
    return value


def sse(y, y_predict):
    r = np.array(y - y_predict)
    return np.dot(r.transpose(), r)


def reg_fit(X, y, degree):
    A = np.zeros((len(X),degree))
    A = pd.DataFrame(A)
    A[A.columns[0]] = np.ones(len(X))
    for i in range(1, degree):
        A[A.columns[i]] = np.array(X**i)
    w = np.dot(np.linalg.inv(np.dot(A.transpose(),A)),np.dot(A.transpose(),y))
    y_new = f(w,X)
    SSE = sse(y,y_new)
    return w, SSE


def reg_predict(X_test, koef):
    y_new = 0
    for i in range(0, degree):
        y_new += koef[i] * X_test ** i
    return y_new


if __name__ == '__main__':
    features, classes,  = load_dataset("reload_forest_fire_simple.csv")
    X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.34, random_state=4)
    print(len(X_train))

    plt.figure('Space2')
    area = 10  # point radius
    print(area)
    colors = X_train * 10 + 70
    plt.scatter(X_train,y_train, c=colors, s=area, alpha=1)

    degree = 3
    print("Degree: ", degree)
    koef, SSE_train = reg_fit(X_train, y_train, degree+1)
    print(koef)
    print("SSE_train=", SSE_train/len(y_train))
    y_predict= reg_predict(X_test, koef)
    SSE_test = sse(y_test, y_predict)
    print("SSE_test = ", SSE_test/len(y_test))

    i = np.arange(0, 0.08, 0.001)
    plt.plot(i, f(koef, i))

    degree = 2
    print("Degree: ", degree)
    koef, SSE_train = reg_fit(X_train, y_train, degree + 1)
    print(koef)
    print("SSE_train=", SSE_train/len(y_train))
    y_predict = reg_predict(X_test, koef)
    SSE_test = sse(y_test, y_predict)
    print("SSE_test = ", SSE_test/len(y_test))

    i = np.arange(0, 0.08, 0.001)
    plt.plot(i, f(koef, i))

    degree = 1
    print("Degree: ", degree)
    koef, SSE_train = reg_fit(X_train, y_train, degree + 1)
    print(koef)
    print("SSE_train=", SSE_train/len(y_train))
    y_predict = reg_predict(X_test, koef)
    SSE_test = sse(y_test, y_predict)
    print("SSE_test = ", SSE_test/len(y_test))

    i = np.arange(0, 0.08, 0.001)
    plt.plot(i, f(koef, i))

    plt.figure('Wrong')
    max_degree = 7
    SSE_list = np.zeros(max_degree)
    for i in range(0, max_degree):
         koef, SSE_list[i] = reg_fit(X_train, y_train, i+1)
    i = range(0, max_degree)
    plt.bar(i, SSE_list[i])
    plt.show()