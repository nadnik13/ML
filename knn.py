import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import operator
import random
import sklearn .base
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
import pylab as plt
from sklearn.feature_extraction.text import CountVectorizer


def load_dataset(filename):
    data = pd.read_csv(filename)
    y = np.ndarray.tolist(data['bin_month'].values)
    del data['bin_month']
    print(data[:1])
    data = preprocessing.normalize(data)  # нормализую данные
    print(data[:1])
    X = data
    return X, y


def get_response(neighbors):
    range_elem_on_class = [0, 0]
    for i in range(neighbors.size):
        range_elem_on_class[int(neighbors[i])] += 1
    if range_elem_on_class[0] < range_elem_on_class[1]:
        return 1
    else:
        return 0


def get_accuracy(write_value, predictions):
    correct = 0
    for x in range(len(write_value)):
        if write_value[x] == predictions[x]:
            correct += 1
    # print("correct: ", correct, "from ",len(write_value))
    return (correct/float(len(write_value))) * 100.0


class kNN(object):
    """ Simple kNN classifier """
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors # количество соседей
        self.X = None #признаки
        self.y = None #принадлежность классу

    def fit(self, X, y): #загружае данные для обучения
        self.X = X
        self.y = y

    def predict(self, X): # передаем тестовые данные
        y_predict = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            dists = np.zeros((self.X.shape[0], 2))  # создаем массив и высчитывае дистанцию для каждого элемента
            for j in range(self.X.shape[0]):
                dists[j][0] = np.linalg.norm(X[i] - self.X[j], ord = None) #дистанция
                dists[j][1] = j #индекс элемента, до которого считаем расстояние
            dists.view('i8,i8').sort(order=['f0'], axis=0) #нужно для сортировки по растоянию f0-колонка дистанции
            nearest_neighbors_dist = dists[:self.n_neighbors] # n первых (дистанция, индекс
            classes_nearest_neighbors = np.zeros(nearest_neighbors_dist.shape[0]) #массив классов ближайших элементов
            for p in range(nearest_neighbors_dist.shape[0]): # не вышло без for
                index = int(nearest_neighbors_dist[p][1]) #index елемента X
                classes_nearest_neighbors[p] = self.y[index] # значение класса 0 или 1
            y_predict[i] = get_response(classes_nearest_neighbors) # получаем, класс нового элемента
        return y_predict


if __name__ == '__main__':
    from sklearn.datasets import fetch_20newsgroups
    newsgroups_train = fetch_20newsgroups(subset='train')
    targets = ['sci.med', 'rec.sport.baseball']
    documents = fetch_20newsgroups(data_home='./', subset='all', categories=targets)

    y = documents.target[:100]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents.data[:100]).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    score = np.zeros(50)
    accuracy_my_knn_list = np.zeros(50)
    accuracy_lib_knn_list = np.zeros(50)
    for i in range(1, 50):
        my_knn = kNN(i)
        my_knn.fit(X_train, y_train)
        y_predict = my_knn.predict(X_test)
        accuracy_my_knn = get_accuracy(y_test, y_predict)
        accuracy_my_knn_list[i] = accuracy_my_knn

        lib_knn = KNeighborsClassifier(n_neighbors=i)
        lib_knn.fit(X_train, y_train)
        y_predict = lib_knn.predict(X_test)
        accuracy_lib_knn = get_accuracy(y_test, y_predict)
        accuracy_lib_knn_list[i] = accuracy_lib_knn
        score[i] = accuracy_my_knn - accuracy_lib_knn
    plt.figure('difference_text')
    plt.plot([i for i in range(0, 50)], score, color='black')
    plt.figure('cross_val_my_text')
    plt.bar([i for i in range(0, 50)], accuracy_my_knn_list, color='red')
    plt.figure('cross_val_lib_text')
    plt.bar([i for i in range(0, 50)], accuracy_lib_knn_list, color='green')

    features, classes,  = load_dataset("reload_forest_fire_simple.csv")

    X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.34, random_state=43)
    score = np.zeros(50)
    for i in range(1, 50):
        my_knn = kNN(i)
        my_knn.fit(X_train, y_train)
        y_predict = my_knn.predict(X_test)
        accuracy_my_knn = get_accuracy(y_test, y_predict)

        lib_knn = KNeighborsClassifier(n_neighbors=i)
        lib_knn.fit(X_train, y_train)
        y_predict = lib_knn.predict(X_test)
        accuracy_lib_knn = get_accuracy(y_test, y_predict)
        score[i] = accuracy_lib_knn - accuracy_my_knn
    plt.figure('difference')
    plt.plot([i for i in range(0, 50)], score, color='black')

    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for i in range(1, 51):
        clf = KNeighborsClassifier(n_neighbors=i)
        score = cross_val_score(cv=k_fold, estimator=clf, scoring='accuracy', X=features, y=classes).mean()
        scores.append(score)
    plt.figure('cross_val')
    plt.bar([i for i in range(0, 50)], scores, color='red')
    print(max(scores), scores.index(max(scores)) + 1)
    features_scaled = scale(X=features)
    plt.show()

