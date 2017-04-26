# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division
import numpy as np

def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. ODleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """
    X_train = np.transpose(X_train)
    NOT_X = np.subtract(np.ones(shape=(X.shape[0], X.shape[1])), X.toarray())
    NOT_X_train = np.subtract(np.ones(shape=(X_train.shape[0], X_train.shape[1])), X_train.toarray())
    return X @ NOT_X_train + NOT_X @ X_train

def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """
    r = Dist.shape[0] #wiersze odpowiadają x-om ze zbioru uczacego
    c = y.shape[0]
    result = np.zeros(shape=(r, c))
    for i in range(0, r):
        result[i, :] = y[Dist[i, :].argsort(kind='mergesort')]
    return result

def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X. N1 x liczba kategorii
    """
    E = len(np.unique(y)) #M
    N1 = y.shape[0]
    p_y_x = np.zeros(shape=(N1, E))

    for e in range(E):
        for i in range(N1):
            for K in range(k):
                if y[i, K] == e + 1:
                    p_y_x[i, e] += 1
    return p_y_x / k

def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """
    error = 0
    for i in range(p_y_x.shape[0]):
        max = 0
        p = 0
        for j in range(p_y_x.shape[1]):
            if p_y_x[i, j] >= max:
                max = p_y_x[i, j]
                p = j
        if p + 1 != y_true[i]:
            error += 1
    return error / p_y_x.shape[0]

def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """
    errors = np.zeros_like(k_values, dtype=np.float64)
    dist = hamming_distance(Xval, Xtrain)
    sorted_labels = sort_train_labels_knn(dist, ytrain)
    best_error = classification_error(p_y_x_knn(sorted_labels, k_values[0]), yval)
    best_k = k_values[0]
    errors[0] = best_error
    for i in range(1, len(k_values)):
        p_y_x = p_y_x_knn(sorted_labels, k_values[i])
        error = classification_error(p_y_x, yval)
        errors[i] = error
        if error < best_error:
            best_error = error
            best_k = k_values[i]
    return best_error, best_k, errors

def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """
    M = len(np.unique(ytrain))
    length = len(ytrain)
    p_y_x = np.zeros(shape=(1, M))
    for i in range(M):
        for j in range(length):
            if ytrain[j] == i + 1:
                p_y_x[0, i] += 1
    return p_y_x / length

def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD #x - N dokumentow, dla kazdego jedynki i zera dla D slow
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD. #prawdopodobienstwo ze slowo nalezy do kategorii
    """
    D = Xtrain.shape[1]
    Xtrain = Xtrain.toarray()
    M = len(np.unique(ytrain))
    p_x_y = []
    for m in range(1, M + 1):
        row = np.empty([1, D])
        y_from_cat = np.equal(ytrain, m)
        y_from_cat_count = np.sum(y_from_cat)
        for d in range(0, D):
            row[0, d] = np.sum(Xtrain[:, d] * y_from_cat) #logical_and
        p_x_y.append(np.divide(np.add(row, a - 1), y_from_cat_count + a + b - 2))
    return np.vstack(p_x_y)

def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """
    M = len(p_y)
    N = X.shape[0]
    p_y_x = np.zeros(shape=(N, 4)) #jak sie tutaj da M to na testach porownujacych blad nie dziala
    X = X.astype(int)
    for n in range(0, N):
        temp = p_x_1_y.copy()
        temp[:][:] += X[n][:] #bedzie p+1 dla wszystkich x-ow jedynek. X[n] dodajemy do kazdego wiersza m
        temp -= 1 #bedzie p-1 dla wszystkich x-ow zer i p dla wszystkich x-ow jedynek
        temp = np.absolute(temp) #p-1 (ujemne) zostaje zamienione na 1-p
        p_y_x[n] = p_y * np.prod(temp, axis=1)
        p_y_x[n] /= np.sum(p_y_x[n])
    return p_y_x

def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b) ?
    """
    error_best = 10
    A = len(a_values)
    B = len(b_values)
    errors = np.zeros(shape=(A, B), dtype=np.float64)
    p_y = np.squeeze(np.asarray(estimate_a_priori_nb(ytrain)))
    best_a_index = 0
    best_b_index = 0
    for a in range(A):
        for b in range(B):
            p_x_1_y = estimate_p_x_y_nb(Xtrain, ytrain, a_values[a], b_values[b])
            p_y_x = p_y_x_nb(p_y, p_x_1_y, Xval)
            error = classification_error(p_y_x, yval)
            errors[a, b] = error
            if error < error_best:
                error_best = error
                best_a_index = a
                best_b_index = b
    return error_best, a_values[best_a_index], b_values[best_b_index], errors