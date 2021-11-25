import pandas as pd
import numpy as np


class KNN:


    def __init__(self, k):
        self.k = k


    def euclidean(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))


    # Guardar aqui na memória os valores anteriores
    def fit(self, X, y):
        self._Xt = X
        self._Yt = y


    def predict(self, X):
        y_pred = [self.predict_aux(x) for x in X.to_numpy()]
        return np.array(y_pred)

    def predict_aux(self, x):
        distances = [self.euclidean(i, x) for i in self._Xt.to_numpy()]
        # index do menor valor
        idx = np.argsort(distances)[:self.k]
        labels = [self._Yt.to_numpy()[i][0] for i in idx]
        classe = max(set(list(labels)), key=labels.count)

        return classe
