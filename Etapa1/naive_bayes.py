import numpy as np
import pandas as pd
from utils import *


class NaiveBayes:


    def fit(self, X, y):
        self._num_classes = np.unique(y)
        self._mean = X.groupby(y.columns.values[0]).mean().to_numpy()
        self._var = X.groupby(y.columns.values[0]).var().to_numpy()
        self._priori = X.groupby(y.columns.values[0]).apply(lambda x: len(x) / len(X))


    def gaussian_pdf(self, idx, X):
        mean = self._mean[idx]
        var = self._var[idx]
        num = np.exp(-((X - mean) ** 2) / (2 * var))
        den = np.sqrt(2 * np.pi * var)

        return num / den


    def predict(self, X):
        y_pred = [self.predict_aux(x) for x in X.to_numpy()]
        return np.array(y_pred)


    def predict_aux(self, x):
        res = []

        # Foi inserido logaritmo para nao causar possiveis overflow
        for idx, classe in enumerate(self._num_classes):
            priori = np.log((self._priori[idx]))
            posteriori = np.log(np.prod((self.gaussian_pdf(idx, x))))
            posteriori = priori + posteriori
            res.append(posteriori)

        return self._num_classes[np.argmax(res)]
