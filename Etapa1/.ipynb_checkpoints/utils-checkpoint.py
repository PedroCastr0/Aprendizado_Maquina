import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class MySimpleImputer(SimpleImputer):


    def fit(self, X, y=None):
        self.columns = X.columns
        return super().fit(X, y)


    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=self.columns)


class MyScaler(StandardScaler):


    def fit(self, X, y=None):
        self.columns = X.columns
        return super().fit(X, y)


    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=self.columns)
