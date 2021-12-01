import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class TableCleaner:
    def __init__(self):
        self.model = RandomForestRegressor()
        self.success_col = None
        self.success_row = None

    def clean_columns(self, X, Y):
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)
        self.success_col = []
        for i in range(X.shape[1]):
            try:
                self.model.fit(X.iloc[:, [i]], Y.values.ravel())
                self.success_col.append(i)
            except:
                continue
        return self.success_col

    def clean_rows(self, X, Y=None):
        X = pd.DataFrame(X)
        self.success_row = [
            i for i, b in enumerate(list(X.isnull().any(axis=1))) if not b
        ]
        return self.success_row
