import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class TableCleaner:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=1, max_depth=1, n_jobs=-1)
        self.success_col = None
        self.success_row = None

    def clean_columns(self, X, Y):
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)
        self.success_col = []

        cols = [i for i in range(X.shape[1])]
        waiting = [cols]
        while len(waiting) > 0:
            cols = waiting.pop()
            try:
                self.model.fit(X.iloc[:, cols], Y.values.ravel())
                self.success_col += cols
            except:
                if len(cols) > 2:
                    waiting.append(cols[: int(len(cols) / 2)])
                    waiting.append(cols[int(len(cols) / 2) :])

        return self.success_col

    def clean_rows(self, X, Y=None):
        X = pd.DataFrame(X)
        self.success_row = [
            i for i, b in enumerate(list(X.isnull().any(axis=1))) if not b
        ]
        return self.success_row
