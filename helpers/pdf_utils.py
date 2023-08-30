import pandas as pd


def _repr_latex_(self):
    return "{\centering{%s}\par}" % self.to_latex()


def head(self, n=5):
    if len(self.shape) > 1 and self.shape[1] > 5:
        df = self.iloc[:n, [0, 1, -2, -1]].copy()
        df.insert(2, "...", ["..."] * df.shape[0])
        return df
    else:
        if len(self.shape) > 1:
            return self.iloc[:n, :]
        return self.iloc[:n]
