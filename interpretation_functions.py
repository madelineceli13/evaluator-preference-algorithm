import numpy as np
from functions import iso_fit_many, unique_xvals, cross_validation
from sklearn.model_selection import train_test_split
from itertools import chain, combinations
from math import factorial
from sklearn.metrics import mean_squared_error
from functools import lru_cache


def unique_xvals(num_vals, num_criteria):
    # Create a grid of all points in [n]^d
    grids = [np.arange(1,num_vals+1) for _ in range(num_criteria)]   # list of arrays [1,2,...,n] repeated d times
    mesh = np.meshgrid(*grids, indexing='ij')  # d arrays of shape (n,n,...,n)
    X_grid = np.vstack([g.ravel() for g in mesh]).T  # stack and transpose to get shape (n^d, d)
    return X_grid

## ----------------------------------- CRITERIA IMPORTANCE -----------------------------------


# we want to compute the shapley value for each feature
# we do this by fitting a function f(S) that predicts the output for a subset of features S
def weight_vector(df):
    x_vals = np.asarray(df.iloc[:, 1:-1])
    f_vals = np.asarray(df.iloc[:, -1])
    numvals = np.unique(x_vals).shape[0]
    d = x_vals.shape[1]

    x_grid = unique_xvals(num_vals=numvals, num_criteria=d)   # (numvals**d, d)
    f_grid = iso_fit_many(x_grid, x_vals, f_vals)      # (numvals**d,)
    # tuple of levels -> row index in x_grid
    lookup = {tuple(row): k for k, row in enumerate(x_grid)}

    weights = np.zeros(d)
    for i in range(d):
        diffs = []
        for k, row in enumerate(x_grid):
            if row[i] == numvals:      # no neighbor above
                continue
            nbr = row.copy()
            nbr[i] += 1
            diffs.append((f_grid[lookup[tuple(nbr)]] - f_grid[k]) ** 2)
        weights[i] = np.sqrt(np.mean(diffs))
    print(weights)
    return weights


# we want to compute the shapley value for each feature
# we do this by fitting a function f(S) that predicts the output for a subset of features S

# construct power sets without ith criterion 
def powerset_without(d, i):
    others = [j for j in range(d) if j != i]
    return list(chain.from_iterable(
        combinations(others, r) for r in range(len(others) + 1)
    ))


def compute_shapley_values(X, Y):
    d = X.shape[1]
    y_min, y_max = np.min(Y), np.max(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    @lru_cache(maxsize=None)
    def mse(features):                      # features: sorted tuple of column indices
        if len(features) == 0:
            baseline = np.full_like(Y_test, np.mean(Y_train), dtype=float)
            return mean_squared_error(Y_test, baseline)
        cols = list(features)
        _, mse_test, _, _ = cross_validation(
            X_train[:, cols], Y_train, X_test[:, cols], Y_test,
            y_min, y_max, save=False,
        )
        return mse_test

    shapley_values = np.zeros(d)
    for i in range(d):
        for S in powerset_without(d, i):
            s = len(S)
            weight = factorial(s) * factorial(d - s - 1) / factorial(d)
            mse_without = mse(S)
            mse_with = mse(tuple(sorted(S + (i,))))
            shapley_values[i] += weight * (mse_without - mse_with)

    return shapley_values
            