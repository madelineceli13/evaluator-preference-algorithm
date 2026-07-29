# test_monotone_gbm.py
# ---------------------------------------------------------------------------
# Minimal smoke test for the monotone gradient-boosting baseline ONLY.
#
# Deliberately imports nothing from `functions` (no cvxpy) and nothing from
# `monotone_nn` (no torch), so this exercises exactly one path: sklearn +
# xgboost. If this runs clean, the GBM estimator works end-to-end and can be
# driven as its own process.
#
# Run (with the venv active):
#     python test_monotone_gbm.py
# ---------------------------------------------------------------------------

import numpy as np
from sklearn.model_selection import train_test_split

from monotone_gbm import (
    GBM_analysis, cross_validation_gbm, _train_one_gbm, predict_gbm,
)


# ------------------------- synthetic monotone data ------------------------- #
def make_data(function_name="cobb_douglas", N=400, d=2, num_vals=5, sd=0.2,
              seed=0):
    """
    Generate integer-grid inputs X in [1, num_vals]^d with a monotone,
    [0,1]-normalized ground truth f(X), plus Gaussian noise. Mirrors the
    families used in synthetic_simulations.py.
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(1, 2, size=d)
    X = rng.integers(1, num_vals + 1, size=(N, d)).astype(float)

    if function_name == "linear":
        fX = (X @ a) / (num_vals * np.sum(a))
    elif function_name == "cobb_douglas":
        fX = np.prod(np.power(X, a), axis=1) / (num_vals ** np.sum(a))
    elif function_name == "leontief":
        fX = np.min(X * a, axis=1) / (np.max(a) * num_vals)
    else:
        raise ValueError(function_name)

    Y = fX + rng.normal(0, sd, size=N)
    return X, Y, fX


# ------------------------------ monotonicity ------------------------------- #
def check_monotone(bundle, d=2, num_vals=5, y_min=0.0, y_max=1.0):
    """
    Confirm predictions are non-decreasing in every coordinate: step one axis
    up by 1 from a fixed base point and verify the prediction never drops.
    """
    base = np.full((1, d), (num_vals + 1) / 2.0)
    ok = True
    for j in range(d):
        prev = -np.inf
        for v in range(1, num_vals + 1):
            pt = base.copy()
            pt[0, j] = v
            yhat = predict_gbm(bundle, pt, y_min, y_max)[0]
            if yhat < prev - 1e-6:
                ok = False
            prev = yhat
    return ok


# ---------------------------------- main ----------------------------------- #
def main():
    y_min, y_max = 0.0, 1.0
    X, Y, fX = make_data(function_name="cobb_douglas", N=400, d=2,
                         num_vals=5, sd=0.2, seed=0)

    # Same style of outer split used across the codebase.
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    # --- 1) single-config smoke test ------------------------------------- #
    # Give it a validation fold so early stopping picks the tree count, exactly
    # as the real pipeline does.
    X_tr, X_val, Y_tr, Y_val = train_test_split(
        X_train, Y_train, test_size=0.25, random_state=42)

    hp = {
        "n_estimators": 400,      # ceiling; early stopping picks the real count
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
        "patience": 40,
    }
    print(">>> single-config fit")
    bundle, mse_tr, mse_te = GBM_analysis(
        X_tr, Y_tr, X_test, Y_test, hp, y_min, y_max,
        printer_friend=True, X_val=X_val, Y_val=Y_val, seed=0, directions=None)

    mono_ok = check_monotone(bundle, d=2, num_vals=5, y_min=y_min, y_max=y_max)
    print(f"monotone predictions: {'OK' if mono_ok else 'VIOLATED'}")

    # --- 2) tiny tuning sweep (exercises the CV path used in the sim) ----- #
    print("\n>>> small cross-validation sweep")
    tiny_grid = {
        "n_estimators": [400],
        "max_depth": [2, 3],
        "learning_rate": [0.05, 0.1],
        "subsample": [1.0],
        "colsample_bytree": [1.0],
        "min_child_weight": [1],
        "reg_lambda": [1.0],
        "patience": [40],
    }
    MSE_train, MSE_test, best_hp, _, seed_mses = cross_validation_gbm(
        X_train, Y_train, X_test, Y_test, y_min, y_max,
        param_grid=tiny_grid, save=False, final_seeds=(0, 1), directions=None)

    print(f"\nbest hp     : {best_hp}")
    print(f"test MSE    : {MSE_test:.4f}")
    print(f"seed spread : {np.std(seed_mses):.4f}")
    print("\nGBM-only test completed without an OpenMP crash.")


if __name__ == "__main__":
    main()
