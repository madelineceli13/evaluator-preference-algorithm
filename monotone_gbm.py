# monotone_gbm.py
# ---------------------------------------------------------------------------
# Monotone-constrained gradient-boosting baseline for the isotonic-preference
# pipeline. Uses XGBoost's built-in `monotone_constraints`.
#
#   pip install xgboost
#
# Design goals (identical to monotone_nn.py):
#   * Reuse the EXACT same train/val/test partitions as `cross_validation`
#     (60/20/20, random_state=42) so the comparison is apples-to-apples.
#   * Mirror the analysis-function signatures and the per-sample-error /
#     standard-error reporting convention used in the rest of the codebase.
#
# The public entry points parallel monotone_nn.py one-for-one:
#     monotone_nn_analysis   ->  GBM_analysis
#     cross_validation_nn    ->  cross_validation_gbm
#     compute_nn_stats       ->  compute_gbm_stats
#     run_monotone_nn        ->  run_monotone_gbm
# ---------------------------------------------------------------------------

import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid

import xgboost as xgb


# --------------------------------------------------------------------------- #
#  Default hyper-parameter grid.                                               #
#  n_estimators is only an UPPER BOUND: when a validation fold is supplied,    #
#  early stopping (patience rounds) picks the effective number of trees, and   #
#  the final refit reuses that count. Shallow trees are the right regime for   #
#  the low-d, few-level criteria spaces here (d ~ 2-6, m ~ 4-7).               #
# --------------------------------------------------------------------------- #
DEFAULT_PARAM_GRID = {
    "n_estimators": [500],          # ceiling; early stopping selects the real count
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [1.0],
    "min_child_weight": [1, 5],
    "reg_lambda": [1.0],
    "patience": [50],               # early_stopping_rounds
}


# --------------------------------------------------------------------------- #
#  Small helpers                                                               #
# --------------------------------------------------------------------------- #
def _as_float_2d(A):
    """Strip DataFrame column names etc. so DMatrix train/predict stay aligned."""
    return np.asarray(A, dtype=np.float64)


def _as_float_1d(Y):
    return np.asarray(Y, dtype=np.float64).ravel()


def _mono_constraint_str(directions, input_dim):
    """
    XGBoost monotone_constraints spec.

    directions : optional 1-D array of {-1, 0, +1}, length input_dim.
                 None -> all +1 (non-decreasing), which matches the isotonic
                 setup (coordinate-wise partial order, nonnegative coefficients).

    Returned as the "(1,1,...,1)" string form, which every XGBoost version
    accepts (tuple/list forms are only accepted by newer releases).
    """
    if directions is None:
        directions = np.ones(input_dim, dtype=int)
    directions = [int(v) for v in np.asarray(directions).ravel()]
    if len(directions) != input_dim:
        raise ValueError(
            f"directions has length {len(directions)} but input_dim is {input_dim}")
    return "(" + ",".join(str(v) for v in directions) + ")"


def _make_params(hp, input_dim, seed, directions):
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": int(hp["max_depth"]),
        "eta": float(hp["learning_rate"]),
        "subsample": float(hp["subsample"]),
        "colsample_bytree": float(hp["colsample_bytree"]),
        "min_child_weight": float(hp["min_child_weight"]),
        "lambda": float(hp["reg_lambda"]),
        "monotone_constraints": _mono_constraint_str(directions, input_dim),
        "seed": int(seed),
        "verbosity": 0,
    }


# --------------------------------------------------------------------------- #
#  Training + prediction                                                       #
# --------------------------------------------------------------------------- #
def _train_one_gbm(X_train, Y_train, hp, X_val=None, Y_val=None, seed=0,
                   directions=None, n_rounds_override=None):
    """
    Train a single monotone GBM. Returns a 'bundle' dict holding the fitted
    booster and the number of trees to use at prediction time.

    If (X_val, Y_val) are given, early stopping (hp['patience'] rounds) selects
    the boosting round of lowest validation RMSE; bundle['best_rounds'] records
    that count. Otherwise the booster trains for `n_rounds_override` rounds if
    provided (used at the final refit, where the val fold has been folded back
    into train and we reuse the round count chosen during CV), else for
    hp['n_estimators'].

    Trees are scale-invariant, so unlike the NN baseline there is no input or
    output standardization; the training Y-range is applied at predict time via
    clipping, mirroring the isotonic convention.
    """
    Xtr = _as_float_2d(X_train)
    Ytr = _as_float_1d(Y_train)
    input_dim = Xtr.shape[1]

    params = _make_params(hp, input_dim, seed, directions)
    dtrain = xgb.DMatrix(Xtr, label=Ytr)

    use_val = X_val is not None and Y_val is not None
    if use_val:
        dval = xgb.DMatrix(_as_float_2d(X_val), label=_as_float_1d(Y_val))
        booster = xgb.train(
            params, dtrain,
            num_boost_round=int(hp["n_estimators"]),
            evals=[(dval, "val")],
            early_stopping_rounds=int(hp["patience"]),
            verbose_eval=False,
        )
        best_rounds = int(booster.best_iteration) + 1
    else:
        n_rounds = int(n_rounds_override) if n_rounds_override is not None \
            else int(hp["n_estimators"])
        booster = xgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            verbose_eval=False,
        )
        best_rounds = n_rounds

    return {"booster": booster, "best_rounds": best_rounds,
            "n_features": input_dim}


def predict_gbm(bundle, X, y_min=None, y_max=None):
    """Predict and (optionally) clip to the training range."""
    dX = xgb.DMatrix(_as_float_2d(X))
    y = bundle["booster"].predict(
        dX, iteration_range=(0, bundle["best_rounds"])).ravel()
    if y_min is not None and y_max is not None:
        y = np.clip(y, y_min, y_max)
    return y


def error_gbm(bundle, X_vals, Y_vals, y_min, y_max):
    """Per-sample squared errors (parallel to `error_nn` / `error_isotonic`)."""
    y_hat = predict_gbm(bundle, X_vals, y_min, y_max)
    return (y_hat - _as_float_1d(Y_vals)) ** 2


# --------------------------------------------------------------------------- #
#  Analysis function (mirrors `monotone_nn_analysis`)                          #
# --------------------------------------------------------------------------- #
def GBM_analysis(X_train, Y_train, X_test, Y_test, hp, y_min, y_max,
                 printer_friend=True, X_val=None, Y_val=None, seed=0,
                 directions=None, n_rounds_override=None):
    """
    Fit a monotone GBM with hyper-parameters `hp`; return (bundle, MSE_train,
    MSE_test). If (X_val, Y_val) are provided they drive early stopping, and
    bundle['best_rounds'] holds the selected number of trees.
    """
    bundle = _train_one_gbm(X_train, Y_train, hp, X_val=X_val, Y_val=Y_val,
                            seed=seed, directions=directions,
                            n_rounds_override=n_rounds_override)
    MSE_train = float(np.mean(error_gbm(bundle, X_train, Y_train, y_min, y_max)))
    MSE_test = float(np.mean(error_gbm(bundle, X_test, Y_test, y_min, y_max)))

    if printer_friend:
        print("---------------------------------")
        print("Monotone GBM results")
        print(f"hparams: {hp}")
        print(f"trees used: {bundle['best_rounds']}")
        print(f"MSE_train: {np.round(MSE_train, 3)}")
        print(f"MSE_test:  {np.round(MSE_test, 3)}")
        print("---------------------------------")

    return bundle, MSE_train, MSE_test


# --------------------------------------------------------------------------- #
#  Cross-validation / tuning wrapper (mirrors `cross_validation_nn`)           #
# --------------------------------------------------------------------------- #
def cross_validation_gbm(X_train, Y_train, X_test, Y_test, y_min, y_max,
                         param_grid=None, save=True, dir_name="Athena",
                         final_seeds=(0, 1, 2, 3, 4), directions=None):
    """
    Tune the monotone GBM on the SAME validation fold used by `cross_validation`
    (train_test_split(..., test_size=0.25, random_state=42)), then refit the
    best config on the full 80% train over several seeds and evaluate on test.

    Returns: MSE_train, MSE_test, best_hp, test_errors_ensemble, seed_test_mses
      * MSE_test / test_errors_ensemble come from the seed-ensemble prediction
        (mean over seeds), so the per-point SE is comparable to SE_cv / SE_iso.
      * seed_test_mses is the list of single-model test MSEs (for stability / std).

    Note: the effective number of trees is chosen once, by early stopping on the
    validation fold for the winning config, and that count is reused for every
    seed at the final refit (where the val fold is folded back into train, so no
    held-out set remains for early stopping).
    """
    start = time.time()
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    # ---- identical split to cross_validation() -> same 60% train / 20% val
    X_tr_cv, X_val, Y_tr_cv, Y_val = train_test_split(
        X_train, Y_train, test_size=0.25, random_state=42)

    summary_path = f"{dir_name}GBM_CV_summary.csv"
    if save:
        pd.DataFrame(columns=["config", "train_mse", "val_mse",
                              "best_rounds"]).to_csv(summary_path, index=False)

    best_val, best_hp, best_rounds = np.inf, None, None
    for hp in ParameterGrid(param_grid):
        bundle, tr_mse, val_mse = GBM_analysis(
            X_tr_cv, Y_tr_cv, X_val, Y_val, hp, y_min, y_max,
            printer_friend=False, X_val=X_val, Y_val=Y_val, seed=0,
            directions=directions)
        if save:
            pd.DataFrame([{"config": str(hp), "train_mse": tr_mse,
                           "val_mse": val_mse,
                           "best_rounds": bundle["best_rounds"]}]).to_csv(
                summary_path, mode="a", header=False, index=False)
        if val_mse < best_val:
            best_val, best_hp, best_rounds = val_mse, hp, bundle["best_rounds"]

    # ---- refit best config on the full 80% train over several seeds ----------
    #      reusing the tree count chosen on the validation fold.
    bundles, preds, seed_test_mses = [], [], []
    Y_test_arr = _as_float_1d(Y_test)
    for s in final_seeds:
        bundle = _train_one_gbm(X_train, Y_train, best_hp, seed=s,
                                directions=directions,
                                n_rounds_override=best_rounds)
        bundles.append(bundle)
        p = predict_gbm(bundle, X_test, y_min, y_max)
        preds.append(p)
        seed_test_mses.append(float(np.mean((p - Y_test_arr) ** 2)))

    ens_pred = np.mean(preds, axis=0)
    test_errors_ensemble = (ens_pred - Y_test_arr) ** 2
    MSE_test = float(np.mean(test_errors_ensemble))

    # train MSE from the first fitted model (informational only)
    tr_pred = predict_gbm(bundles[0], X_train, y_min, y_max)
    MSE_train = float(np.mean((tr_pred - _as_float_1d(Y_train)) ** 2))

    print("---------------------------------")
    print("Monotone-GBM cross-validation results")
    print(f"best hparams : {best_hp}")
    print(f"best rounds  : {best_rounds}")
    print(f"MSE_train    : {np.round(MSE_train, 3)}")
    print(f"MSE_test     : {np.round(MSE_test, 3)} "
          f"(seed mean {np.mean(seed_test_mses):.3f} +/- {np.std(seed_test_mses):.3f})")
    print(f"solve time   : {time.time() - start:.1f}s")
    print("---------------------------------")

    return MSE_train, MSE_test, best_hp, test_errors_ensemble, seed_test_mses


# --------------------------------------------------------------------------- #
#  Stats writer (parallel to compute_nn_stats, own CSV)                        #
# --------------------------------------------------------------------------- #
def compute_gbm_stats(name, test_errors_ensemble, seed_test_mses, best_hp,
                      results_path="results/monotone_gbm_stats.csv"):
    n = len(test_errors_ensemble)
    MSE_gbm = float(np.mean(test_errors_ensemble))
    SE_gbm = float(np.std(test_errors_ensemble) / np.sqrt(n))   # per-point SE (matches SE_cv/SE_iso)
    row = {
        "Name": name,
        "MSE_gbm": MSE_gbm,
        "SE_gbm": SE_gbm,
        "seed_mean_MSE": float(np.mean(seed_test_mses)),
        "seed_std_MSE": float(np.std(seed_test_mses)),
        "n_seeds": len(seed_test_mses),
        "best_hparams": str(best_hp),
    }
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    file_exists = os.path.exists(results_path)
    pd.DataFrame([row]).to_csv(results_path, mode="a",
                               header=not file_exists, index=False)
    return row


# --------------------------------------------------------------------------- #
#  Top-level driver (reproduces empirical_simulation's outer split)            #
# --------------------------------------------------------------------------- #
def run_monotone_gbm(X, Y, dir_name, name, test_size=0.2, param_grid=None,
                     final_seeds=(0, 1, 2, 3, 4), directions=None,
                     use_train_range=True):
    """
    Standalone driver whose outer split is IDENTICAL to empirical_simulation():
        train_test_split(X, Y, test_size=test_size, random_state=42)
    so the monotone GBM sees exactly the same 80/20 test split, and the same
    60/20 train/val split inside cross_validation_gbm.

    use_train_range=True clips predictions to the TRAIN Y range (recommended,
    no leakage). Set False to use the full-data range, matching the convention
    in empirical_simulation (y_min=Y.min(), y_max=Y.max()).
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42)

    if use_train_range:
        y_min, y_max = np.min(Y_train), np.max(Y_train)
    else:
        y_min, y_max = np.min(Y), np.max(Y)

    MSE_train, MSE_test, best_hp, test_errs, seed_mses = cross_validation_gbm(
        X_train, Y_train, X_test, Y_test, y_min, y_max,
        param_grid=param_grid, save=True, dir_name=dir_name,
        final_seeds=final_seeds, directions=directions)

    compute_gbm_stats(name, test_errs, seed_mses, best_hp)
    return MSE_train, MSE_test, best_hp
