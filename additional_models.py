# additional_models.py
# ---------------------------------------------------------------------------
# NN + GBM counterpart to `empirical_simulation` (functions.py).
#
# `additional_models(X, Y, dir_name, name)` mirrors empirical_simulation
# one-for-one so the monotone-NN and monotone-GBM baselines are directly
# comparable to the linear / CV / isotonic numbers you already produced:
#
#   * SAME outer split         -> train_test_split(X, Y, test_size=0.2,
#                                                   random_state=42)
#   * SAME clip range          -> y_min, y_max = Y.min(), Y.max()
#                                  (full-data range, as in empirical_simulation;
#                                   set use_train_range=True for the leakage-free
#                                   train-range convention instead)
#   * SAME per-point SE         -> std(per-sample test errors)/sqrt(len(Y_test))
#
# What it writes:
#   * {dir_name}NN_CV_summary.csv / {dir_name}GBM_CV_summary.csv
#         every hyper-parameter config with its train/val MSE (and, for GBM,
#         the early-stopping round count) -- the full tuning trail.
#   * results/additional_models_stats.csv
#         ONE row per dataset: MSE_nn, SE_nn, MSE_gbm, SE_gbm, the chosen
#         hyper-parameters, and seed-stability stats -- parallel to
#         results/empirical_stats_rebuttal.csv.
#
# Efficiency knobs (see the driver at the bottom):
#   * hpo_subsample : tune on a random subsample, refit the winner on the full
#                     train. The big lever for 300k-row datasets.
#   * nn_grid / gbm_grid, final_seeds : shrink the search / ensemble.
#   * REMINDER: do NOT run with OMP_NUM_THREADS=1 anymore -- that was only a
#     crutch for the libomp crash and now just serializes XGBoost/torch.
# ---------------------------------------------------------------------------

import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid

# NN + GBM primitives (already used by synthetic_simulations.py)
from monotone_nn import monotone_nn_analysis, _train_one, predict_nn
from monotone_gbm import GBM_analysis, _train_one_gbm, predict_gbm


# ------------------------------- default grids ----------------------------- #
# Deliberately lighter than the modules' own DEFAULT_PARAM_GRIDs, since a single
# empirical dataset is large and we tune once (not inside a T-loop).
NN_GRID_DEFAULT = {
    "hidden": [32, 64],
    "depth": [1, 2],
    "activation": ["elu"],
    "lr": [1e-2, 1e-3],
    "weight_decay": [1e-4],
    "max_epochs": [300],
    "patience": [40],
}
GBM_GRID_DEFAULT = {
    "n_estimators": [500],          # ceiling; early stopping picks the real count
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1],
    "subsample": [1.0],
    "colsample_bytree": [1.0],
    "min_child_weight": [1],
    "reg_lambda": [1.0],
    "patience": [50],
}

# Same internal fold as cross_validation / cross_validation_nn / _gbm.
_VAL_SPLIT = dict(test_size=0.25, random_state=42)


# ------------------------------ small helpers ------------------------------ #
def _subsample(X, Y, k, seed=123):
    """Return a random size-min(k, n) subsample of (X, Y); k=None -> unchanged."""
    n = X.shape[0]
    if k is None or k >= n:
        return X, Y
    idx = np.random.default_rng(seed).choice(n, size=int(k), replace=False)
    return X[idx], Y[idx]


def _clip_range(Y, Y_train, use_train_range):
    if use_train_range:
        return float(np.min(Y_train)), float(np.max(Y_train))
    return float(np.min(Y)), float(np.max(Y))    # empirical_simulation convention


# ------------------------------ NN: tune + fit ----------------------------- #
def _tune_nn(X_train, Y_train, grid, y_min, y_max, directions,
             hpo_subsample, summary_path):
    """Grid-search the NN on the shared val fold (optionally subsampled)."""
    Xs, Ys = _subsample(X_train, Y_train, hpo_subsample)
    X_tr, X_val, Y_tr, Y_val = train_test_split(Xs, Ys, **_VAL_SPLIT)

    if summary_path:
        pd.DataFrame(columns=["config", "train_mse", "val_mse"]).to_csv(
            summary_path, index=False)

    best_val, best_hp = np.inf, None
    for hp in ParameterGrid(grid):
        _, tr_mse, val_mse = monotone_nn_analysis(
            X_tr, Y_tr, X_val, Y_val, hp, y_min, y_max, printer_friend=False,
            X_val=X_val, Y_val=Y_val, seed=0, directions=directions)
        if summary_path:
            pd.DataFrame([{"config": str(hp), "train_mse": tr_mse,
                           "val_mse": val_mse}]).to_csv(
                summary_path, mode="a", header=False, index=False)
        if val_mse < best_val:
            best_val, best_hp = val_mse, hp
    return best_hp


def _ensemble_nn(X_train, Y_train, X_test, Y_test, best_hp, y_min, y_max,
                 seeds, directions):
    """Refit best config on the FULL train over seeds; ensemble-predict on test."""
    Y_test = np.asarray(Y_test, dtype=np.float64).ravel()
    preds, seed_mses = [], []
    for s in seeds:
        bundle = _train_one(X_train, Y_train, best_hp, seed=s,
                            directions=directions)
        p = predict_nn(bundle, X_test, y_min, y_max)
        preds.append(p)
        seed_mses.append(float(np.mean((p - Y_test) ** 2)))
    ens = np.mean(preds, axis=0)
    errs = (ens - Y_test) ** 2                      # per-point test errors
    return errs, seed_mses


# ------------------------------ GBM: tune + fit ---------------------------- #
def _tune_gbm(X_train, Y_train, grid, y_min, y_max, directions,
              hpo_subsample, summary_path):
    """Grid-search the GBM; early stopping picks each config's round count."""
    Xs, Ys = _subsample(X_train, Y_train, hpo_subsample)
    X_tr, X_val, Y_tr, Y_val = train_test_split(Xs, Ys, **_VAL_SPLIT)

    if summary_path:
        pd.DataFrame(columns=["config", "train_mse", "val_mse",
                              "best_rounds"]).to_csv(summary_path, index=False)

    best_val, best_hp, best_rounds = np.inf, None, None
    for hp in ParameterGrid(grid):
        bundle, tr_mse, val_mse = GBM_analysis(
            X_tr, Y_tr, X_val, Y_val, hp, y_min, y_max, printer_friend=False,
            X_val=X_val, Y_val=Y_val, seed=0, directions=directions)
        if summary_path:
            pd.DataFrame([{"config": str(hp), "train_mse": tr_mse,
                           "val_mse": val_mse,
                           "best_rounds": bundle["best_rounds"]}]).to_csv(
                summary_path, mode="a", header=False, index=False)
        if val_mse < best_val:
            best_val, best_hp, best_rounds = val_mse, hp, bundle["best_rounds"]
    return best_hp, best_rounds


def _ensemble_gbm(X_train, Y_train, X_test, Y_test, best_hp, best_rounds,
                  y_min, y_max, seeds, directions):
    """Refit best config (fixed round count) over seeds; ensemble-predict on test."""
    Y_test = np.asarray(Y_test, dtype=np.float64).ravel()
    preds, seed_mses = [], []
    for s in seeds:
        bundle = _train_one_gbm(X_train, Y_train, best_hp, seed=s,
                                directions=directions,
                                n_rounds_override=best_rounds)
        p = predict_gbm(bundle, X_test, y_min, y_max)
        preds.append(p)
        seed_mses.append(float(np.mean((p - Y_test) ** 2)))
    ens = np.mean(preds, axis=0)
    errs = (ens - Y_test) ** 2
    return errs, seed_mses


# --------------------------------------------------------------------------- #
#  Main entry point (parallels empirical_simulation)                          #
# --------------------------------------------------------------------------- #
def additional_models(X, Y, dir_name, name, test_size=0.2,
                      nn_grid=None, gbm_grid=None, final_seeds=(0, 1, 2),
                      directions=None, use_train_range=False,
                      hpo_subsample=None,
                      results_path="results/additional_models_stats.csv"):
    """
    Fit the monotone NN and monotone GBM baselines on (X, Y) exactly as
    empirical_simulation fits lin/CV/iso, and append one summary row.

    Parameters
    ----------
    X, Y            : full design matrix / targets (raw, un-split).
    dir_name, name  : output dir prefix (for the CV-summary CSVs) and the label
                      written to the stats row -- same meaning as elsewhere.
    nn_grid, gbm_grid : hyper-parameter grids (default: the lighter grids above).
    final_seeds     : seeds for the refit ensemble (predictions averaged; the
                      per-point SE is then comparable to SE_cv / SE_iso).
    directions      : per-feature monotone directions (None -> all non-decreasing).
    use_train_range : False -> clip to Y.min()/Y.max() (empirical_simulation
                      convention); True -> clip to the TRAIN range (no leakage).
    hpo_subsample   : tune on this many random train rows, refit the winner on
                      the full train. None -> tune on the full train. The main
                      speed lever for large datasets (e.g. 40000-50000).

    Returns
    -------
    dict : the summary row that was appended to `results_path`.
    """
    nn_grid = NN_GRID_DEFAULT if nn_grid is None else nn_grid
    gbm_grid = GBM_GRID_DEFAULT if gbm_grid is None else gbm_grid

    X = np.asarray(X)
    Y = np.asarray(Y, dtype=np.float64).ravel()

    # --- identical outer split to empirical_simulation -----------------------
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42)
    y_min, y_max = _clip_range(Y, Y_train, use_train_range)
    n_test = len(Y_test)

    os.makedirs(dir_name, exist_ok=True) if os.path.dirname(dir_name) or dir_name else None

    # ------------------------------- NN -------------------------------------
    t0 = time.time()
    best_hp_nn = _tune_nn(X_train, Y_train, nn_grid, y_min, y_max, directions,
                          hpo_subsample, summary_path=f"{dir_name}NN_CV_summary.csv")
    nn_errs, nn_seed_mses = _ensemble_nn(
        X_train, Y_train, X_test, Y_test, best_hp_nn, y_min, y_max,
        final_seeds, directions)
    MSE_nn = float(np.mean(nn_errs))
    SE_nn = float(np.std(nn_errs) / np.sqrt(n_test))
    t_nn = time.time() - t0
    print(f"[{name}] NN  MSE={MSE_nn:.4f} SE={SE_nn:.4f}  "
          f"best={best_hp_nn}  ({t_nn:.1f}s)")

    # ------------------------------- GBM ------------------------------------
    t0 = time.time()
    best_hp_gbm, best_rounds = _tune_gbm(
        X_train, Y_train, gbm_grid, y_min, y_max, directions,
        hpo_subsample, summary_path=f"{dir_name}GBM_CV_summary.csv")
    gbm_errs, gbm_seed_mses = _ensemble_gbm(
        X_train, Y_train, X_test, Y_test, best_hp_gbm, best_rounds,
        y_min, y_max, final_seeds, directions)
    MSE_gbm = float(np.mean(gbm_errs))
    SE_gbm = float(np.std(gbm_errs) / np.sqrt(n_test))
    t_gbm = time.time() - t0
    print(f"[{name}] GBM MSE={MSE_gbm:.4f} SE={SE_gbm:.4f}  "
          f"rounds={best_rounds}  best={best_hp_gbm}  ({t_gbm:.1f}s)")

    # --------------------------- summary row --------------------------------
    row = {
        "Name": name,
        "N": int(X.shape[0]),
        "n_test": int(n_test),
        "MSE_nn": MSE_nn, "SE_nn": SE_nn,
        "MSE_gbm": MSE_gbm, "SE_gbm": SE_gbm,
        "nn_seed_mean": float(np.mean(nn_seed_mses)),
        "nn_seed_std": float(np.std(nn_seed_mses)),
        "gbm_seed_mean": float(np.mean(gbm_seed_mses)),
        "gbm_seed_std": float(np.std(gbm_seed_mses)),
        "best_hp_nn": str(best_hp_nn),
        "best_hp_gbm": str(best_hp_gbm),
        "best_rounds_gbm": int(best_rounds),
        "use_train_range": bool(use_train_range),
        "hpo_subsample": (int(hpo_subsample) if hpo_subsample else None),
    }
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    pd.DataFrame([row]).to_csv(
        results_path, mode="a",
        header=not os.path.exists(results_path), index=False)
    return row


if __name__ == "__main__":
    # quick self-test on synthetic data (no real files needed)
    rng = np.random.default_rng(0)
    Xd = rng.integers(1, 6, size=(4000, 6)).astype(float)
    a = rng.uniform(1, 2, size=6)
    Yd = (Xd @ a) / (5 * a.sum()) + rng.normal(0, 0.2, size=4000)
    additional_models(Xd, Yd, dir_name="data/_selftest/", name="selftest",
                      hpo_subsample=1500)
