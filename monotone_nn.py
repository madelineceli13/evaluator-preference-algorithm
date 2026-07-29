# monotone_nn.py
# ---------------------------------------------------------------------------
# Monotone-constrained neural-network baseline for the isotonic-preference
# pipeline. Uses Constrained Monotonic Neural Networks (Runje &
# Shankaranarayana, ICML 2023) via the `mononet` package.
#
#   pip install "mononet[torch]"
#
# Design goals:
#   * Reuse the EXACT same train/val/test partitions as `cross_validation`
#     (60/20/20, random_state=42) so the comparison is apples-to-apples.
#   * Mirror the analysis-function signatures and the per-sample-error /
#     standard-error reporting convention used in the rest of the codebase.
# ---------------------------------------------------------------------------

import os
import time
import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid

import torch
from torch import nn

from mononet import MonotonicityMask
from mononet.torch import MonoInput, MonoLinear, MonoResidual


# --------------------------------------------------------------------------- #
#  Default hyper-parameter grid (shallow, per mononet's own benchmark advice:  #
#  <= 4 effective layers is the right regime).                                 #
# --------------------------------------------------------------------------- #
DEFAULT_PARAM_GRID = {
    "hidden": [32, 64],
    "depth": [1, 2],          # number of MonoResidual blocks between the two MonoLinear layers
    "activation": ["elu"],
    "lr": [1e-2, 1e-3],
    "weight_decay": [1e-4],
    "max_epochs": [500],
    "patience": [50],
}


# --------------------------------------------------------------------------- #
#  Model construction                                                          #
# --------------------------------------------------------------------------- #
def build_monotone_net(input_dim, hidden=64, depth=1, activation="elu",
                       directions=None):
    """
    Fully-monotone (non-decreasing in every coordinate by default) network.

    directions : optional 1-D array of {-1, +1}, length input_dim.
                 None -> all +1 (non-decreasing), which matches the isotonic
                 setup (coordinate-wise partial order, nonnegative coefficients).
    """
    if directions is None:
        directions = np.ones(input_dim, dtype=np.int8)
    mask = MonotonicityMask(np.asarray(directions, dtype=np.int8))

    layers = [MonoInput(mask), MonoLinear(input_dim, hidden, activation=activation)]
    for _ in range(depth):
        layers.append(MonoResidual(hidden, hidden, activation=activation))
    layers.append(MonoLinear(hidden, 1))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
#  Small helpers                                                               #
# --------------------------------------------------------------------------- #
def _standardize_fit(A):
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)        # guard constant columns
    return mu, sd


def _to_tensor(A):
    return torch.as_tensor(np.asarray(A, dtype=np.float32))


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
#  Training + prediction                                                       #
# --------------------------------------------------------------------------- #
def _train_one(X_train, Y_train, hp, X_val=None, Y_val=None, seed=0,
               directions=None):
    """
    Train a single monotone net. Returns a 'bundle' dict holding the fitted
    model and the scalers/clip range needed for prediction.

    Optimization:
      * hp['batch_size'] controls mini-batching. If absent, None, or >= the
        training-set size, training falls back to the original full-batch
        gradient descent (one Adam step per epoch) -- so existing callers such
        as synthetic_simulations.py behave EXACTLY as before.
      * With a finite batch_size, each epoch shuffles the training set and takes
        one Adam step per mini-batch. This converges far better for wider/deeper
        nets under the same epoch budget, so the NN competes fairly instead of
        being optimization-limited into always picking the smallest config.

    If (X_val, Y_val) are given, early-stopping restores the weights at the
    epoch of lowest validation MSE. Otherwise the net trains for hp['max_epochs']
    (used at final refit, where the val fold has been folded back into train).
    """
    _set_seed(seed)

    X_train = np.asarray(X_train, dtype=np.float64)
    Y_train = np.asarray(Y_train, dtype=np.float64).ravel()
    n = X_train.shape[0]

    # --- standardize on TRAIN only (positive scaling preserves monotone order)
    x_mu, x_sd = _standardize_fit(X_train)
    y_mu, y_sd = Y_train.mean(), Y_train.std() or 1.0

    Xtr = _to_tensor((X_train - x_mu) / x_sd)
    Ytr = _to_tensor(((Y_train - y_mu) / y_sd)).reshape(-1, 1)

    model = build_monotone_net(X_train.shape[1], hidden=hp["hidden"],
                               depth=hp["depth"], activation=hp["activation"],
                               directions=directions)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"],
                           weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()

    # --- mini-batch config (full-batch fallback preserves old behavior) ------
    batch_size = hp.get("batch_size", None)
    if batch_size is None or batch_size >= n:
        batch_size = n                       # single full-batch step per epoch
    full_batch = (batch_size == n)
    # dedicated generator so the per-epoch shuffle is reproducible under `seed`
    g = torch.Generator()
    g.manual_seed(int(seed))

    use_val = X_val is not None and Y_val is not None
    if use_val:
        Xva = _to_tensor((np.asarray(X_val, dtype=np.float64) - x_mu) / x_sd)
        Yva = np.asarray(Y_val, dtype=np.float64).ravel()

    best_val = np.inf
    best_state = None
    epochs_no_improve = 0

    for _ in range(hp["max_epochs"]):
        model.train()
        if full_batch:
            opt.zero_grad()
            loss = loss_fn(model(Xtr), Ytr)
            loss.backward()
            opt.step()
        else:
            perm = torch.randperm(n, generator=g)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                opt.zero_grad()
                loss = loss_fn(model(Xtr[idx]), Ytr[idx])
                loss.backward()
                opt.step()

        if use_val:
            model.eval()
            with torch.no_grad():
                yv = model(Xva).numpy().ravel() * y_sd + y_mu
            val_mse = float(np.mean((yv - Yva) ** 2))
            if val_mse < best_val - 1e-9:
                best_val = val_mse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= hp["patience"]:
                    break

    if use_val and best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return {"model": model, "x_mu": x_mu, "x_sd": x_sd,
            "y_mu": y_mu, "y_sd": y_sd}


def predict_nn(bundle, X, y_min=None, y_max=None):
    """Predict, un-standardize, and (optionally) clip to the training range."""
    Xs = (np.asarray(X, dtype=np.float64) - bundle["x_mu"]) / bundle["x_sd"]
    with torch.no_grad():
        y = bundle["model"](_to_tensor(Xs)).numpy().ravel()
    y = y * bundle["y_sd"] + bundle["y_mu"]
    if y_min is not None and y_max is not None:
        y = np.clip(y, y_min, y_max)
    return y


def error_nn(bundle, X_vals, Y_vals, y_min, y_max):
    """Per-sample squared errors (parallel to `error_isotonic`)."""
    y_hat = predict_nn(bundle, X_vals, y_min, y_max)
    return (y_hat - np.asarray(Y_vals, dtype=np.float64).ravel()) ** 2


# --------------------------------------------------------------------------- #
#  Analysis function (mirrors `isotonic_regression_analysis`)                  #
# --------------------------------------------------------------------------- #
def monotone_nn_analysis(X_train, Y_train, X_test, Y_test, hp, y_min, y_max,
                         printer_friend=True, X_val=None, Y_val=None, seed=0,
                         directions=None):
    """
    Fit a monotone NN with hyper-parameters `hp`; return (bundle, MSE_train,
    MSE_test). If (X_val, Y_val) are provided they drive early stopping.
    """
    bundle = _train_one(X_train, Y_train, hp, X_val=X_val, Y_val=Y_val,
                        seed=seed, directions=directions)
    MSE_train = float(np.mean(error_nn(bundle, X_train, Y_train, y_min, y_max)))
    MSE_test = float(np.mean(error_nn(bundle, X_test, Y_test, y_min, y_max)))

    if printer_friend:
        print("---------------------------------")
        print("Monotone NN results")
        print(f"hparams: {hp}")
        print(f"MSE_train: {np.round(MSE_train, 3)}")
        print(f"MSE_test:  {np.round(MSE_test, 3)}")
        print("---------------------------------")

    return bundle, MSE_train, MSE_test


# --------------------------------------------------------------------------- #
#  Cross-validation / tuning wrapper (mirrors `cross_validation`)              #
# --------------------------------------------------------------------------- #
def cross_validation_nn(X_train, Y_train, X_test, Y_test, y_min, y_max,
                        param_grid=None, save=True, dir_name="Athena",
                        final_seeds=(0, 1, 2, 3, 4), directions=None):
    """
    Tune the monotone NN on the SAME validation fold used by `cross_validation`
    (train_test_split(..., test_size=0.25, random_state=42)), then refit the
    best config on the full 80% train over several seeds and evaluate on test.

    Returns: MSE_train, MSE_test, best_hp, test_errors_ensemble, seed_test_mses
      * MSE_test / test_errors_ensemble come from the seed-ensemble prediction
        (mean over seeds), so the per-point SE is comparable to SE_cv / SE_iso.
      * seed_test_mses is the list of single-model test MSEs (for stability / std).
    """
    start = time.time()
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    # ---- identical split to cross_validation() -> same 60% train / 20% val
    X_tr_cv, X_val, Y_tr_cv, Y_val = train_test_split(
        X_train, Y_train, test_size=0.25, random_state=42)

    summary_path = f"{dir_name}NN_CV_summary.csv"
    if save:
        pd.DataFrame(columns=["config", "train_mse", "val_mse"]).to_csv(
            summary_path, index=False)

    best_val, best_hp = np.inf, None
    for hp in ParameterGrid(param_grid):
        _, tr_mse, val_mse = monotone_nn_analysis(
            X_tr_cv, Y_tr_cv, X_val, Y_val, hp, y_min, y_max,
            printer_friend=False, X_val=X_val, Y_val=Y_val, seed=0,
            directions=directions)
        if save:
            pd.DataFrame([{"config": str(hp), "train_mse": tr_mse,
                           "val_mse": val_mse}]).to_csv(
                summary_path, mode="a", header=False, index=False)
        if val_mse < best_val:
            best_val, best_hp = val_mse, hp

    # ---- refit best config on the full 80% train over several seeds ----------
    bundles, preds, seed_test_mses = [], [], []
    for s in final_seeds:
        bundle = _train_one(X_train, Y_train, best_hp, seed=s,
                            directions=directions)
        bundles.append(bundle)
        p = predict_nn(bundle, X_test, y_min, y_max)
        preds.append(p)
        seed_test_mses.append(float(np.mean(
            (p - np.asarray(Y_test, dtype=np.float64).ravel()) ** 2)))

    ens_pred = np.mean(preds, axis=0)
    Y_test_arr = np.asarray(Y_test, dtype=np.float64).ravel()
    test_errors_ensemble = (ens_pred - Y_test_arr) ** 2
    MSE_test = float(np.mean(test_errors_ensemble))

    # train MSE from the first fitted model (informational only)
    tr_pred = predict_nn(bundles[0], X_train, y_min, y_max)
    MSE_train = float(np.mean(
        (tr_pred - np.asarray(Y_train, dtype=np.float64).ravel()) ** 2))

    print("---------------------------------")
    print("Monotone-NN cross-validation results")
    print(f"best hparams : {best_hp}")
    print(f"MSE_train    : {np.round(MSE_train, 3)}")
    print(f"MSE_test     : {np.round(MSE_test, 3)} "
          f"(seed mean {np.mean(seed_test_mses):.3f} +/- {np.std(seed_test_mses):.3f})")
    print(f"solve time   : {time.time() - start:.1f}s")
    print("---------------------------------")

    return MSE_train, MSE_test, best_hp, test_errors_ensemble, seed_test_mses


# --------------------------------------------------------------------------- #
#  Stats writer (parallel to compute_empirical_stats, own CSV to avoid         #
#  clobbering the fixed schema of empirical_stats_rebuttal.csv)                #
# --------------------------------------------------------------------------- #
def compute_nn_stats(name, test_errors_ensemble, seed_test_mses, best_hp,
                     results_path="results/monotone_nn_stats.csv"):
    n = len(test_errors_ensemble)
    MSE_nn = float(np.mean(test_errors_ensemble))
    SE_nn = float(np.std(test_errors_ensemble) / np.sqrt(n))   # per-point SE (matches SE_cv/SE_iso)
    row = {
        "Name": name,
        "MSE_nn": MSE_nn,
        "SE_nn": SE_nn,
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
def run_monotone_nn(X, Y, dir_name, name, test_size=0.2, param_grid=None,
                    final_seeds=(0, 1, 2, 3, 4), directions=None,
                    use_train_range=True):
    """
    Standalone driver whose outer split is IDENTICAL to empirical_simulation():
        train_test_split(X, Y, test_size=test_size, random_state=42)
    so the monotone NN sees exactly the same 80/20 test split, and the same
    60/20 train/val split inside cross_validation_nn.

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

    MSE_train, MSE_test, best_hp, test_errs, seed_mses = cross_validation_nn(
        X_train, Y_train, X_test, Y_test, y_min, y_max,
        param_grid=param_grid, save=True, dir_name=dir_name,
        final_seeds=final_seeds, directions=directions)

    compute_nn_stats(name, test_errs, seed_mses, best_hp)
    return MSE_train, MSE_test, best_hp


