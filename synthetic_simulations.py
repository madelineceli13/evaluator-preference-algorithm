# synthetic_simulations.py
# ---------------------------------------------------------------------------
# Synthetic estimation-error study for the isotonic-preference pipeline,
# extended to the two new monotone baselines (neural net + GBM).
#
# Mirrors the existing `simulate_CV` design: for each function family
# (linear / Cobb-Douglas / Leontief), sample size N, and trial t, draw random
# preference parameters and data, fit every estimator, and score each on the
# TRUE estimation error -- the mean squared gap between the estimator's
# predictions on the full grid X = [m]^d and the true function values there
# (we know the ground truth, so this is exact, not a test-set proxy).
#
# Estimators:
#     lin  -> non-negative least squares            (Linear regression)
#     iso  -> isotonic regression, lambda = 0        (optional)
#     cv   -> regularized isotonic + CV              (Our algorithm)
#     nn   -> monotone-constrained neural net        (monotone_nn.py)
#     gbm  -> monotone-constrained XGBoost           (monotone_gbm.py)
#
# ALL estimators tune on the identical internal fold used by `cross_validation`
# (train_test_split(X_train, Y_train, test_size=0.25, random_state=42)), so the
# 60/20/20 train/val/test partition is shared across methods.
#
# Nothing is written to disk except: optimal parameter choices, MSE (the true
# estimation error), and its across-trial standard error -- a per-trial raw CSV
# (MSE + chosen hyper-parameters) and an aggregated summary CSV (mean MSE + SE).
# The NN/GBM grids default to lighter settings than the estimators' own module
# defaults, since the grid search runs inside every trial (see the two GRID
# dicts below); pass your own grids to widen them.
# ---------------------------------------------------------------------------

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid

from functions import (
    linear_regression_analysis, isotonic_regression_analysis,
    cross_validation, estimation_error, unique_xvals, iso_fit_many,
)

# ------------------------------ configuration ------------------------------ #
NUM_SAMPLES = np.linspace(50, 1000, num=20, dtype="int16")   # x-axis of the figure
OUTPUT_DIR = "data/synthetic/"

# Internal-fold split shared by every estimator (identical to cross_validation).
_VAL_SPLIT = dict(test_size=0.25, random_state=42)

# Lighter-than-default grids because tuning happens once per (N, trial).
NN_GRID_SYNTH = {
    "hidden": [32],
    "depth": [1, 2],
    "activation": ["elu"],
    "lr": [1e-2, 1e-3],
    "weight_decay": [1e-4],
    "max_epochs": [300],
    "patience": [40],
}
GBM_GRID_SYNTH = {
    "n_estimators": [400],       # ceiling; early stopping picks the real count
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.1],
    "subsample": [1.0],
    "colsample_bytree": [1.0],
    "min_child_weight": [1],
    "reg_lambda": [1.0],
    "patience": [40],
}

# Plot metadata.
_LABELS = {"lin": "Linear regression", "iso": "Isotonic (\u03bb=0)",
           "cv": "Our algorithm", "nn": "Monotone NN", "gbm": "Monotone GBM"}
_TITLES = {"linear": "Linear", "cobb_douglas": "Cobb-Douglas", "leontief": "Leontief"}
_STYLE = {  # (color, marker, linestyle)
    "lin": ("#1f77b4", "o", "-"),
    "cv":  ("#ff7f0e", "s", ":"),
    "nn":  ("#2ca02c", "^", "--"),
    "gbm": ("#d62728", "D", "-."),
    "iso": ("#9467bd", "v", "--"),
}


# --------------------------- fd-level output mute -------------------------- #
class _Quiet:
    """
    Silence C-level stdout/stderr (the SCS solver in cvxpy prints there, so a
    plain contextlib.redirect_stdout would not catch it). Used only around the
    isotonic / CV solves so trial progress stays readable.
    """
    def __init__(self, enabled=True):
        self.enabled = enabled

    def __enter__(self):
        if not self.enabled:
            return self
        sys.stdout.flush(); sys.stderr.flush()
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        self._old = (os.dup(1), os.dup(2))
        os.dup2(self._devnull, 1); os.dup2(self._devnull, 2)
        return self

    def __exit__(self, *exc):
        if not self.enabled:
            return False
        os.dup2(self._old[0], 1); os.dup2(self._old[1], 2)
        os.close(self._devnull); os.close(self._old[0]); os.close(self._old[1])
        return False


# ------------------------------ data generation ---------------------------- #
def _preference_values(function_name, X, X_grid, a, num_vals):
    """Return (f(X), f(X_grid)) for the requested utility family, normalized to [0,1]."""
    if function_name == "linear":
        norm = num_vals * np.sum(a)
        return (X @ a) / norm, (X_grid @ a) / norm
    if function_name == "leontief":
        norm = np.max(a) * num_vals
        return np.min(X * a, axis=1) / norm, np.min(X_grid * a, axis=1) / norm
    if function_name == "cobb_douglas":
        norm = num_vals ** np.sum(a)
        return (np.prod(np.power(X, a), axis=1) / norm,
                np.prod(np.power(X_grid, a), axis=1) / norm)
    raise ValueError(f"unknown function_name: {function_name}")


# --------------------- per-estimator grid tuning + grid fit ---------------- #
def _tune_and_fit_nn(X_train, Y_train, X_grid, y_min, y_max, grid, seed, directions):
    """Tune the monotone NN on the shared val fold, refit on the 80% train, predict on the grid."""
    from monotone_nn import monotone_nn_analysis, _train_one, predict_nn
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, **_VAL_SPLIT)
    best_val, best_hp = np.inf, None
    for hp in ParameterGrid(grid):
        _, _, val_mse = monotone_nn_analysis(
            X_tr, Y_tr, X_val, Y_val, hp, y_min, y_max, printer_friend=False,
            X_val=X_val, Y_val=Y_val, seed=seed, directions=directions)
        if val_mse < best_val:
            best_val, best_hp = val_mse, hp
    bundle = _train_one(X_train, Y_train, best_hp, seed=seed, directions=directions)
    return best_hp, predict_nn(bundle, X_grid, y_min, y_max)


def _tune_and_fit_gbm(X_train, Y_train, X_grid, y_min, y_max, grid, seed, directions):
    """Tune the monotone GBM on the shared val fold, refit on the 80% train, predict on the grid."""
    from monotone_gbm import GBM_analysis, _train_one_gbm, predict_gbm
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, **_VAL_SPLIT)
    best_val, best_hp, best_rounds = np.inf, None, None
    for hp in ParameterGrid(grid):
        bundle, _, val_mse = GBM_analysis(
            X_tr, Y_tr, X_val, Y_val, hp, y_min, y_max, printer_friend=False,
            X_val=X_val, Y_val=Y_val, seed=seed, directions=directions)
        if val_mse < best_val:
            best_val, best_hp, best_rounds = val_mse, hp, bundle["best_rounds"]
    bundle = _train_one_gbm(X_train, Y_train, best_hp, seed=seed,
                            directions=directions, n_rounds_override=best_rounds)
    return best_hp, predict_gbm(bundle, X_grid, y_min, y_max)


# ------------------------------- main simulation --------------------------- #
def simulate(function_name, d=2, num_vals=5, T=15, sd=0.2, test_size=0.2,
             estimators=("lin", "cv", "nn", "gbm"),
             num_samples=NUM_SAMPLES, nn_grid=None, gbm_grid=None,
             directions=None, random_state=0, quiet=True, verbose=True):
    """
    Run the estimation-error study for one function family.

    Returns a per-trial DataFrame with columns:
        function, N, t, mse_<est> for each estimator, and the optimal parameter
        choices (lambda for cv, hp_nn for nn, hp_gbm for gbm).

    Only MSE (true estimation error) and the optimal parameters are recorded;
    standard errors are formed later by `aggregate`.
    """
    y_min, y_max = 0.0, 1.0
    nn_grid = NN_GRID_SYNTH if nn_grid is None else nn_grid
    gbm_grid = GBM_GRID_SYNTH if gbm_grid is None else gbm_grid
    np.random.seed(random_state)             # reproducible data + splits for the whole run
    X_grid = unique_xvals(num_vals, d)

    rows = []
    t0 = time.time()
    for N in num_samples:
        for t in range(T):
            a = np.random.uniform(1, 2, size=d)
            X = np.random.randint(1, num_vals + 1, size=(N, d))
            noise = np.random.normal(0, sd, size=N)
            fX, y_true = _preference_values(function_name, X, X_grid, a, num_vals)
            Y = fX + noise

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
            row = {"function": function_name, "N": int(N), "t": int(t)}

            if "lin" in estimators:
                coef, _, _ = linear_regression_analysis(
                    X_train, Y_train, X_test, Y_test, y_min=y_min, y_max=y_max,
                    printer_friend=False)
                y_guess = X_grid @ coef[0] + coef[1]      # unclipped, matching simulate_CV
                row["mse_lin"] = estimation_error(y_guess, y_true)

            if "iso" in estimators:
                with _Quiet(quiet):
                    df_iso, _, _ = isotonic_regression_analysis(
                        X_train, Y_train, X_test, Y_test, lam=0,
                        y_min=y_min, y_max=y_max, printer_friend=False)
                x_vals, f_vals = np.asarray(df_iso.iloc[:, :-1]), np.asarray(df_iso.iloc[:, -1])
                row["mse_iso"] = estimation_error(iso_fit_many(X_grid, x_vals, f_vals), y_true)

            if "cv" in estimators:
                with _Quiet(quiet):
                    _, _, df_CV, lam = cross_validation(
                        X_train, Y_train, X_test, Y_test, y_min=0, y_max=1, save=False)
                if lam != np.inf:
                    x_vals, f_vals = np.asarray(df_CV.iloc[:, :-1]), np.asarray(df_CV.iloc[:, -1])
                    y_guess = iso_fit_many(X_grid, x_vals, f_vals)
                else:
                    y_guess = X_grid @ df_CV[0] + df_CV[1]
                row["mse_cv"] = estimation_error(y_guess, y_true)
                row["lambda"] = lam

            if "nn" in estimators:
                best_hp, y_guess = _tune_and_fit_nn(
                    X_train, Y_train, X_grid, y_min, y_max, nn_grid, seed=t,
                    directions=directions)
                row["mse_nn"] = estimation_error(y_guess, y_true)
                row["hp_nn"] = str(best_hp)

            if "gbm" in estimators:
                best_hp, y_guess = _tune_and_fit_gbm(
                    X_train, Y_train, X_grid, y_min, y_max, gbm_grid, seed=t,
                    directions=directions)
                row["mse_gbm"] = estimation_error(y_guess, y_true)
                row["hp_gbm"] = str(best_hp)

            rows.append(row)

        if verbose:
            done = (np.where(num_samples == N)[0][0] + 1)
            print(f"[{function_name}] N={int(N):>4}  ({done}/{len(num_samples)})  "
                  f"elapsed {time.time() - t0:6.1f}s", flush=True)

    return pd.DataFrame(rows)


# --------------------------- aggregation (MSE + SE) ------------------------ #
def aggregate(df, estimators=("lin", "cv", "nn", "gbm")):
    """
    Collapse per-trial MSEs to mean estimation error and across-trial standard
    error (std / sqrt(T)) per (function, N, estimator) -- the quantities the
    figure's points and error bars use.
    """
    value_cols = [f"mse_{e}" for e in estimators if f"mse_{e}" in df.columns]
    long = df.melt(id_vars=["function", "N"], value_vars=value_cols,
                   var_name="estimator", value_name="mse")
    long["estimator"] = long["estimator"].str.replace("mse_", "", regex=False)
    g = long.groupby(["function", "N", "estimator"])["mse"]
    summary = g.agg(mse_mean="mean", mse_std="std", n_trials="count").reset_index()
    summary["mse_se"] = summary["mse_std"] / np.sqrt(summary["n_trials"])
    return summary[["function", "N", "estimator", "mse_mean", "mse_se", "n_trials"]]


# ------------------------------- plotting ---------------------------------- #
def plot_results(summary, functions=("linear", "cobb_douglas", "leontief"),
                 estimators=("lin", "cv", "nn", "gbm"), sharey=True,
                 save_path=None, show=False):
    """Reproduce the multi-panel estimation-error figure (one panel per function)."""
    import matplotlib.pyplot as plt

    n = len(functions)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.2), sharey=sharey)
    if n == 1:
        axes = [axes]

    handles, seen = [], set()
    for ax, fn in zip(axes, functions):
        sub = summary[summary["function"] == fn]
        for est in estimators:
            s = sub[sub["estimator"] == est].sort_values("N")
            if s.empty:
                continue
            color, marker, ls = _STYLE.get(est, ("gray", "o", "-"))
            line = ax.errorbar(s["N"], s["mse_mean"], yerr=s["mse_se"],
                               color=color, marker=marker, linestyle=ls,
                               markersize=4, capsize=2, elinewidth=1,
                               label=_LABELS.get(est, est))
            if est not in seen:
                handles.append(line); seen.add(est)
        ax.set_title(_TITLES.get(fn, fn))
        ax.set_xlabel("Number of evaluations")
        ax.margins(x=0.02)
    axes[0].set_ylabel("Estimation error")
    if sharey:
        axes[0].set_ylim(bottom=0)

    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.04), frameon=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ------------------------------- driver ------------------------------------ #
def run_all(functions=("linear", "cobb_douglas", "leontief"), d=2, num_vals=5,
            T=15, sd=0.2, estimators=("lin", "cv", "nn", "gbm"),
            num_samples=NUM_SAMPLES, output_dir=OUTPUT_DIR,
            fig_path="plots/nn_gmb.png", **kw):
    """
    Run every family, save the per-trial raw CSV (MSE + optimal params) and the
    aggregated summary CSV (MSE mean + SE), and render the figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    parts = [simulate(fn, d=d, num_vals=num_vals, T=T, sd=sd,
                      estimators=estimators, num_samples=num_samples, **kw)
             for fn in functions]
    raw = pd.concat(parts, ignore_index=True)
    summary = aggregate(raw, estimators=estimators)

    raw_path = os.path.join(output_dir, "synthetic_results_raw.csv")
    sum_path = os.path.join(output_dir, "synthetic_results_summary.csv")
    raw.to_csv(raw_path, index=False)         # per-trial MSE + optimal parameters
    summary.to_csv(sum_path, index=False)     # per (function, N, estimator): MSE mean + SE

    plot_results(summary, functions=functions, estimators=estimators, save_path=fig_path)
    print(f"\nsaved: {raw_path}\n       {sum_path}\n       {fig_path}")
    return raw, summary


if __name__ == "__main__":
    # d=2, m=5 matches the reference figure; T reduced because NN/GBM tuning
    # runs inside every trial. Bump T (and widen the grids) for final runs.
    run_all(T=15)
