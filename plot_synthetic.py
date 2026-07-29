import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.size'] = 24
fontsize = 24
plt.rcParams['font.family'] = 'Times New Roman'
extensions = 'd_2_n_5'
def get_data(function_name):
    df = pd.read_csv(f'data/synthetic/{function_name}_{extensions}_sd_02.csv')  
    # linear regression
    df_lin = df.groupby('N').agg(
        mse_mean=('mse_lin', 'mean'),
        mse_se=('mse_lin', 'sem')
    ).reset_index()

    # cross-validation
    df_CV = df.groupby('N').agg(
        mse_mean=('mse_CV', 'mean'),
        mse_se=('mse_CV', 'sem')
    ).reset_index()


    df_list = [df_lin, df_CV]
    z_alpha = 1.96

    for df in df_list:
        df['mse_se'] = z_alpha * df[df.columns[2]]  

    xs = df_lin['N']  # N is num_samples
    return df_lin, df_CV, xs


fig, (ax_lin, ax_cd, ax_leon) = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

# Linear plot
(df_lin, df_CV, xs) = get_data('linear')
ax_lin.errorbar(xs, df_lin['mse_mean'], yerr=df_lin['mse_se'],
                label='Linear regression',  linestyle='-', fmt='o', capsize=4)
ax_lin.errorbar(xs, df_CV['mse_mean'], yerr=df_CV['mse_se'],
                label='Our algorithm', linestyle=':', fmt='s', capsize=4)
ax_lin.set_ylabel('Estimation error')
ax_lin.set_title('Linear')
# ax_lin.legend()
ax_lin.grid(True)

# Cobb-douglas plot
(df_lin, df_CV, xs) = get_data('cobb_douglas')
ax_cd.errorbar(xs, df_lin['mse_mean'], yerr=df_lin['mse_se'],
               label='Linear regression',  linestyle='-', fmt='o', capsize=4)
ax_cd.errorbar(xs, df_CV['mse_mean'], yerr=df_CV['mse_se'],
               label='Our algorithm', linestyle=':', fmt='s', capsize=4)
ax_cd.set_title('Cobb-Douglas')
ax_cd.set_xlabel('Number of evaluations')
ax_cd.grid(True)

# Leontief plot
(df_lin, df_CV, xs) = get_data('leontief')
ax_leon.errorbar(xs, df_lin['mse_mean'], yerr=df_lin['mse_se'],
                 label='Linear regression',  linestyle='-', fmt='o', capsize=4)
ax_leon.errorbar(xs, df_CV['mse_mean'], yerr=df_CV['mse_se'],
                 label='Our algorithm', linestyle=':', fmt='s', capsize=4)
ax_leon.set_title('Leontief')
ax_leon.grid(True)
print(df_lin['mse_mean'].iloc[0]/df_CV['mse_mean'][0])
print(df_lin['mse_mean'].iloc[-1]/df_CV['mse_mean'].iloc[-1])
ax_lin.tick_params(left=False, bottom=False)
ax_cd.tick_params(left=False, bottom=False)
ax_leon.tick_params(left=False, bottom=False)
yticks = ax_lin.get_yticks()

# Add single shared legend
handles, labels = ax_lin.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
           ncol=3, frameon=True, fontsize=fontsize)

# fig.legend(handles, labels, loc='upper right',frameon=True)
plt.tight_layout()
plt.subplots_adjust(bottom=0.3, top=0.92, left = 0.1)  # Make room for shared xlabel and legend
plt.savefig(f'plots/linear_CV_comparison.png', bbox_inches='tight', dpi=300)
plt.close()

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (display name, file stem) ----------------------------------------------------
FILES = [
    ("Linear",       "linear_d_2_n_5_sd_02.csv"),
    ("Cobb-Douglas", "cobb_douglas_d_2_n_5_sd_02.csv"),
    ("Leontief",     "leontief_d_2_n_5_sd_02.csv"),
]

# Ordered lambda grid: 0, 2^-9 ... 2^8, inf  ->  20 ordered categories ----------
EXPS = list(range(-9, 9))                       # -9 .. 8
CATS = ["0"] + [str(e) for e in EXPS] + ["inf"]
POS  = np.arange(len(CATS))
INTERIOR_COLOR, ENDPOINT_COLOR = "#4C72B0", "#DD8452"


def lambda_to_category(v):
    """Map a lambda value to its index on the ordered categorical axis."""
    v = float(v)
    if v == 0:
        return 0
    if np.isinf(v):
        return len(CATS) - 1
    return EXPS.index(int(round(np.log2(v)))) + 1


def make_figure(data_dir="."):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)

    for ax, (name, fname) in zip(axes, FILES):
        df = pd.read_csv(os.path.join(data_dir, fname))
        idx = df["lambda"].map(lambda_to_category).values
        counts = np.bincount(idx, minlength=len(CATS)).astype(float)
        prop = 100 * counts / counts.sum()

        colors = [ENDPOINT_COLOR] + [INTERIOR_COLOR] * len(EXPS) + [ENDPOINT_COLOR]
        ax.bar(POS, prop, color=colors, width=0.85, edgecolor="white", linewidth=0.4)

        interior_pct = prop[1:-1].sum()
        ax.set_title(name, fontsize=12, pad=8)
        ax.text(0.5, 0.93, f"{interior_pct:.0f}% interior",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=10, color=INTERIOR_COLOR, fontweight="bold")

        # sparse, readable x labels
        show = {0, len(CATS) - 1} | {
            i + 1 for i, e in enumerate(EXPS) if e in (-9, -6, -3, 0, 3, 6, 8)
        }
        labels = []
        for i, c in enumerate(CATS):
            if i not in show:
                labels.append("")
            elif c == "0":
                labels.append("0")
            elif c == "inf":
                labels.append(r"$\infty$")
            else:
                labels.append(r"$2^{%s}$" % c)
        ax.set_xticks(POS)
        ax.set_xticklabels(labels, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3)
        ax.margins(x=0.01)

    axes[0].set_ylabel("% of fits (of 1000)", fontsize=11)
    fig.supxlabel(
        r"chosen $\lambda^\star$   ($0$ = pure isotonic  $\cdot$  "
        r"interior = powers of 2  $\cdot$  $\infty$ = linear regression)",
        fontsize=10.5, y=0.02,
    )
    fig.suptitle(
        r"Distribution of cross-validated regularization weight $\lambda^\star$ "
        r"(synthetic simulations, $d{=}2$, $m{=}5$, $\sigma{=}0.2$)",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig("plots/lambda_distribution.png", dpi=200, bbox_inches="tight")
    return fig



make_figure(data_dir="data/synthetic/")

# plot_noise_and_alignment()