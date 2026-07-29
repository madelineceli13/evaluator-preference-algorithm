import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 30
plt.rcParams['font.family'] = 'Times New Roman'

# ── Shared constants ─────────────────────────────────────────────────────────
BAR_WIDTH    = 0.35   # single width; paired bars use ±BAR_WIDTH/2
COL_WIDTH    = 2.5    # inches per bar-group / category
FIG_HEIGHT   = 6      # inches — same for every figure
BAR_KWARGS   = dict(
    width     = BAR_WIDTH,
    edgecolor = 'grey',
    linewidth = 1.2,
    capsize   = 5,
    error_kw  = {'linewidth': 2, 'ecolor': 'grey'},
)

def plot(start, stop, title):
    df = pd.read_csv('results/empirical_stats_processed.csv').iloc[start:stop]

    ci_lin = 1.96 * df['SE_lin']
    ci_cv  = 1.96 * df['SE_cv']
    x      = np.arange(len(df))
    n      = len(df)

    # Each subplot gets n columns; two subplots side by side
    fig, (ax2, ax1) = plt.subplots(
        1, 2,
        figsize=(COL_WIDTH * n * 2, FIG_HEIGHT),
        sharey=False,
    )

    # ── Left subplot: Prediction error ──────────────────────────────────────
    ax1.bar(x - BAR_WIDTH/2, df['MSE_lin'], color='lightblue', hatch='.',
            yerr=ci_lin, label='Linear regression', **BAR_KWARGS)
    ax1.bar(x + BAR_WIDTH/2, df['MSE_cv'],  color='thistle',   hatch='|',
            yerr=ci_cv,  label='Our algorithm',     **BAR_KWARGS)

    # ── Right subplot: Reducible error ──────────────────────────────────────

    ax2.bar(x - BAR_WIDTH/2, df['est_error_lin'], color='lightblue', hatch='.',
            yerr=ci_lin, label='Linear regression', **BAR_KWARGS)
    ax2.bar(x + BAR_WIDTH/2, df['est_error_cv'],  color='thistle',   hatch='|',
            yerr=ci_cv,  label='Our algorithm',     **BAR_KWARGS)

    for ax, ylabel in [(ax1, 'Prediction error'), (ax2, 'Reducible error')]:
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Name'], rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xlim(-0.5, n - 0.5)   # consistent margins

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left',
               bbox_to_anchor=(1.01, 0.5), frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(f'plots/combined_results_{title}.png', dpi=300, bbox_inches='tight')

plot(4, 8, 'trip_advisor')


def plot_noise_and_alignment():
    df_noise = pd.read_csv('results/irreducible_error.csv')
    df_align = pd.read_csv('results/residuals_human.csv')

    ci_noise = 1.96 * df_noise['se_noise']
    ci_align = 1.96 * df_align['se']

    x_noise = np.arange(len(df_noise))
    x_align = np.arange(len(df_align))

    n_noise = len(df_noise)
    n_align = len(df_align)

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(COL_WIDTH * (n_noise + n_align), FIG_HEIGHT),
        gridspec_kw={'width_ratios': [n_noise, n_align]},
        sharey=False,
    )

    x_noise = np.arange(len(df_noise))

    yerr_noise = np.array([
        df_noise['mean_noise'] - df_noise['lower_ci'],  # lower error
        df_noise['upper_ci'] - df_noise['mean_noise'],  # upper error
    ])
    # ── Left subplot: Noise ──────────────────────────────────────────────
    ax1.bar(x_noise, df_noise['mean_noise'], color='thistle', hatch='|',
            yerr=yerr_noise, **bar_kwargs)
    ax1.set_ylabel('Irreducible error')
    ax1.set_xticks(x_noise)
    ax1.set_xticklabels(df_noise['name'], rotation=15, ha='center')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_xlim(-0.5, len(df_noise) - 0.5)

    # ── Right subplot: Alignment ─────────────────────────────────────────────
    ax2.bar(x_align, df_align['mean'], color='lightblue', hatch='.',
            yerr=ci_align, **BAR_KWARGS)
    ax2.set_ylabel('Preference misalignment')
    ax2.set_xticks(x_align)
    ax2.set_xticklabels(df_align['comparison'], rotation=15, ha='center')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_xlim(-0.5, n_align - 0.5)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.savefig('plots/noise_and_alignment.png', dpi=300, bbox_inches='tight')
# plot_noise_and_alignment()