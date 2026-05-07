import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 24
plt.rcParams['font.family'] = 'Times New Roman'

def plot(start, stop, title):
    df = pd.read_csv('results/empirical_stats.csv').iloc[start:stop]

    ci_lin = 1.96 * df['SE_lin']
    ci_cv  = 1.96 * df['SE_cv']

    x     = np.arange(len(df))
    width = 0.35
    w     = len(df)

    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(5 / 2 * 3 * 2 + 2, 6),
                                    sharey=False)

    bar_kwargs = dict(width=width, edgecolor='grey', linewidth=1.2,
                      capsize=5, error_kw={'linewidth': 2, 'ecolor': 'grey'})

    # ── Left subplot: Prediction error ──────────────────────────────────────
    bars1 = ax1.bar(x - width/2, df['MSE_lin'], color='lightblue', hatch='.',
                    yerr=ci_lin,
                    label='Linear regression', **bar_kwargs)
    bars2 = ax1.bar(x + width/2, df['MSE_cv'],  color='thistle',   hatch='|',
                    yerr=ci_cv,
                    label='Our algorithm', **bar_kwargs)


    # ── Right subplot: Oracle-based estimation error ─────────────────────────
    vals_lin = df['MSE_lin'] - df['EV_test']
    vals_cv  = df['MSE_cv']  - df['EV_test']

    ax2.bar(x - width/2, vals_lin, color='lightblue', hatch='.',
            yerr=ci_lin,
            label='Linear regression', **bar_kwargs)
    ax2.bar(x + width/2, vals_cv,  color='thistle',   hatch='|',
            yerr=ci_cv,
            label='Our algorithm', **bar_kwargs)


    # ── Shared legend to the right ───────────────────────────────────────────
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left',
               bbox_to_anchor=(1.01, 0.5), frameon=True)


    ax1.set_ylabel('Prediction error')
    ax1.set_title('')          # remove title
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Name'], rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    ax2.set_ylabel('Reducible error')
    ax2.set_title('')          # remove title
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Name'], rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.subplots_adjust(left = 0.1, wspace=0.2)   # fixes the overlap
    plt.savefig(f'plots/combined_results_{title}.png', dpi=300, bbox_inches='tight')
    # plt.show()

plot(4,8, 'trip_advisor')


def plot_noise():
    df = pd.read_csv('noise_summary.csv')
    ci = 1.96 * df['se_noise']
    x  = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(2 * len(df) + 1, 6))

    bar_kwargs = dict(
        width     = 0.5,
        edgecolor = 'grey',
        linewidth = 1.2,
        capsize   = 5,
        error_kw  = {'linewidth': 2, 'ecolor': 'grey'},
    )

    ax.bar(x, df['mean_noise'], color='thistle', hatch='|', yerr=ci, **bar_kwargs)

    ax.set_ylabel('Irreducible error')
    ax.set_xticks(x)
    ax.set_xticklabels(df['name'], rotation=15, ha='center')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('plots/noise_results.png', dpi=300, bbox_inches='tight')
    # plt.show()

# plot_noise()

def plot_alignment():
    df = pd.read_csv('results/residuals_human.csv')
    ci = 1.96 * df['se']
    x  = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(2 * len(df) + 1, 6))

    bar_kwargs = dict(
        width     = 0.5,
        edgecolor = 'grey',
        linewidth = 1.2,
        capsize   = 5,
        error_kw  = {'linewidth': 2, 'ecolor': 'grey'},
    )

    ax.bar(x, df['mean'],color='lightblue', hatch='.', yerr=ci, **bar_kwargs)

    ax.set_ylabel('Preference misalignment')
    ax.set_xticks(x)
    ax.set_xticklabels(df['comparison'], rotation=15, ha='center')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('plots/alignment.png', dpi=300, bbox_inches='tight')
    plt.show()

#plot_alignment()

def plot_noise_and_alignment():
    df_align = pd.read_csv('results/residuals_human.csv')
    df_noise = pd.read_csv('results/irreducible_error.csv')

    ci_align = 1.96 * df_align['se']

    x_align = np.arange(len(df_align))

    w = max(len(df_noise), len(df_align))
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5 / 2 * w * 2 + 2, 6), sharey=False)
    fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(5 / 2 * w * 2 + 2, 6),
    gridspec_kw={'width_ratios': [len(df_noise), len(df_align)]},
    sharey=False,
)

    bar_kwargs = dict(
        width     = 0.5,
        edgecolor = 'grey',
        linewidth = 1.2,
        capsize   = 5,
        error_kw  = {'linewidth': 2, 'ecolor': 'grey'},
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

    # ── Right subplot: Alignment ─────────────────────────────────────────
    ax2.bar(x_align, df_align['mean'], color='lightblue', hatch='.',
            yerr=ci_align, **bar_kwargs)
    ax2.set_ylabel('Preference misalignment')
    ax2.set_xticks(x_align)
    ax2.set_xticklabels(df_align['comparison'], rotation=15, ha='center')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, wspace=0.3)
    plt.savefig('plots/noise_and_alignment.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_noise_and_alignment()