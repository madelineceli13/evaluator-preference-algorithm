import matplotlib.pyplot as plt
import numpy as np
from functions import unique_xvals, iso_fit_many
import pandas as pd
plt.rcParams['font.size'] = 24
plt.rcParams['font.family'] = 'Times New Roman'

fontsize = 20 # for smaller components


# for error bar computation
def error(MSE, SE, noise):
    err = 1.96*SE/(2*np.sqrt(MSE - noise))
    return err
df_stats = pd.read_csv('results/empirical_stats.csv')

# ----------------------------- TripAdvisor Understanding ------------------------ #
dir_name = 'data/hotelrec/outputs_all_2019/'
dir_names = ['data/hotelrec/outputs_all_2019/', 'data/hotelrec/outputs_business/', 
             'data/hotelrec/outputs_couple/', 'data/hotelrec/outputs_family/']
names = ['All - 2019', 'Business travel', 'Couples travel', 'Family travel']

styles = [
    {'color': 'red', 'marker': 'o', 'linestyle': '-'},
    {'color': 'orange', 'marker': 's', 'linestyle': '--'},
    {'color': 'green', 'marker': 'D', 'linestyle': ':'},
    {'color': 'blue', 'marker': '^', 'linestyle': '-.'},
]


fig, ax = plt.subplots(figsize=(8, 5))
i = 1
for iter in range(4):
    df = pd.read_csv(f'{dir_names[iter]}cv_df.csv')
    x_vals, f_vals = np.asarray(df.iloc[:, 1:-1]), np.asarray(df.iloc[:, -1])

    mask = np.ones(len(x_vals), dtype=bool)
    for j in range(6):
        if j != i:
            mask &= x_vals[:, j] == 3

    varying_vals = x_vals[mask, i].tolist()
    f_vals_CV = f_vals[mask].tolist()

    s = styles[iter]
    ax.plot(varying_vals, f_vals_CV,
            color=s['color'],
            marker=s['marker'],
            linestyle=s['linestyle'],
            linewidth=3,
            markersize=10,
            label=names[iter])

ax.set_ylabel('Estimated overall rating')
ax.set_xlabel('Service rating')
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('plots/understanding_varying_service.png', dpi=300, bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(figsize=(8, 5))

MSE = 0.21676203587311083
SE = 0.0033088485582759898
irreducible_error = 0.20407158323858074

y_err = error(df_stats.loc[df_stats['Name'] == 'All - 2019', 'MSE_cv'].values[0],
                         df_stats.loc[df_stats['Name'] == 'All - 2019', 'SE_cv'].values[0],
                         df_stats.loc[df_stats['Name'] == 'All - 2019', 'noise'].values[0])

criteria_names = ['Service', 'Location', 'Value']
for i in [0,1,2]:
    df = pd.read_csv(f'{dir_name}cv_df.csv')
    x_vals, f_vals = np.asarray(df.iloc[:, 1:-1]), np.asarray(df.iloc[:, -1])
    
    mask = np.ones(len(x_vals), dtype=bool)
    for j in range(6):
        if j != i:
            mask &= x_vals[:, j] == 3

    varying_vals = x_vals[mask, i].tolist()
    f_vals_CV = f_vals[mask].tolist()

    s = styles[i]
    ax.errorbar(varying_vals, f_vals_CV,
            yerr = y_err,
            color=s['color'],
            marker=s['marker'],
            linestyle=s['linestyle'],
            linewidth=3,
            markersize=10,
            label=criteria_names[i])

ax.set_ylabel('Estimated overall rating')
ax.set_xlabel('Criterion rating')
ax.legend(fontsize=fontsize)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('plots/understanding_varying_criteria.png', dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------- LLMs Understanding ------------------------ #

def error(MSE, SE, noise):
    err = 1.96*SE/(2*np.sqrt(MSE - noise))
    return err

dir_names = ['data/LLMs/outputs_gpt/', 'data/LLMs/outputs_llama/', 'data/LLMs/outputs_human/']
names = ['GPT-4o', 'Llama-3.1-70b', 'Human']

error_gpt = error(df_stats.loc[df_stats['Name'] == 'GPT-4o', 'MSE_cv'].values[0],
                     df_stats.loc[df_stats['Name'] == 'GPT-4o', 'SE_cv'].values[0],
                     df_stats.loc[df_stats['Name'] == 'GPT-4o', 'noise'].values[0])
error_llama = error(df_stats.loc[df_stats['Name'] == 'Llama 3.1 70b', 'MSE_cv'].values[0],
                     df_stats.loc[df_stats['Name'] == 'Llama 3.1 70b', 'SE_cv'].values[0],
                     df_stats.loc[df_stats['Name'] == 'Llama 3.1 70b', 'noise'].values[0])
error_human = error(df_stats.loc[df_stats['Name'] == 'Human evaluators', 'MSE_cv'].values[0],
                     df_stats.loc[df_stats['Name'] == 'Human evaluators', 'SE_cv'].values[0],
                     df_stats.loc[df_stats['Name'] == 'Human evaluators', 'noise'].values[0])

criteria_names = ['Soundness', 'Presentation', 'Contribution']
y_err = [error_gpt, error_llama, error_human]

fixed_vals = [2, 3]
for i in range(3):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, fixed_val in zip(axes, fixed_vals):
        for iter in range(3):
            df = pd.read_csv(f'{dir_names[iter]}cv_df.csv')
            x_vals, f_vals = np.asarray(df.iloc[:, 1:-1]), np.asarray(df.iloc[:, -1])
            mask = np.ones(len(x_vals), dtype=bool)
            for j in range(3):
                if j != i:
                    mask &= x_vals[:, j] == fixed_val
            varying_vals = x_vals[mask, i].tolist()
            f_vals_CV = f_vals[mask].tolist()
            s = styles[iter]
            ax.errorbar(varying_vals, f_vals_CV,yerr=y_err[iter],
                    color=s['color'],
                    marker=s['marker'],
                    linestyle=s['linestyle'],
                    linewidth=3,
                    markersize=10,
                    label=names[iter])
        ax.set_ylabel('Estimated overall rating')
        ax.set_xlabel(f'{criteria_names[i]} rating')
        ax.set_title(f'Other criteria fixed at {fixed_val}/4')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    plt.savefig(f'plots/LLM_comparison/understanding_{criteria_names[i]}.png', dpi=300, bbox_inches='tight')
    plt.close()

