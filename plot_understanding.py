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

# ax.errorbar(xs, df_lin['mse_mean'], yerr=df_lin['mse_se'],
#                 label='Linear regression',  linestyle='-', fmt='o', capsize=4)
# ax.errorbar(xs, df_CV['mse_mean'], yerr=df_CV['mse_se'],
#                 label='Our algorithm', linestyle=':', fmt='s', capsize=4)
ax.set_ylabel('Estimated overall rating')
ax.set_xlabel('Criteria rating')
ax.legend(fontsize=fontsize)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('plots/understanding_varying_criteria.png', dpi=300, bbox_inches='tight')
plt.close()




# ----------------------------- LLMs Understanding ------------------------ #

dir_names = ['data/LLMs/outputs_gpt/', 'data/LLMs/outputs_llama/', 'data/LLMs/outputs_human/']
names = ['GPT-4o', 'Llama-3.1-70b', 'Human']
df_human = pd.read_csv('data/LLMs/ICLR2024_human_reviews.csv')
modes = [df_human['soundness'].mode()[0], df_human['presentation'].mode()[0], df_human['contribution'].mode()[0]]



def error(MSE, SE, noise):
    return np.sqrt((MSE - noise) + 1.96*SE)

error_human = error(df_stats.loc[df_stats['Name'] == 'Human evaluators', 'MSE_cv'].values[0],
                     df_stats.loc[df_stats['Name'] == 'Human evaluators', 'SE_cv'].values[0],
                     df_stats.loc[df_stats['Name'] == 'Human evaluators', 'noise'].values[0])
error_gpt = error(df_stats.loc[df_stats['Name'] == 'GPT-4o', 'MSE_cv'].values[0],
                     df_stats.loc[df_stats['Name'] == 'GPT-4o', 'SE_cv'].values[0],
                     df_stats.loc[df_stats['Name'] == 'GPT-4o', 'noise'].values[0])
error_llama = error(df_stats.loc[df_stats['Name'] == 'Llama 3.1 70b', 'MSE_cv'].values[0],
                     df_stats.loc[df_stats['Name'] == 'Llama 3.1 70b', 'SE_cv'].values[0],
                     df_stats.loc[df_stats['Name'] == 'Llama 3.1 70b', 'noise'].values[0])
criteria_names = ['Soundness', 'Presentation', 'Contribution']

y_err = [error_gpt, error_llama, error_human]
for val in [2,3]:
    for i in range(3):
            fig, ax = plt.subplots(figsize=(8, 5))
            for iter in range(3):
                df = pd.read_csv(f'{dir_names[iter]}cv_df.csv')
                x_vals, f_vals = np.asarray(df.iloc[:, 1:-1]), np.asarray(df.iloc[:, -1])

                mask = np.ones(len(x_vals), dtype=bool)
                for j in range(3):
                    if j != i:
                        mask &= x_vals[:, j] == val

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
            ax.legend()
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(f'plots/LLM_comparison/varying_{criteria_names[i]}_{val}.png', dpi=300, bbox_inches='tight')
            plt.close()

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

# fixed_vals = [2, 3]
# i=1
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# for ax, fixed_val in zip(axes, fixed_vals):
#     for iter in range(3):
#         df = pd.read_csv(f'{dir_names[iter]}cv_df.csv')
#         x_vals, f_vals = np.asarray(df.iloc[:, 1:-1]), np.asarray(df.iloc[:, -1])
#         mask = np.ones(len(x_vals), dtype=bool)
#         for j in range(3):
#             if j != i:
#                 mask &= x_vals[:, j] == fixed_val
#         varying_vals = x_vals[mask, i].tolist()
#         f_vals_CV = f_vals[mask].tolist()
#         s = styles[iter]
#         ax.errorbar(varying_vals, f_vals_CV,yerr=y_err[iter],
#                 color=s['color'],
#                 marker=s['marker'],
#                 linestyle=s['linestyle'],
#                 linewidth=3,
#                 markersize=10,
#                 label=names[iter])
#     ax.set_ylabel('Estimated overall rating')
#     ax.set_xlabel(f'{criteria_names[i]} rating')
#     ax.set_title(f'Soundness and presentation fixed at {fixed_val} / 4')
#     ax.grid(axis='y', alpha=0.3, linestyle='--')
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles[:3], labels[:3], loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.08))
# plt.tight_layout()
# plt.savefig(f'plots/LLM_comparison/understanding_{criteria_names[i]}.png', dpi=300, bbox_inches='tight')
# plt.close()

# from scipy.stats import mode
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# import matplotlib.colors as mcolors

# df = pd.read_csv('data/LLMs/ICLR2024_human_reviews.csv')
# contribution_mode = df['contribution'].mode()[0]
# modes = [df['presentation'].mode()[0], df['soundness'].mode()[0], df['contribution'].mode()[0]]

# pres_vals = sorted(set(np.concatenate([pd.read_csv(f'{dir_names[i]}cv_df.csv').iloc[:, 1].values for i in range(3)])))
# sound_vals = sorted(set(np.concatenate([pd.read_csv(f'{dir_names[i]}cv_df.csv').iloc[:, 2].values for i in range(3)])))
# P, S = np.meshgrid(pres_vals, sound_vals)

# # Compute all Z values first so we can set a shared color scale
# all_Z = []
# for iter in range(3):
#     df = pd.read_csv(f'{dir_names[iter]}cv_df.csv')
#     x_vals, f_vals = np.asarray(df.iloc[:, 1:-1]), np.asarray(df.iloc[:, -1])
#     Z = np.full_like(P, np.nan, dtype=float)
#     for pi, p in enumerate(pres_vals):
#         for si, s in enumerate(sound_vals):
#             mask = (x_vals[:, 0] == p) & (x_vals[:, 1] == s) & (x_vals[:, 2] == contribution_mode)
#             if mask.sum() > 0:
#                 Z[si, pi] = f_vals[mask].mean()
#     all_Z.append(Z)

# vmin = np.nanmin([Z.min() for Z in all_Z])
# vmax = np.nanmax([Z.max() for Z in all_Z])
# norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# fig = plt.figure(figsize=(18, 5))
# for iter in range(3):
#     ax = fig.add_subplot(1, 3, iter + 1, projection='3d')
#     surf = ax.plot_surface(P, S, all_Z[iter], norm=norm, alpha=0.8, edgecolor='none')
#     ax.set_xlabel('Presentation')
#     ax.set_ylabel('Soundness')
#     ax.set_zlabel('Estimated overall rating')
#     ax.set_title(f'{names[iter]}\n(contribution fixed at {contribution_mode})')

# # Single shared colorbar on the right
# fig.subplots_adjust(right=0.85)
# cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
# fig.colorbar(cm.ScalarMappable(norm=norm), cax=cbar_ax, label='Estimated overall rating')

# plt.suptitle('Surface plot: varying presentation and soundness', y=1.02)
# plt.savefig(f'plots/LLM_comparison/understanding_surface_pres_sound_contrib{contribution_mode}.png',
#             dpi=300, bbox_inches='tight')
# # plt.show()
# plt.close()


# from scipy.stats import mode
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

# contribution_mode = pd.read_csv('data/LLMs/ICLR2024_human_reviews.csv')['contribution'].mode()[0]

# pres_vals = sorted(set(np.concatenate([pd.read_csv(f'{dir_names[i]}cv_df.csv').iloc[:, 1].values for i in range(3)])))
# sound_vals = sorted(set(np.concatenate([pd.read_csv(f'{dir_names[i]}cv_df.csv').iloc[:, 2].values for i in range(3)])))
# P, S = np.meshgrid(pres_vals, sound_vals)


# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# for iter in range(3):
#     df = pd.read_csv(f'{dir_names[iter]}cv_df.csv')
#     x_vals, f_vals = np.asarray(df.iloc[:, 1:-1]), np.asarray(df.iloc[:, -1])

#     Z = np.full_like(P, np.nan, dtype=float)
#     for pi, p in enumerate(pres_vals):
#         for si, s in enumerate(sound_vals):
#             mask = (x_vals[:, 0] == p) & (x_vals[:, 2] == s) & (x_vals[:, 1] == 3)
#             if mask.sum() > 0:
#                 Z[si, pi] = f_vals[mask].mean()

#     surf = ax.plot_surface(P, S, Z, alpha=0.5, edgecolor='none', label=names[iter])
#     surf._facecolors2d = surf._facecolor3d  # needed for legend to work
#     surf._edgecolors2d = surf._edgecolor3d

# ax.set_xlabel('Presentation')
# ax.set_ylabel('Soundness')
# ax.set_zlabel('Estimated overall rating')
# ax.set_title(f'Varying presentation and soundness\n(contribution fixed at {contribution_mode})')
# ax.legend()

# plt.tight_layout()
# plt.savefig(f'plots/LLM_comparison/understanding_surface_pres_sound_contrib{contribution_mode}.png',
#             dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()