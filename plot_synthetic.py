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
plt.show()
