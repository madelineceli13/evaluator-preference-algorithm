import pandas as pd
import numpy as np
df_stats = pd.read_csv('results/empirical_stats_rebuttal.csv')
df_noise = pd.read_csv('results/noise_summary.csv')
df = df_stats.merge(df_noise, on='Name')

df['est_error_lin'] = df['MSE_lin'] - df['mean_noise']
df['est_error_cv'] = df['MSE_cv'] - df['mean_noise']
df['est_error_iso'] = df['MSE_iso'] - df['mean_noise']
df['est_error_constant'] = df['MSE_constant'] - df['mean_noise']

df['percent_improvement_prediction_cv'] = np.round((df['MSE_lin'] - df['MSE_cv']) / df['MSE_lin'] * 100, 1)
df['percent_improvement_estimation_cv'] = np.round((df['est_error_lin'] - df['est_error_cv']) / df['est_error_lin'] * 100, 1)
df['percent_improvement_prediction_iso'] = np.round((df['MSE_lin'] - df['MSE_iso']) / df['MSE_lin'] * 100, 1)
df['percent_improvement_estimation_iso'] = np.round((df['MSE_lin'] - df['MSE_iso']) / df['MSE_lin'] * 100, 1)


df = df.drop(columns=['mean_noise', 'se_noise'])
df.to_csv('results/empirical_stats_processed_rebuttal.csv', index=False, header = True)

# irreduible error bootstrapping

# for name in ['GPT-4o', 'Human evaluators', 'Llama 3.1 70b']:
#     df = pd.read_csv(f'results/bootstrap_noise_summary_{name}.csv')
#     samples = df['mean_noise'].values

#     mean   = np.mean(samples)
#     lower  = np.percentile(samples, 2.5)
#     upper  = np.percentile(samples, 97.5)

#     result = pd.DataFrame({
#         'name':       [name],
#         'mean_noise': [mean],
#         'lower_ci':   [lower],
#         'upper_ci':   [upper],
#     })

#     result.to_csv('results/irreducible_error.csv', index=False, mode='a', header=not pd.io.common.file_exists('results/irreducible_error.csv'))

# for name in ['All evaluators', 'Multivalent evaluators', 'Univalent evaluators', 'Outside expertise evaluators']:
#     df = pd.read_csv(f'results/bootstrap_noise_summary_{name}.csv')
#     samples = df['mean_noise'].values

#     mean   = np.round(np.mean(samples), 3)
#     lower  = np.round(np.percentile(samples, 2.5), 3)
#     upper  = np.round(np.percentile(samples, 97.5), 3)

#     result = pd.DataFrame({
#         'name':       [name],
#         'mean_noise': [mean],
#         'lower_ci':   [lower],
#         'upper_ci':   [upper],
#     })

#     result.to_csv('results/irreducible_error.csv', index=False, mode='a', header=not pd.io.common.file_exists('results/irreducible_error.csv'))


import pandas as pd

# Load results
df = pd.read_csv("results/empirical_stats_processed_rebuttal.csv")


print(f"{'| Dataset':<22}| {'Linear':>24} | {'CV':>24} | {'Isotonic':>24} | {'Constant':>24} |")
print(f"|:{'-'*20}|{'-'*25}:|{'-'*25}:|{'-'*25}:|{'-'*25}:|")

for _, row in df.iterrows():

    lin = f"{row.est_error_lin:.3f} ({row.SE_lin:.3f})"
    cv = f"{row.est_error_cv:.3f} ({row.SE_cv:.3f})"
    iso = f"{row.est_error_iso:.3f} ({row.SE_iso:.3f})"
    con = f"{row.est_error_constant:.3f} ({row.SE_constant:.3f})"

    print(
        f"| {row['Name']:<20}"
        f"| {lin:>24}"
        f" | {cv:>24}"
        f" | {iso:>24}"
        f" | {con:>24} |"
    )


print(
    "| Dataset | Prediction Error (Lin.) | Prediction Error (CV) | Δ (%) | Reducible Error (Lin.) | Reducible Error (CV) | Δ (%) |"
)
print(
    "|:--------|-------------------------:|----------------------:|------:|-----------------------:|---------------------:|------:|"
)

for _, row in df.iterrows():

    pred_lin = f"{row.MSE_lin:.3f} ({row.SE_lin:.3f})"
    pred_cv = f"{row.MSE_cv:.3f} ({row.SE_cv:.3f})"

    red_lin = f"{row.est_error_lin:.3f} ({row.SE_lin:.3f})"
    red_cv = f"{row.est_error_cv:.3f} ({row.SE_cv:.3f})"

    pred_delta = f"{row.percent_improvement_prediction_cv:.1f}%"
    red_delta = f"**{row.percent_improvement_estimation_cv:.1f}%**"

    print(
        f"| {row['Name']:<18}"
        f"| {pred_lin:>24}"
        f"| {pred_cv:>23}"
        f"| {pred_delta:>6}"
        f"| {red_lin:>24}"
        f"| {red_cv:>22}"
        f"| {red_delta:>7} |"
    )

print(
    "| Dataset | Linear Regression | Our Algorithm | Isotonic Regression | $\\lambda^{\\star}$ |"
)
print(
    "|:--------|------------------:|--------------:|--------------------:|--------------------:|"
)
df['lam']=df['lambda']
for _, row in df.iterrows():

    lin = f"{row.est_error_lin:.3f} ({row.SE_lin:.3f})"
    ours = f"{row.est_error_cv:.3f} ({row.SE_cv:.3f})"
    iso = f"{row.est_error_iso:.3f} ({row.SE_iso:.3f})"

    # Format lambda nicely
    lam = f"{row.lam:.4g}"

    print(
        f"| {row['Name']}"
        f"| {lin}"
        f"| **{ours}**"
        f"| {iso}"
        f"| {lam} |"
    )

print(
    "| Dataset | Constant | Linear regression | Isotonic | Our Algo. |"
)
print(
    "|:--------|-----------------------------:|----------------------------:|---------------------------:|---------------------------:|"
)

for _, row in df.iterrows():

    lin = f"{row.est_error_lin:.3f} ({row.SE_lin:.3f})"
    ours = f"{row.est_error_cv:.3f} ({row.SE_cv:.3f})"
    iso = f"{row.est_error_iso:.3f} ({row.SE_iso:.3f})"
    con = f"{row.est_error_constant:.3f} ({row.SE_constant:.3f})"

    print(
        f"| {row['Name']}"
        f"| {con}"
        f"| {lin}"
        f"| {iso}"
        f"| {ours} |"
    )

# --- load raw pieces and merge (mirrors your existing merge) ----------------
df_stats = pd.read_csv('results/empirical_stats_rebuttal.csv')
df_noise = pd.read_csv('results/noise_summary.csv')
df_add   = pd.read_csv('results/additional_models_stats.csv')

# additional_models APPENDS, so keep the latest run per dataset
df_add = df_add.drop_duplicates(subset='Name', keep='last')

df = (df_stats
      .merge(df_noise, on='Name')
      .merge(df_add[['Name', 'MSE_nn', 'SE_nn', 'MSE_gbm', 'SE_gbm']],
             on='Name', how='inner'))          # inner -> only datasets with NN/GBM

# --- reducible (estimation) error = MSE - mean_noise, per method ------------
for m in ['lin', 'cv', 'nn', 'gbm']:
    df[f'est_error_{m}'] = df[f'MSE_{m}'] - df['mean_noise']

# --- table: reducible error with SE in parentheses --------------------------
print("| Dataset | Linear regression | Monotone NN | Monotone GBM | Our Algo. |")
print("|:--------|------------------:|------------:|-------------:|----------:|")

for _, row in df.iterrows():
    lin  = f"{row.est_error_lin:.3f} ({row.SE_lin:.3f})"
    nn   = f"{row.est_error_nn:.3f} ({row.SE_nn:.3f})"
    gbm  = f"{row.est_error_gbm:.3f} ({row.SE_gbm:.3f})"
    ours = f"{row.est_error_cv:.3f} ({row.SE_cv:.3f})"

    print(
        f"| {row['Name']}"
        f"| {lin}"
        f"| {nn}"
        f"| {gbm}"
        f"| {ours} |"
    )