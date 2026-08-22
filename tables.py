import pandas as pd
import numpy as np
df_stats = pd.read_csv('results/empirical_stats.csv')
df_noise = pd.read_csv('results/noise_summary.csv')
df = df_stats.merge(df_noise, on='Name')

df['est_error_lin'] = df['MSE_lin'] - df['mean_noise']
df['est_error_cv'] = df['MSE_cv'] - df['mean_noise']
df['est_error_iso'] = df['MSE_iso'] - df['mean_noise']
df['est_error_constant'] = df['MSE_constant'] - df['mean_noise']

df['percent_improvement_prediction_cv'] = np.round((df['MSE_lin'] - df['MSE_cv']) / df['MSE_lin'] * 100, 1)
df['percent_improvement_estimation_cv'] = np.round((df['est_error_lin'] - df['est_error_cv']) / df['est_error_lin'] * 100, 1)
df['percent_improvement_prediction_iso'] = np.round((df['MSE_lin'] - df['MSE_iso']) / df['MSE_lin'] * 100, 1)
df['percent_improvement_estimation_iso'] = np.round((df['est_error_lin'] - df['est_error_iso']) / df['est_error_lin'] * 100, 1)


df = df.drop(columns=['mean_noise', 'se_noise'])
df.to_csv('results/empirical_stats_processed.csv', index=False, header = True)

# irreduible error bootstrapping

for name in ['GPT-4o', 'Human evaluators', 'Llama 3.1 70b']:
    df = pd.read_csv(f'results/bootstrap_noise_summary_{name}.csv')
    samples = df['mean_noise'].values

    mean   = np.mean(samples)
    lower  = np.percentile(samples, 2.5)
    upper  = np.percentile(samples, 97.5)

    result = pd.DataFrame({
        'name':       [name],
        'mean_noise': [mean],
        'lower_ci':   [lower],
        'upper_ci':   [upper],
    })

    result.to_csv('results/irreducible_error.csv', index=False, mode='a', header=not pd.io.common.file_exists('results/irreducible_error.csv'))

for name in ['All evaluators', 'Multivalent evaluators', 'Univalent evaluators', 'Outside expertise evaluators']:
    df = pd.read_csv(f'results/bootstrap_noise_summary_{name}.csv')
    samples = df['mean_noise'].values

    mean   = np.round(np.mean(samples), 3)
    lower  = np.round(np.percentile(samples, 2.5), 3)
    upper  = np.round(np.percentile(samples, 97.5), 3)

    result = pd.DataFrame({
        'name':       [name],
        'mean_noise': [mean],
        'lower_ci':   [lower],
        'upper_ci':   [upper],
    })

    result.to_csv('results/irreducible_error.csv', index=False, mode='a', header=not pd.io.common.file_exists('results/irreducible_error.csv'))


# Load results
df = pd.read_csv("results/empirical_stats_processed.csv")


print(f"{'| Dataset':<22}| {'Constant':>24} | {'Lin. reg.':>24} | {'Isotonic reg.':>24} | {'Our algo':>24} |")
print(f"|:{'-'*20}|{'-'*25}:|{'-'*25}:|{'-'*25}:|{'-'*25}:|")

for _, row in df.iterrows():

    lin = f"{row.est_error_lin:.3f} ({row.SE_lin:.3f})"
    cv = f"{row.est_error_cv:.3f} ({row.SE_cv:.3f})"
    iso = f"{row.est_error_iso:.3f} ({row.SE_iso:.3f})"
    con = f"{row.est_error_constant:.3f} ({row.SE_constant:.3f})"

    print(
        f"| {row['Name']:<20}"
        f"| {con:>24}"
        f" | {lin:>24}"
        f" | {iso:>24}"
        f" | {cv:>24} |"
    )


print(
    "| Dataset | Prediction Error (Lin. reg.) | Prediction Error (Our algo) | Δ (%) | Reducible Error (Lin. reg.) | Reducible Error (Our algo) | Δ (%) |"
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
    "| Dataset | Lin. reg. | Isotonic reg. | Our algo | $\\lambda^{\\star}$ |"
)
print(
    "|:--------|------------------:|--------------:|--------------------:|--------------------:|"
)
df['lam']=df['lambda']
for _, row in df.iterrows():

    lin = f"{row.est_error_lin:.4f} ({row.SE_lin:.4f})"
    ours = f"{row.est_error_cv:.7f} ({row.SE_cv:.7f})"
    iso = f"{row.est_error_iso:.7f} ({row.SE_iso:.7f})"

    # Format lambda nicely
    lam = f"{row.lam:.4g}"

    print(
        f"| {row['Name']}"
        f"| {lin}"
        f"| {iso}"
        f"| **{ours}**"
        f"| {lam} |"
    )

print(
    "| Dataset | Constant | Lin. reg. | Isotonic reg. | Our algo |"
)
print(
    "|:--------|-----------------------------:|----------------------------:|---------------------------:|---------------------------:|"
)

for _, row in df.iterrows():

    lin = f"{row.est_error_lin:.4f} ({row.SE_lin:.4f})"
    ours = f"{row.est_error_cv:.4f} ({row.SE_cv:.4f})"
    iso = f"{row.est_error_iso:.4f} ({row.SE_iso:.4f})"
    con = f"{row.est_error_constant:.4f} ({row.SE_constant:.4f})"

    print(
        f"| {row['Name']}"
        f"| {con}"
        f"| {lin}"
        f"| {iso}"
        f"| {ours} |"
    )

# --- load raw pieces and merge (mirrors your existing merge) ----------------
df_stats = pd.read_csv('results/empirical_stats.csv')
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
print("| Dataset | Monotone NN | Monotone GBM | Our algo |")
print("|:--------|------------:|-------------:|----------:|")

for _, row in df.iterrows():
    lin  = f"{row.est_error_lin:.3f} ({row.SE_lin:.3f})"
    nn   = f"{row.est_error_nn:.3f} ({row.SE_nn:.3f})"
    gbm  = f"{row.est_error_gbm:.3f} ({row.SE_gbm:.3f})"
    ours = f"{row.est_error_cv:.3f} ({row.SE_cv:.3f})"

    print(
        f"| {row['Name']}"
        f"| {nn}"
        f"| {gbm}"
        f"| {ours} |"
    )


import math

GROUPS = [
    ("Tripadvisor", ["All - 2019", "Business", "Couples", "Family"]),
    ("ICLR peer reviews", ["ICLR 2023", "ICLR 2024", "ICLR 2025", "ICLR 2026"]),
    ("LLM Case Study", ["GPT-4o", "Llama 3.1 70b", "Human evaluators"]),
    ("NASA", ["All evaluators", "Multivalent evaluators", "Univalent evaluators", "Outside expertise evaluators"])
]

SYNTHETIC_PREFERENCES = [
    ("Cobb-Douglas", "data/synthetic/cobb_douglas_d_2_n_5_sd_02.csv"),
    ("Linear", "data/synthetic/linear_d_2_n_5_sd_02.csv"),
    ("Leontief", "data/synthetic/leontief_d_2_n_5_sd_02.csv"),
]

SYNTHETIC_COLUMNS = [
    ("$\\lambda = 0$", lambda lam: lam == 0),
    ("$0 < \\lambda < 1$", lambda lam: (lam > 0) & (lam < 1)),
    ("$1 \\leq \\lambda < \\infty$", lambda lam: (lam >= 1) & (lam != math.inf)),
    ("$\\lambda = \\infty$", lambda lam: lam == math.inf),
]


def format_lambda(lam):
    if lam == 0.0:
        return "$0$"
    exponent = round(math.log2(lam))
    return f"$2^{{{exponent}}}$"


def print_empirical_lambda_table():
    df = pd.read_csv("results/empirical_stats.csv").set_index("Name")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Dataset & $\lambda^{\star}$ \\",
        r"\midrule",
    ]
    for group_name, dataset_names in GROUPS:
        lines.append(rf"\textit{{{group_name}}} & \\")
        for name in dataset_names:
            lam = df.loc[name, "lambda"]
            lines.append(rf"\quad {name} & {format_lambda(lam)} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Optimal regularization parameter $\lambda^{\star}$ selected via cross-validation for each dataset.}",
        r"\label{tab:lambda_star}",
        r"\end{table}",
    ]

    print("\n".join(lines))


def print_synthetic_lambda_table():
    rows = []
    for label, path in SYNTHETIC_PREFERENCES:
        lam = pd.read_csv(path)["lambda"]
        pcts = [100 * cond(lam).mean() for _, cond in SYNTHETIC_COLUMNS]
        rows.append((label, pcts))

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{l" + "c" * len(SYNTHETIC_COLUMNS) + "}",
        r"\toprule",
        "Preferences & " + " & ".join(name for name, _ in SYNTHETIC_COLUMNS) + r" \\",
        r"\midrule",
    ]
    for label, pcts in rows:
        formatted = " & ".join(f"{p:.1f}\\%" for p in pcts)
        lines.append(f"{label} & {formatted} " + r"\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Distribution of the cross-validation-selected $\lambda$ across synthetic simulations, "
        r"averaged over all sample sizes ($N = 50, \dots, 1000$) and trials, by preference structure.}",
        r"\label{tab:synthetic_lambda}",
        r"\end{table}",
    ]

    print("\n".join(lines))


print_empirical_lambda_table()
print()
print_synthetic_lambda_table()