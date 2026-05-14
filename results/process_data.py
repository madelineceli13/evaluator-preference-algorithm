import pandas as pd
import numpy as np
df_stats = pd.read_csv('results/empirical_stats.csv')
df_noise = pd.read_csv('results/noise_summary.csv')
df = df_stats.merge(df_noise, on='Name')

df['est_error_lin'] = df['MSE_lin'] - df['mean_noise']
df['est_error_cv'] = df['MSE_cv'] - df['mean_noise']
df['percent_improvement_prediction'] = np.round((df['MSE_lin'] - df['MSE_cv']) / df['MSE_lin'] * 100, 1)
df['percent_improvement_estimation'] = np.round((df['est_error_lin'] - df['est_error_cv']) / df['est_error_lin'] * 100, 1)

df = df.drop(columns=['mean_noise', 'se_noise'])
df.to_csv('results/empirical_stats_processed.csv', index=False)

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
