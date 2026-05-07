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

# used rounding code to make tables 
# df['est_error_lin'] = np.round(df['est_error_lin'], 3)
# df['est_error_cv'] = np.round(df['est_error_cv'], 3)
# df['EV_test'] = np.round(df['EV_test'], 3)
# df['MSE_lin'] = np.round(df['MSE_lin'], 3)
# df['MSE_cv'] = np.round(df['MSE_cv'], 3)
# df['est_error_lin'] = np.round(df['est_error_lin'], 3)
# df['SE_lin'] = np.round(df['SE_lin'], 3)
# df['SE_cv'] = np.round(df['SE_cv'], 3)
