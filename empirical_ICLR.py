import pandas as pd
import numpy as np
from functions import empirical_simulation, noise_level

# -------------------- ICLR ------------------------------ # 
# 2023
dir_name = 'data/ICLR/outputs_2023/'
name = 'ICLR 2023'
# some data checks to make sure everything is reasonable
df = pd.read_csv('../data/ICLR/ICLR_reviews_2023.csv')
print(df.head())
print(df[['recommendation','empirical_novelty_and_significance','technical_novelty_and_significance','correctness']].describe())
df = df[['recommendation','empirical_novelty_and_significance','technical_novelty_and_significance','correctness']]
print(df.isna().sum())
print(df['recommendation'].unique())
print(df['empirical_novelty_and_significance'].unique())
print(df['technical_novelty_and_significance'].unique())
print(df['correctness'].unique())

df['empirical_novelty_and_significance'] = pd.to_numeric(df['empirical_novelty_and_significance'], errors='coerce')

print(f'Dataframe shape before dropping NA: {df.shape}')
df = df.dropna(how='any')

print(f'Dataframe shape after dropping NA: {df.shape}')

# the y-value is the solution quality, and the x-values are the solution feasibility and novelty
Y = np.asarray(df['recommendation'])
X = np.asarray(df[['correctness','empirical_novelty_and_significance','technical_novelty_and_significance']])

empirical_simulation(X,Y, dir_name, name)
# noise_level(X,Y,name, split = True)

# 2024
dir_name = 'data/ICLR/outputs_2024/'
name = 'ICLR 2024'
# some data checks to make sure everything is reasonable
df = pd.read_csv('../data/ICLR/ICLR_reviews_2024.csv')
print(df.head())
print(df[['rating','soundness','presentation','contribution']].describe())
df = df[['rating','soundness','presentation','contribution']]
print(df.isna().sum())
print(df['rating'].unique())
print(df['soundness'].unique())
print(df['presentation'].unique())
print(df['contribution'].unique())

print(f'Dataframe shape before dropping NA: {df.shape}')
df = df.dropna(how='any')
print(f'Dataframe shape after dropping NA: {df.shape}')
# the y-value is the solution quality, and the x-values are the solution feasibility and novelty
Y = np.asarray(df['rating'])
X = np.asarray(df[['soundness','presentation','contribution']])
empirical_simulation(X,Y, dir_name, name)
# noise_level(X,Y,name, split = True)

# 2025
dir_name = 'data/ICLR/outputs_2025/'
name = 'ICLR 2025'
# some data checks to make sure everything is reasonable
df = pd.read_csv('../data/ICLR/ICLR_reviews_2025.csv')
print(df.head())
print(df[['rating','soundness','presentation','contribution']].describe())
df = df[['rating','soundness','presentation','contribution']]
print(df.isna().sum())
print(df['rating'].unique())
print(df['soundness'].unique())
print(df['presentation'].unique())
print(df['contribution'].unique())

print(f'Dataframe shape before dropping NA: {df.shape}')
df = df.dropna(how='any')
print(f'Dataframe shape after dropping NA: {df.shape}')

# the y-value is the solution quality, and the x-values are the solution feasibility and novelty
Y = np.asarray(df['rating'])
X = np.asarray(df[['soundness','presentation','contribution']])
empirical_simulation(X,Y, dir_name, name)
# noise_level(X,Y,name, split = True)
