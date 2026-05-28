import pandas as pd
import numpy as np
from functions import empirical_simulation, iso_fit_many, bootstrap_function_estimation, bootstrap_noise
test_size = 0.2
min_val = 1
max_val = 10

# -------------------- GPT ------------------------------ # 
dir_name = 'data/LLMs/outputs_gpt/'
name = 'GPT-4o'
df = pd.read_csv('data/LLMs/ICLR2024.gpt-4o_reviews.csv')

print(f'Number of samples is: {df.shape[0]}')
Y = np.asarray(df['overall'])
X = np.asarray(df[['soundness', 'presentation', 'contribution']])
bootstrap_noise(X,Y,name, B = 1000)
print(df.describe())
empirical_simulation(X,Y, dir_name, name)
# # -------------------- Llama ------------------------------ # 
dir_name = 'data/LLMs/outputs_llama/'
name = 'Llama 3.1 70b'
df = pd.read_csv('data/LLMs/ICLR2024.llama-3.1-70b_reviews.csv')

print(f'Number of samples is: {df.shape[0]}')
Y = np.asarray(df['overall'])
X = np.asarray(df[['soundness', 'presentation', 'contribution']])
bootstrap_noise(X,Y,name, B = 1000)
print(df.describe())
empirical_simulation(X,Y, dir_name, name)

# # -------------------- Human ------------------------------ # 
dir_name = 'data/LLMs/outputs_human/'
name = 'Human evaluators'
df = pd.read_csv('data/LLMs/ICLR2024_human_reviews.csv')

print(f'Number of samples is: {df.shape[0]}')
Y = np.asarray(df['overall'])
X = np.asarray(df[['soundness', 'presentation', 'contribution']])
bootstrap_noise(X,Y,name, B = 1000)
print(df.describe())
empirical_simulation(X,Y, dir_name, name)

# -------------------- Function distance ------------------------------ # 
df_llama = pd.read_csv('data/LLMs/outputs_llama/cv_df.csv')
df_human = pd.read_csv('data/LLMs/outputs_human/cv_df.csv')
df_gpt = pd.read_csv('data/LLMs/outputs_gpt/cv_df.csv')

X_grid = X


x_vals, f_vals = np.asarray(df_gpt.iloc[:, 1:-1]), np.asarray(df_gpt.iloc[:, -1])
y_guess_gpt = iso_fit_many(X_grid, x_vals, f_vals)

x_vals, f_vals = np.asarray(df_human.iloc[:, 1:-1]), np.asarray(df_human.iloc[:, -1])
y_guess_human = iso_fit_many(X_grid, x_vals, f_vals)

x_vals, f_vals = np.asarray(df_llama.iloc[:, 1:-1]), np.asarray(df_llama.iloc[:, -1])
y_guess_llama = iso_fit_many(X_grid, x_vals, f_vals)


print(f"Distance between GPT-4o and Human: {np.mean((y_guess_gpt - y_guess_human)**2):.4f}")
print(f"Distance between Llama-3.1-70b and Human: {np.mean((y_guess_llama - y_guess_human)**2):.4f}")
print(f"Distance between Llama-3.1-70b and GPT-4o: {np.mean((y_guess_llama - y_guess_gpt)**2):.4f}")

print(f"Distance between GPT-4o and Human: {np.mean((y_guess_gpt - y_guess_human)**2):.4f}")
residuals = {
    'comparison': [
        'GPT-4o',
        'Llama 3.1 70b',
    ],
    'mean': [
        np.mean((y_guess_gpt   - y_guess_human)**2),
        np.mean((y_guess_llama - y_guess_human)**2),
    ],
    'std': [
        np.std((y_guess_gpt   - y_guess_human)**2)/np.sqrt(len(X_grid)),
        np.std((y_guess_llama - y_guess_human)**2)/np.sqrt(len(X_grid)),
    ],
}


df_residuals = pd.DataFrame(residuals)
df_residuals.to_csv('data/LLMs/preference_misalignment.csv', index=False)