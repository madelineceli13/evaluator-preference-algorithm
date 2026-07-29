# imports 
import numpy as np
import pandas as pd
import os 
from sklearn.model_selection import train_test_split
from functions import linear_regression_analysis, isotonic_regression_analysis, cross_validation, estimation_error, unique_xvals, iso_fit_many


num_samples = np.linspace(50, 1000, num=20, dtype = 'int16')
output_dir = 'data/synthetic/'


def simulate_CV(function_name, d = 10, num_vals = 5, T = 30, sd = 0.2, test_size = 0.2):
    y_min, y_max = 0, 1
    file_name = f'{output_dir}{function_name}_d_{d}_n_{num_vals}_sd_02.csv'

    if os.path.exists(file_name):
        os.remove(file_name)
    # Create a fresh file with headers

    df_summary = pd.DataFrame(columns=['N','t','mse_lin','mse_iso','mse_CV'])
    print(f"Attempting to save to: {file_name}")
    print(f"File exists after header write: {os.path.exists(file_name)}")

    df_summary.to_csv(file_name, index=False)
    for N in num_samples:
        for t in range(T):

            # preference paramaters 
            a = np.random.uniform(1, 2, size=(d))
            X_grid = unique_xvals(num_vals, d)
            # random inputs 
            X = np.random.randint(1, num_vals+1, size=(N, d))
            noise = np.random.normal(0, sd, size=N)
    
            # function value
            if function_name == 'linear':
                normalization = num_vals*np.sum(a)
                Y = 1/normalization * (X @ a ) + noise 
                y_true = (X_grid @ a) / normalization
            if function_name == 'leontief':
                normalization = np.max(a)*num_vals 
                weighted_X = X * a
                Y = np.min(weighted_X, axis=1) / normalization + noise
                y_true = np.min(X_grid * a, axis=1) / normalization
            if function_name == 'cobb_douglas':
                normalization = num_vals**np.sum(a) 
                Y = (np.prod(np.power(X, a), axis=1)) / normalization + noise
                y_true = (np.prod(np.power(X_grid, a), axis=1)) / normalization
            
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)            
            # linear regression 
            coefficients, train_mse_lin, test_mse_lin = linear_regression_analysis(X_train, Y_train, X_test, Y_test, y_min = y_min, y_max = y_max, printer_friend= False)
            y_guess = X_grid @ coefficients[0] + coefficients[1]
            mse_lin = estimation_error(y_guess, y_true)

            # isotonic regression
            df_iso, MSE_train, MSE_val = isotonic_regression_analysis(X_train, Y_train, X_test, Y_test, lam = 0, y_min = y_min, y_max = y_max, printer_friend = False)
            x_vals, f_vals = np.asarray(df_iso.iloc[:, :-1]), np.asarray(df_iso.iloc[:,-1])
            y_guess = iso_fit_many(X_grid, x_vals, f_vals)
            mse_iso = estimation_error(y_guess, y_true)
            

            # CV regression
            train_mse_CV, test_mse_CV, df_CV, lam = cross_validation(X_train, Y_train, X_test, Y_test, y_min=0, y_max=1, save = False)

            if lam != np.inf:
                x_vals, f_vals = np.asarray(df_CV.iloc[:, :-1]), np.asarray(df_CV.iloc[:,-1])
                y_guess = iso_fit_many(X_grid, x_vals, f_vals)
            else:
                y_guess = X_grid @ df_CV[0] + df_CV[1]

            mse_CV = estimation_error(y_guess, y_true)

            summary_dict = {
                'N': N,
                't': t,
                'mse_lin': mse_lin,
                'mse_iso': mse_iso,
                'mse_CV': mse_CV,
                'lambda':lam
            }
            df_row = pd.DataFrame([summary_dict])

            # Append to CSV without headers and without index column
            df_row.to_csv(file_name, mode='a', header=False, index=False)

simulate_CV('cobb_douglas', d = 2, num_vals = 5, T = 50, test_size = 0.2)
simulate_CV('leontief', d = 2, num_vals = 5, T = 50, test_size = 0.2)
simulate_CV('linear', d = 2, num_vals = 5, T = 50, test_size = 0.2)