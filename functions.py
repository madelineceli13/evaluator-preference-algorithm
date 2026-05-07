# packages
import numpy as np
import cvxpy as cvx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time


def linear_regression_analysis(X_train, Y_train, X_test, Y_test, y_min, y_max, printer_friend = True, CV = False):
    
    # Fit a linear regression model --- we enforce nonnegative regression coefficients 
    linreg = LinearRegression(fit_intercept=True, positive = True)
    linreg.fit(X_train, Y_train)

    # Get coefficients
    coefficients = linreg.coef_

    # Predict and calculate mean squared error on training and testing data
    Y_train_pred = linreg.predict(X_train)
    Y_test_pred = linreg.predict(X_test)

    # clip outputs if doing CV
    if CV == True:
      Y_train_pred = np.clip(Y_train_pred, y_min, y_max)
      Y_test_pred = np.clip(Y_test_pred, y_min, y_max)  
    train_mse = mean_squared_error(Y_train, Y_train_pred)
    test_mse = mean_squared_error(Y_test, Y_test_pred)
    

    if printer_friend == True:
        print("---------------------------------")
        print("Linear regression results")
        print("Coefficients:", np.round(coefficients, 3))
        print("Intercept:", np.round(linreg.intercept_, 3))
        print("MSE_train:", round(train_mse, 3))
        print("MSE_test:", round(test_mse, 3))
        print("---------------------------------")

    # Return values as well for programmatic use
    return ((np.asarray(linreg.coef_), np.asarray(linreg.intercept_)), train_mse, test_mse)


def inverse_index(X, vec):
    """
    Find the row index of vector `vec` in matrix `X`.
    Returns the index if found; otherwise, returns -1.
    """

    # Create a boolean mask comparing `vec` to each row in X
    mask = np.all(X == vec, axis=1)
    indices = np.where(mask)[0]
    return indices[0]


# istotonic fit --- compute prediction for unseen datapoints
def iso_fit(z, x_vals, f_vals):
    """
    Predict f(z) given known pairs (x, f(x)) in training and linear fit function linear_fit(z).

    Returns:
    - predicted scalar f(z).
    """
    # Separate inputs and function values from df_unique
    X = x_vals # criteria scores seen in training
    F = f_vals # learning values for those scores

    # Check if z matches an existing x in X
    for i, x in enumerate(X):
        if np.array_equal(x, z):
            return F[i]

    # Define partial order checks coordinate-wise
    less_equal_mask = np.all(X <= z, axis=1)   # Boolean mask for A = {x in X: x <= z}
    greater_equal_mask = np.all(X >= z, axis=1) # Boolean mask for B = {x in X: x >= z}

    A_vals = F[less_equal_mask]
    B_vals = F[greater_equal_mask]

    # Handle empty sets as per instructions
    min_f = np.min(F)
    max_f = np.max(F)

    if len(A_vals) == 0:
        A_z = min_f
    else:
        A_z = np.max(A_vals)

    if len(B_vals) == 0:
        B_z = max_f
    else:
        B_z = np.min(B_vals)
    
    return (A_z + B_z) / 2

# function to compute loss given dataset
def MSE_isotonic(X_vals, Y_vals, x_vals, f_vals):
  loss = 0
  n = X_vals.shape[0]
  for i in range(n):
    x = X_vals[i,:]
    y_hat = iso_fit(x, x_vals, f_vals)
    loss += (y_hat-Y_vals[i])**2
  return loss/n


# function that combines everything into one: given input data we output errors and function
def isotonic_regression_analysis(X_train, Y_train, X_test, Y_test, lam, y_min, y_max, printer_friend = True):

    (x_vals, f_vals) = isotonic_regression(X_train, Y_train, lam, y_min, y_max)
    min_val = np.min(np.concatenate((Y_train, Y_test)))
    max_val = np.max(np.concatenate((Y_train, Y_test)))
    f_vals = np.clip(f_vals, min_val, max_val)
    MSE_train = MSE_isotonic(X_train, Y_train, x_vals, f_vals)
    MSE_test = MSE_isotonic(X_test, Y_test, x_vals, f_vals)
    df = pd.DataFrame(x_vals)
    df['function_value'] = f_vals

    if printer_friend == True:
        print("---------------------------------")
        print("Isotonic regression results")
        print(f'MSE_train: { np.round(MSE_train,3)}')
        print(f'MSE_test: { np.round(MSE_test,3)}')
        print("---------------------------------")

    return df, MSE_train, MSE_test

# create set of regularization parameters
exponents = np.arange(-9, 9, dtype=float)  
powers_of_two = 2.0 ** exponents          
Lambda = np.append(powers_of_two, [0, np.inf])
Lambda = np.sort(Lambda)

def isotonic_regression(X_train, Y_train, lam, y_min, y_max):

  # compute functional fit 
  num_samples = X_train.shape[0]
  num_criteria = X_train.shape[1]

  # determine the set of unique x-values
  unique_x_vals, inverse_indices = np.unique(X_train,axis=0,return_inverse=True)

  # this function maps training data to the respective function value 
  f = cvx.Variable(len(unique_x_vals))	# we have a variable for f-value on each possible x value
  a = cvx.Variable(num_criteria, nonneg = True)
  c = cvx.Variable()

  constraints = []

  # vectorized finding of minimal set of monotonicity constraints
  mask = (unique_x_vals[:, np.newaxis, :] >= unique_x_vals[np.newaxis, :, :]).all(axis=2)  # shape (n, n)
  np.fill_diagonal(mask, False)

  mask = mask & ~(mask @ mask).astype(bool)

  i_indices, j_indices = np.where(mask)
  constraints = [f[i] >= f[j] for i, j in zip(i_indices, j_indices)]

  # defined squared loss
  loss_data = cvx.pnorm(f[inverse_indices]-Y_train,2)**2/num_samples

  # loss 
  linear_fit = unique_x_vals @ a + c
  loss_linear = lam*cvx.pnorm(f - linear_fit)**2/len(unique_x_vals)

  # our objective
  loss = loss_data + loss_linear
  obj = cvx.Minimize(loss)

  # solution
  prob = cvx.Problem(obj, constraints)

  prob.solve(
    verbose=True,
    solver=cvx.SCS)
  
  f_values = np.clip(f.value, y_min, y_max)  # clip to training range for CV case
  
  if prob.status in ["optimal_inaccurate", "AlmostSolved"]:
    f_vals = f.value  
    violations = np.maximum(0, f_vals[j_indices] - f_vals[i_indices])
    print(f"Max constraint violation:  {violations.max():.2e}")

  if None in f.value:  
    print("Warning: some variables have None values — solution may be incomplete")
  return (unique_x_vals, f_values)


def cross_validation(X_train, Y_train, X_test, Y_test, y_min, y_max, Lambda = Lambda, save = True, dir_name = 'Athena'):

  start = time.time()

  # we use a 3 - way split 20% test, 20% validation, 60% train 
  X_train_CV, X_val, Y_train_CV, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)
  df_summary = pd.DataFrame(columns=['lambda', 'train_mse', 'val_mse'])

    # Only create the summary CSV if saving
  if save:
    df_summary.to_csv(f'{dir_name}CV_summary.csv', index=False)

  # save 
  val_errors = []
  for lam in Lambda:
    if lam != np.inf:
      df, MSE_train, MSE_val = isotonic_regression_analysis(X_train_CV, Y_train_CV, X_val, Y_val, lam = lam, y_min = y_min, y_max = y_max, printer_friend = False)
      if save:
        df.to_csv(f'{dir_name}_{lam}_df.csv', index=False)   
    else:
      df, MSE_train, MSE_val = linear_regression_analysis(X_train_CV, Y_train_CV, X_val, Y_val, y_min = 0, y_max = 1,printer_friend = False, CV = True)
    if save:
      summary_dict = {'lambda': lam, 'train_mse': MSE_train, 'val_mse': MSE_val}
      df_row = pd.DataFrame([summary_dict])
      df_row.to_csv(f'{dir_name}CV_summary.csv', mode='a', header=False, index=False)
    val_errors.append(MSE_val)

  
  lam_star_idx = np.argmin(val_errors)
  lam_star = Lambda[lam_star_idx]

  if lam_star != np.inf:
    df, MSE_train, MSE_test = isotonic_regression_analysis(X_train, Y_train, X_test, Y_test, lam = lam_star, y_min = y_min, y_max = y_max, printer_friend = False)       
  else:
    df, MSE_train, MSE_test = linear_regression_analysis(X_train, Y_train, X_test, Y_test, y_min = 0, y_max = 1,printer_friend = False, CV = True)
  end = time.time()
  print(f"Solve time: {end-start}")
  print("---------------------------------")
  print("Cross-Validation regression results")
  print(f'optimal lambda is {Lambda[lam_star_idx]}')
  print(f'MSE_train: { np.round(MSE_train,3)}')
  print(f'MSE_test: { np.round(MSE_test,3)}')
  print("---------------------------------")
  
  return MSE_train, MSE_test, df, lam_star


def empirical_simulation(X,Y, dir_name, name, test_size = 0.2):
    y_min = Y.min()
    y_max = Y.max()
    start_time = time.time()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    # linear regression 
    coefficients, train_mse_lin, test_mse_lin = linear_regression_analysis(X_train, Y_train, X_test, Y_test, y_min=0, y_max =1, printer_friend= True)

    # CV regression
    train_mse_CV, test_mse_CV, df_CV, lam = cross_validation(X_train, Y_train, X_test, Y_test, y_min, y_max, Lambda = Lambda, save = True, dir_name = dir_name)
    
    if lam!=np.inf:
        df_CV.to_csv(f'{dir_name}cv_df.csv')   
    df = pd.read_csv(f'{dir_name}cv_df.csv')
    compute_empirical_stats(df, X_test, Y_test, name, coefficients)

    end_time = time.time()
    print(f'total time is {end_time - start_time} seconds')
    return coefficients


## compute summary statistics
def compute_empirical_stats(df, X_test, Y_test, dir_name, a):
    x_vals, f_vals = np.asarray(df.iloc[:, 1:-1]), np.asarray(df.iloc[:,-1])
    errors = error_isotonic(X_test, Y_test, x_vals, f_vals)
    MSE_cv = np.mean(errors)
    SE_cv = np.std(errors) / np.sqrt(len(Y_test))
    print(np.std(errors))
    Y_hat =  X_test @ a[0] + a[1]
    errors = (Y_hat - Y_test) ** 2
    MSE_lin = np.mean(errors)
    SE_lin = np.std(errors) / np.sqrt(len(Y_test))
    print(np.std(errors))
    noise = noise_level(X_test, Y_test, y_min = np.min(Y_test), y_max = np.max(Y_test), name = dir_name)
    summary_dict = {
                'Name': dir_name,
                'MSE_lin': MSE_lin,
                'SE_lin': SE_lin, 
                'MSE_cv':MSE_cv,
                'SE_cv': SE_cv,
                'noise': noise,
            }
    df_row = pd.DataFrame([summary_dict])

    # Append to CSV without headers and without index column
    df_row.to_csv('results/empirical_stats.csv', mode='a', header=False, index=False)

# old method for computing noise
# def compute_variances(X,Y):
#     unique_vals = np.unique(X, axis=0)
#     n = X.shape[0]
#     EV = 0
#     total_number = 0
#     for x in unique_vals:
#         indices = np.where((X == x).all(axis=1))[0]
#         if len(indices) > 1:
#             variance = np.var(Y[indices]) #-np.mean(Y[indices]))**2/(len(indices)-1)
#         else:
#             variance = 0
#         EV += variance * len(indices)
#         total_number += len(indices)
#     print(f'Expected Variance: {EV / total_number}')
#     return EV / n


def error_isotonic(X_vals, Y_vals, x_vals, f_vals):
  y_hat = np.empty(Y_vals.shape[0])
  n = X_vals.shape[0]
  for i in range(n):
    x = X_vals[i,:]
    y_hat[i] = iso_fit(x, x_vals, f_vals)
  return (y_hat-Y_vals)**2


# ------------------------- FUNCTIONS FOR ANALYZING SYNTHETIC SIMULATIONS --------------------------- #
def iso_fit_many(Z, x_vals, f_vals):
    """
    Vectorized prediction for multiple z points.
    Z: shape (n_query, d) - multiple points to predict
    Returns: shape (n_query,) predictions
    """
    X = x_vals  # shape (n_unique, d)
    F = f_vals  # shape (n_unique,)
    n_query = Z.shape[0]
    
    # Exact matches (vectorized)
    match_mask = np.all(Z[:, None, :] == X[None, :, :], axis=2)  # (n_query, n_unique)
    exact_matches = match_mask.any(axis=1)
    predictions = np.full(n_query, np.nan)
    
    # Handle exact matches first
    match_indices = np.where(match_mask)
    predictions[match_indices[0]] = F[match_indices[1]]
    
    # For non-exact matches, compute A and B sets
    remaining = ~exact_matches # remaining as those where exact matches is not true
    if np.any(remaining):
        Z_rem = Z[remaining]
        n_rem = Z_rem.shape[0]
        
        # Vectorized partial order masks
        le_mask = np.all(Z_rem[:, None, :] >= X[None, :, :], axis=2)  # (n_rem, n_unique)
        ge_mask = np.all(Z_rem[:, None, :] <= X[None, :, :], axis=2)
        
        # max or min values if dominated and domnating sets are empty
        A_vals = np.full(n_rem, np.min(F))
        B_vals = np.full(n_rem, np.max(F))
        
        # if sets are nonempty, fill in with relevant values 
        for i in range(n_rem):
            A_idx = np.where(le_mask[i])[0]
            if len(A_idx) > 0:
                A_vals[i] = np.max(F[A_idx])
            B_idx = np.where(ge_mask[i])[0]
            if len(B_idx) > 0:
                B_vals[i] = np.min(F[B_idx])
        
        predictions[remaining] = 0.5 * (A_vals + B_vals)
    
    return predictions

def estimation_error(y_guess, y_true):
    y_guess = np.asarray(y_guess)
    y_true = np.asarray(y_true)
    return np.mean((y_guess - y_true) ** 2)

def unique_xvals(num_vals, num_criteria):
    # Create a grid of all points in [n]^d
    grids = [np.arange(1,num_vals+1) for _ in range(num_criteria)]   # list of arrays [1,2,...,n] repeated d times
    mesh = np.meshgrid(*grids, indexing='ij')  # d arrays of shape (n,n,...,n)
    X_grid = np.vstack([g.ravel() for g in mesh]).T  # stack and transpose to get shape (n^d, d)
    return X_grid

def noise_level(X,Y,name, split = False, min_val=-1, max_val=-1, bootstrap = False):
    if min_val == -1:
        min_val = np.min(Y)
        max_val = np.max(Y)
    if split == True:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X = X_test
        Y = Y_test
    # Compute the noise level as the standard deviation of the residuals
    (unique_x_vals, f_values) = isotonic_regression(X, Y, 0, min_val, max_val)
    residuals = (Y - iso_fit_many(X, unique_x_vals, f_values))**2
    df_row = pd.DataFrame({
       'name': name,
        'mean_noise': [np.mean(residuals)],
        'se_noise': [np.std(residuals)/np.sqrt(len(Y))]

    })
    if bootstrap == False:
        df_row.to_csv(f'results/noise_summary.csv', mode='a', header=False, index=False)
        return residuals 
    else:
       return np.mean(residuals)


from sklearn.utils import resample
def bootstrap_noise(X,Y,name, B = 1000):
   df = pd.DataFrame({
       'number': [],
       'mean_noise': [],
   })
   df.to_csv(f'results/bootstrap_noise_summary_{name}.csv', index=False)
   samples = []
   for i in range(B):
        X_sample, Y_sample = resample(X, Y, replace=True, random_state=i)
        noise_sample =noise_level(X_sample, Y_sample, name = f'{name}_bootstrap_{i}', split = False, bootstrap = True)
        samples.append(noise_sample)
        df_row = pd.DataFrame({
            'number': [i],
            'mean_noise': [noise_sample],
        })
        df_row.to_csv(f'results/bootstrap_noise_summary_{name}.csv', mode='a', header=False, index=False)
   return samples
