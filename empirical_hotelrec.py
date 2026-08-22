import pandas as pd
import numpy as np
from functions import empirical_simulation, noise_level
from additional_models import additional_models


NN_GRID_MID = {
    "hidden": [32, 64, 128],
    "depth": [1, 2, 3],
    "activation": ["elu"],
    "lr": [1e-2, 1e-3],
    "weight_decay": [1e-4],
    "batch_size": [256, 1024],
    "max_epochs": [200],       # fewer epochs needed now — each does many steps
    "patience": [30],
}

GBM_GRID_MID = {
    "n_estimators": [1000],        # higher ceiling; early stopping picks the count
    "max_depth": [3, 4, 5, 6],     # extend past the 4 it kept hitting
    "learning_rate": [0.02, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [1.0],
    "min_child_weight": [1, 5],
    "reg_lambda": [1.0],
    "patience": [50],
}   # 96 configs — trim min_child_weight or subsample to [1]/[1.0] to halve it

# -------------------- All - 2019 ------------------------------ # 
dir_name = 'data/hotelrec/outputs_all_2019/'
name = 'All - 2019'
df = pd.read_csv('../data/hotelrec/reviews_final.csv')
df = df[df['year']>=2019] 

print(df.head())
print(df.describe())
df = df.iloc[:,1:]
print(df.head())
print(df.isna().sum())
for i in range(df.shape[1]):
    print(df.iloc[:, i].unique())
print(df.iloc[:,1:].head())
criteria  = df.columns[1:]

Y = np.asarray(df['rating'])
X = np.asarray(df.iloc[:,1:])
additional_models(X, Y, dir_name, name, nn_grid=NN_GRID_MID, gbm_grid=GBM_GRID_MID,
                 hpo_subsample=50000)
empirical_simulation(X,Y, dir_name, name)

# -------------------- Business ------------------------------ # 
df = pd.read_csv('../data/hotelrec/reviews_business.csv')
df = df[df['year']>=2014]
dir_name = 'data/hotelrec/outputs_business/'
name = 'Business'

print(df.head())
print(df.describe())
df = df.iloc[:,1:]
print(df.head())
print(df.isna().sum())
for i in range(df.shape[1]):
    print(df.iloc[:, i].unique())
print(df.iloc[:,1:].head())
print(df.shape)

Y = np.asarray(df['rating'])
X = np.asarray(df[['service','location', 'value', 'cleanliness', 'sleep quality', 'rooms']])
additional_models(X, Y, dir_name, name, nn_grid=NN_GRID_MID, gbm_grid=GBM_GRID_MID,
                  hpo_subsample=50000)
empirical_simulation(X,Y, dir_name, name)

# -------------------- Couples ------------------------------ # 
df = pd.read_csv('../data/hotelrec/reviews_couple.csv')
df = df[df['year']>=2014]
dir_name = 'data/hotelrec/outputs_couple/'
name = 'Couples'

print(df.head())
print(df.describe())
df = df.iloc[:,1:]
print(df.head())
print(df.isna().sum())
for i in range(df.shape[1]):
    print(df.iloc[:, i].unique())
print(df.iloc[:,1:].head())
print(df.shape)

Y = np.asarray(df['rating'])
X = np.asarray(df[['service','location', 'value', 'cleanliness', 'sleep quality', 'rooms']])

empirical_simulation(X,Y, dir_name, name)
noise_level(X,Y,name, split = True)
additional_models(X, Y, dir_name, name, nn_grid=NN_GRID_MID, gbm_grid=GBM_GRID_MID,
                  hpo_subsample=50000)
# -------------------- Family ------------------------------ # 
df = pd.read_csv('../data/hotelrec/reviews_family.csv')
df = df[df['year']>=2014]
dir_name = 'data/hotelrec/outputs_family/'
name = 'Family'

print(df.head())
print(df.describe())
df = df.iloc[:,1:]
print(df.head())
print(df.isna().sum())
for i in range(df.shape[1]):
    print(df.iloc[:, i].unique())
print(df.iloc[:,1:].head())
print(df.shape)

Y = np.asarray(df['rating'])
X = np.asarray(df[['service','location', 'value', 'cleanliness', 'sleep quality', 'rooms']])
additional_models(X, Y, dir_name, name, nn_grid=NN_GRID_MID, gbm_grid=GBM_GRID_MID,
                  hpo_subsample=50000)
empirical_simulation(X,Y, dir_name, name)
