import pandas as pd
import numpy as np
from functions import empirical_simulation, noise_level

# -------------------- All - 2019 ------------------------------ # 
dir_name = 'data/hotelrec/outputs_all_2019/'
name = 'All - 2019'
df = pd.read_csv('data/hotelrec/final_reviews.csv')
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
empirical_simulation(X,Y, dir_name, name)
noise_level(X,Y,name, split = True)
# -------------------- Business ------------------------------ # 
df = pd.read_csv('data/hotelrec/reviews_business.csv')
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
empirical_simulation(X,Y, dir_name, name)
noise_level(X,Y,name, split = True)

# -------------------- Couples ------------------------------ # 
df = pd.read_csv('data/hotelrec/reviews_couple.csv')
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

# -------------------- Family ------------------------------ # 
df = pd.read_csv('data/hotelrec/reviews_family.csv')
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
noise_level(X,Y,name, split = True)
empirical_simulation(X,Y, dir_name, name)

