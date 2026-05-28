import pandas as pd
import numpy as np
from functions import empirical_simulation, bootstrap_noise

# load data and break down by evaluator expertise
df = pd.read_csv("../../solver_evaluator_pairs_13.csv")

df['robotics_expertise'] = (
    (df['score'] >= 13) & (df['TechOrg_RoboticsMechatronics_Years'] >= 2)
).astype(int)

df['expertise'] = df['robotics_expertise'] + df['Discipline_AerospaceDefense']

# multivalent evaluators
df_multivalent = df[df['expertise'] == 2]
df_multivalent = df_multivalent[['solution_feasibility', 'solution_novelty', 'solution_quality']]

# univalent evaluators
df_univalent = df[df['expertise'] == 1]
df_univalent = df_univalent[['solution_feasibility', 'solution_novelty', 'solution_quality']]

# no expertise evaluators
df_no_expertise = df[df['expertise'] == 0]
df_no_expertise = df_no_expertise[['solution_feasibility', 'solution_novelty', 'solution_quality']]


# -------------------- Multivalent ------------------------------ # 

dir_name = 'data/NASA/multivalent/'
name = 'Multivalent evaluators'

# the y-value is the solution quality, and the x-values are the solution feasibility and novelty
Y = np.asarray(df_multivalent['solution_quality'])
X = np.asarray(df_multivalent[['solution_feasibility', 'solution_novelty']])

empirical_simulation(X,Y, dir_name, name, test_size = 0.2, nasa_expertise = True)
bootstrap_noise(X,Y,name, B = 1000)

# -------------------- Univalent ------------------------------ # 
dir_name = 'data/NASA/univalent/'
name = 'Univalent evaluators'

# the y-value is the solution quality, and the x-values are the solution feasibility and novelty
Y = np.asarray(df_univalent['solution_quality'])
X = np.asarray(df_univalent[['solution_feasibility', 'solution_novelty']])

empirical_simulation(X,Y, dir_name, name, test_size = 0.2, nasa_expertise = True)
bootstrap_noise(X,Y,name, B = 1000)

# -------------------- No expertise ------------------------------ # 
dir_name = 'data/NASA/outside_expertise/'
name = 'Outside expertise evaluators'

Y = np.asarray(df_no_expertise['solution_quality'])
X = np.asarray(df_no_expertise[['solution_feasibility', 'solution_novelty']])

empirical_simulation(X,Y, dir_name, name, test_size = 0.2, nasa_expertise = True)
bootstrap_noise(X,Y, name, B = 1000)

# -------------------- All evaluators ------------------------------ # 
dir_name = 'data/NASA/all_evaluators/'
name = 'All evaluators'

Y = np.asarray(df['solution_quality'])
X = np.asarray(df[['solution_feasibility', 'solution_novelty']])

empirical_simulation(X,Y, dir_name, name, test_size = 0.2, nasa_expertise = True)
bootstrap_noise(X,Y,name, B = 1000)
