import matplotlib.pyplot as plt
import numpy as np
from functions import unique_xvals, iso_fit_many
import pandas as pd
plt.rcParams['font.size'] = 24
plt.rcParams['font.family'] = 'Times New Roman'
fontsize = 20 # for smaller components



# ----------------------------- TripAdvisor Understanding ------------------------ #
dir_name = 'data/hotelrec/outputs_all_2019/'
dir_names = ['data/hotelrec/outputs_all_2019/', 'data/hotelrec/outputs_business/', 
             'data/hotelrec/outputs_couple/', 'data/hotelrec/outputs_family/']
names = ['All - 2019', 'Business travel', 'Couples travel', 'Family travel']

styles = [
    {'color': 'pink', 'marker': 'o', 'linestyle': '-'},
    {'color': 'orange', 'marker': 's', 'linestyle': '--'},
    {'color': 'lightgreen', 'marker': 'D', 'linestyle': ':'},
    {'color': 'lightblue', 'marker': '^', 'linestyle': '-.'},
]

fig, ax = plt.subplots(figsize=(8, 5))
i = 1
for iter in range(4):
    df = pd.read_csv(f'{dir_names[iter]}cv_df.csv')
    x_vals, f_vals = np.asarray(df.iloc[:, 1:-1]), np.asarray(df.iloc[:, -1])

    mask = np.ones(len(x_vals), dtype=bool)
    for j in range(6):
        if j != i:
            mask &= x_vals[:, j] == 3

    varying_vals = x_vals[mask, i].tolist()
    f_vals_CV = f_vals[mask].tolist()

    s = styles[iter]
    ax.plot(varying_vals, f_vals_CV,
            color=s['color'],
            marker=s['marker'],
            linestyle=s['linestyle'],
            linewidth=2,
            markersize=7,
            label=names[iter])

ax.set_ylabel('Estimated overall rating')
ax.set_xlabel('Service rating')
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('plots/understanding_varying_service.png', dpi=300, bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(figsize=(8, 5))
styles = [
    {'color': 'pink', 'marker': 'o', 'linestyle': '-'},   
    {'color': 'orange', 'marker': 's', 'linestyle': '--'}, 
    {'color': 'lightgreen', 'marker': 'D', 'linestyle': ':'}
]

criteria_names = ['Service', 'Location', 'Value']
for i in [0,1,2]:
    df = pd.read_csv(f'{dir_name}cv_df.csv')
    x_vals, f_vals = np.asarray(df.iloc[:, 1:-1]), np.asarray(df.iloc[:, -1])
    
    mask = np.ones(len(x_vals), dtype=bool)
    for j in range(6):
        if j != i:
            mask &= x_vals[:, j] == 3

    varying_vals = x_vals[mask, i].tolist()
    f_vals_CV = f_vals[mask].tolist()

    s = styles[i]
    ax.plot(varying_vals, f_vals_CV,
            color=s['color'],
            marker=s['marker'],
            linestyle=s['linestyle'],
            linewidth=2,
            markersize=7,
            label=criteria_names[i])

ax.set_ylabel('Estimated overall rating')
ax.set_xlabel('Criteria rating')
ax.legend(fontsize=fontsize)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('plots/understanding_varying_criteria.png', dpi=300, bbox_inches='tight')
plt.close()




# ----------------------------- LLMs Understanding ------------------------ #

dir_names = ['data/LLMs/outputs_gpt/', 'data/LLMs/outputs_llama/', 'data/LLMs/outputs_human/']
names = ['GPT-4o', 'Llama-3.1-70b', 'Human']

styles = [
    {'color': 'pink', 'marker': 'o', 'linestyle': '-'},
    {'color': 'orange', 'marker': 's', 'linestyle': '--'},
    {'color': 'lightgreen', 'marker': 'D', 'linestyle': ':'},
]

criteria_names = ['presentation', 'soundness', 'contribution']
for value in [1,2,3,4]:
    for i in range(3):
        fig, ax = plt.subplots(figsize=(8, 5))
        for iter in range(3):
            df = pd.read_csv(f'{dir_names[iter]}cv_df.csv')
            X_grid = unique_xvals(4, 3)

            x_vals, f_vals = np.asarray(df.iloc[:, 1:-1]), np.asarray(df.iloc[:, -1])
            y_guess = iso_fit_many(X_grid, x_vals, f_vals)
            # x_vals = X_grid
            # f_vals = y_guess

            mask = np.ones(len(x_vals), dtype=bool)
            for j in range(3):
                if j != i:
                    mask &= x_vals[:, j] == value

            varying_vals = x_vals[mask, i].tolist()
            f_vals_CV = f_vals[mask].tolist()

            s = styles[iter]
            ax.plot(varying_vals, f_vals_CV,
                    color=s['color'],
                    marker=s['marker'],
                    linestyle=s['linestyle'],
                    linewidth=2,
                    markersize=7,
                    label=names[iter])

        ax.set_ylabel('Estimated overall rating')
        ax.set_xlabel(f'{criteria_names[i]} rating')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.title(f'Varying {criteria_names[i]} rating with others fixed at {value}')
        plt.tight_layout()
        plt.savefig(f'plots/LLM_comparison/understanding_varying_{criteria_names[i]}_{value}.png', dpi=300, bbox_inches='tight')
        plt.close()



