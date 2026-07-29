import pandas as pd
import numpy as np

folder_names = ['data/hotelrec/outputs_family/', 'data/hotelrec/outputs_couple/', 'data/hotelrec/outputs_business/', 'data/hotelrec/outputs_all_2019/',
                'data/LLMs/outputs_gpt/', 'data/LLMs/outputs_llama/', 'data/LLMs/outputs_human/',
                'data/ICLR/outputs_2023/', 'data/ICLR/outputs_2024/', 'data/ICLR/outputs_2025/', 'data/NASA/all_evaluators/'
                ]

for name in folder_names:
    df = pd.read_csv(f'{name}CV_summary.csv')
    i = df['val_mse'].idxmin()
    
    # get best lamba
    print(name)
    print(f'optimal $\lambda$ is {df.loc[i, "lambda"]}')

    print()
    
    min_val = df['val_mse'].min()

    # 2. Check if the count of that minimum value is greater than 1
    has_multiple_mins = (df['val_mse'] == min_val).sum() 
    if has_multiple_mins > 1:
        print('Multiple minimums')
        print(has_multiple_mins)

# run isotonic regression and constant function stuff for hotelrec dataset 