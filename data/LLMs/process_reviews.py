import pandas as pd
import numpy as np
df_2024_gtp_test = pd.read_csv('../data/LLM reviews/ICLR2024.test.gpt-4o.csv')
df_2024_gtp_extended = pd.read_csv('../data/LLM reviews/ICLR2024.extended.gpt-4o.csv')

df_gpt = pd.concat([df_2024_gtp_test, df_2024_gtp_extended])
df_gpt_human = df_gpt[df_gpt['label']==0]
df_gpt = df_gpt[df_gpt['label']==1]  # only keep the "positive" samples where the model is the reviewer

df_2024_llama_test = pd.read_csv('../data/LLM reviews/ICLR2024.test.llama-3.1-70b.csv')
df_2024_llama_extended = pd.read_csv('../data/LLM reviews/ICLR2024.extended.llama-3.1-70b.csv')

df_llama = pd.concat([df_2024_llama_test, df_2024_llama_extended])
df_llama_human = df_llama[df_llama['label']==0]

df_llama = df_llama[df_llama['label']==1]  # only keep the "positive" samples where the model is the reviewer
print(df_llama_human.shape[0]/df_llama.shape[0])

df_all = pd.concat([df_gpt, df_llama])

# see all unique ratings to create a ratings map
print(df_all['rating'].unique())
print(df_gpt_human['soundness'].unique())

# map ratings to integer values
RATING_MAP = {
    "strong reject":                                           1,
    "1: strong reject":                                        1,
    "marginally above the acceptance threshold":               6,
    "6: marginally above the acceptance threshold":            6,
    "marginally below the acceptance threshold":               5,
    "5: marginally below the acceptance threshold":            5,
    "reject, not good enough":                                 3,
    "3: reject, not good enough":                              3,
    "strong accept, should be highlighted at the conference": 10,
    "10: strong accept, should be highlighted at the conference": 10,
    "strong accept":                                          10,
    "accept, good paper":                                      8,
    "8: accept, good paper":                                   8,
    "Accept, good paper":                                      8,
}


def map_rating(raw):
    if pd.isna(raw):
        return None
    return RATING_MAP.get(str(raw).strip())

def map_ratings(series):
    return series.map(RATING_MAP)

CRITERIA_MAP = {
    '3 good': 3,
    '4 excellent': 4,
    '2 fair': 2,
    '1 poor': 1
}

def map_criteria(raw):
    if pd.isna(raw):
        return None
    return CRITERIA_MAP.get(str(raw))

def map_criterias(series):
    return series.map(CRITERIA_MAP)

print(df_gpt_human['soundness'])

print(type(df_gpt_human['soundness']))
print(type(df_gpt['rating']))
df_gpt_human['soundness'] = df_gpt_human['soundness'].apply(map_criteria)
df_gpt_human['presentation'] = df_gpt_human['presentation'].apply(map_criteria)
df_gpt_human['contribution'] = df_gpt_human['contribution'].apply(map_criteria)

df_gpt['overall'] = df_gpt['rating'].apply(map_rating)
df_llama['overall'] = df_llama['rating'].apply(map_rating)
df_gpt_human['overall'] = df_gpt_human['rating'].apply(map_rating)

COLUMNS = ['overall', 'soundness', 'presentation', 'contribution']
print(df_gpt_human[COLUMNS].head())
for df, name in [(df_gpt, 'gpt'), (df_llama, 'llama'), (df_gpt_human, 'gpt_human')]:
    for col in COLUMNS:
        before = df[col].notna().sum()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        after = df[col].notna().sum()
        lost = before - after
        print(f"[{name}] {col}: lost {lost} / {before} ({100 * lost / before:.1f}%)")

    before_rows = len(df)
    after_rows = df[COLUMNS].dropna().shape[0]
    lost_rows = before_rows - after_rows
    df = df[COLUMNS].dropna()
    print(f"[{name}] Row loss from dropping ANY NA: {lost_rows} / {before_rows} ({100 * lost_rows / before_rows:.1f}%)\n")

df_gpt = df_gpt.dropna(subset=COLUMNS)
df_llama = df_llama.dropna(subset=COLUMNS)
df_gpt_human = df_gpt_human.dropna(subset=COLUMNS)
df_gpt[COLUMNS + ['PaperId']].to_csv('../data/LLM reviews/ICLR2024.gpt-4o_reviews.csv', index=False)
df_llama[COLUMNS + ['PaperId']].to_csv('../data/LLM reviews/ICLR2024.llama-3.1-70b_reviews.csv', index=False)
df_gpt_human[COLUMNS + ['PaperId']].to_csv('../data/LLM reviews/ICLR2024_human_reviews.csv', index=False)


print(df_gpt.shape)
print(df_llama.shape)
print(df_gpt_human.shape)
print(len(df_gpt['PaperId'].unique()))
print(len(df_llama['PaperId'].unique()))
print(len(df_gpt_human['PaperId'].unique()))