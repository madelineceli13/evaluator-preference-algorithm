
import pandas as pd


out_path = '../data/hotelrec_data/final_reviews_full.csv'

# If ANY of these has a value, drop the row (keep only where all are empty)
drop_if_present = [
    'check in / front desk', 'business service (e.g., internet access)', 'ur_question.prompt.11', 'userrating.prompt.190', 'userrating.prompt.46', 'userrating.prompt.48'
]

# Every one of these must be filled, or the row is dropped
require_filled = [
    'rating',
    'service',
    'location',
    'value',
    'cleanliness',
    'sleep quality',
    'rooms',
]

# Fixed output columns so appended chunks stay aligned in the CSV
output_cols = ['rating', 'title', 'text', 'service', 'location',
               'value', 'cleanliness', 'sleep quality', 'rooms', 'date', 'year']

first = True
total = 0

file_indices = range(1, 52)

# ---------- PASS 1: build the union of all columns ----------
import pyarrow.parquet as pq

all_cols = []
for i in file_indices:
    path = f'../data/hotelrec_data/reviews_part_{i:04d}.parquet'
    # read_schema reads only metadata, not the row data — very fast
    schema_cols = pq.read_schema(path).names
    for c in schema_cols:
        if c not in all_cols:
            all_cols.append(c)   # preserve first-seen order, no duplicates

print(all_cols)

for i in range(1, 52):
    path = f'../data/hotelrec_data/reviews_part_{i:04d}.parquet'
    df = pd.read_parquet(path)

    # --- Year filter ---
    df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
    df = df[df['year'] >= 2014]

    # --- Drop rows where any "should be empty" column is filled ---
    for col in drop_if_present:
        if col in df.columns:          # column may be absent in some files
            df = df[df[col].isna()]

    # --- Keep only rows where all required columns are filled ---
    df = df.dropna(subset=require_filled)

    # --- Force a consistent column set/order before writing ---
    for col in output_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df_part = df[output_cols]

    df_part.to_csv(
        out_path,
        mode='w' if first else 'a',
        header=first,
        index=False,
    )
    first = False
    total += len(df_part)
    print(f"{path}: wrote {df_part.shape[0]:,} rows (running total {total:,})")

print(f"\nDone. {total:,} rows written to {out_path}")