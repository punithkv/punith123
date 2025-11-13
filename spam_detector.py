import os
import pandas as pd

# find the CSV next to the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "spam.csv")

# show CSV columns
print(pd.read_csv(csv_path, encoding="latin-1").columns.tolist())

# load correct columns (rename for clarity if needed)
df = pd.read_csv(csv_path, encoding="latin-1")[['v1', 'v2']]
df.columns = ['label', 'text']

# (optional) quick sanity-check print
print("Loaded", len(df), "rows. Example:")
print(df.head(3))
