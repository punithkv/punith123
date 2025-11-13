import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "spam.csv")

print(pd.read_csv(csv_path, encoding="latin-1").columns.tolist())

df = pd.read_csv(csv_path, encoding="latin-1")[['v1', 'v2']]
