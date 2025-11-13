import os
import pandas as pd

# find the CSV next to the script, regardless of current working dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "spam.csv")

df = pd.read_csv(r"C:\Users\punit\punith123\spam.csv", encoding="latin-1")[['v1', 'v2']]

