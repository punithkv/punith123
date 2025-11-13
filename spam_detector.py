import os
import pandas as pd
import pandas as pd
print(pd.read_csv(csv_path, encoding="latin-1").columns.tolist())


# find the CSV next to the script, regardless of current working dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "spam.csv")

df = pd.read_csv(r"C:\Users\punit\punith123\spam.csv", encoding="latin-1")[['v1', 'v2']]

