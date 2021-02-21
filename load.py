import pandas as pd
import json

df = pd.read_csv('v4_atomic_all.csv' ,index_col=0)
df.iloc[:, :9] = df.iloc[:, :9].apply(lambda col: col.apply(json.loads))
