import json
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1] ,index_col=0)
df.iloc[:, :9] = df.iloc[:, :9].apply(lambda col: col.apply(json.loads))

df = df[df['oReact'].apply(lambda x: len(x) > 0)]
df = df.loc[:, ['oReact', 'xIntent','xReact', 'prefix', 'split']]

df = df.apply(lambda col: col.apply(json.dumps))
df.to_csv('mentalstate.csv')
