import json

import pandas as pd

for split in ['trn', 'dev', 'tst']:
    texts = []
    labels = []

    df = pd.read_csv(f'./atomic_data/v4_atomic_{split}.csv', index_col=0)
    df.iloc[:,:9] = df.iloc[:,:9].apply(lambda col: col.apply(json.loads))

    for index, series in df.iterrows():
        for column in df.columns[:9]:
            if series[column]:
                for label in series[column]:                    
                    texts.append(f'{index} {column}')
                    labels.append(label)

    with open(f'./data/{split}_src.txt', 'w') as f:
        f.write('\n'.join(texts) + '\n')

    with open(f'./data/{split}_tgt.txt', 'w') as f:
        f.write('\n'.join(labels) + '\n')
