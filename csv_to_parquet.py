import pandas as pd

df = pd.read_csv('train.csv', encoding='utf-8')
df.to_parquet('train.parquet', compression='gzip')