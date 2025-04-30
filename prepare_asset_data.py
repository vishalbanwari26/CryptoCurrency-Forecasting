import pandas as pd
import numpy as np

df = pd.read_csv("ADABTC.csv")
df.columns = df.columns.str.strip().str.lower()

df = df.rename(columns={
    'unix': 'timestamp',
    'volume btc': 'volume'
})

df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
df['asset_id'] = 101
df['count'] = 1

# Smoothed future return
df['target'] = df['close'].pct_change(periods=3).shift(-3)

# Clean
df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume', 'vwap', 'target'])
df = df.sort_values('timestamp').drop_duplicates('timestamp')

# Split
split = int(0.8 * len(df))
train = df.iloc[:split]
test = df.iloc[split:]

cols = ['timestamp', 'asset_id', 'count', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'target']
train[cols].to_csv("adabtc_train.csv", index=False)
test[cols].to_csv("adabtc_test.csv", index=False)
