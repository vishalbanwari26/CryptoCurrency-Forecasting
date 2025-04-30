import numpy as np

def add_features(df, include_target=True):
    df['log_return'] = np.log(df['close'] / df['open'])
    df['high_low_spread'] = df['high'] - df['low']
    df['close_open_spread'] = df['close'] - df['open']
    df['rolling_mean_3'] = df.groupby('asset_id')['close'].transform(lambda x: x.rolling(3).mean())
    df['rolling_std_3'] = df.groupby('asset_id')['close'].transform(lambda x: x.rolling(3).std())
    df['rolling_mean_15'] = df.groupby('asset_id')['close'].transform(lambda x: x.rolling(15).mean())
    df['rolling_std_15'] = df.groupby('asset_id')['close'].transform(lambda x: x.rolling(15).std())
    df['price_to_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    df['return_1'] = df.groupby('asset_id')['close'].transform(lambda x: x.pct_change(1))
    df['return_5'] = df.groupby('asset_id')['close'].transform(lambda x: x.pct_change(5))
    df['momentum_10'] = df.groupby('asset_id')['close'].transform(lambda x: x.diff(10))
    
    #if include_target and 'target' in df.columns:
    #    df['lag_target_1'] = df.groupby('asset_id')['target'].shift(1)
    #else:
        # Drop it completely in live mode
    #    pass

    df['volume_log'] = np.log1p(df['volume'])
    df['rolling_vol_15'] = df.groupby('asset_id')['volume_log'].transform(lambda x: x.rolling(15).mean())

    return df
