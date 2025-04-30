import pandas as pd
import joblib
from features import add_features

features = [
    'log_return', 'high_low_spread', 'close_open_spread',
    'rolling_mean_3', 'rolling_std_3',
    'rolling_mean_15', 'rolling_std_15',
    'price_to_vwap', 'return_1', 'return_5',
    'momentum_10', 'lag_target_1',
    'volume_log', 'rolling_vol_15'
]

model = joblib.load("ridge_model_asset_101.pkl")

df = pd.read_csv("adabtc_test.csv")
df = add_features(df)
df[features] = df[features].fillna(method='bfill')

df['prediction'] = model.predict(df[features])
df[['timestamp', 'prediction']].to_csv("adabtc_ridge_predictions.csv", index=False)
print("Saved: adabtc_ridge_predictions.csv")
