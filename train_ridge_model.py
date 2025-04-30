import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from features import add_features

features = [
    'log_return', 'high_low_spread', 'close_open_spread',
    'rolling_mean_3', 'rolling_std_3',
    'rolling_mean_15', 'rolling_std_15',
    'price_to_vwap', 'return_1', 'return_5',
    'momentum_10', 'lag_target_1',
    'volume_log', 'rolling_vol_15'
]

df = pd.read_csv("adabtc_train.csv")
df = add_features(df)
df = df.dropna(subset=features + ['target'])

X = df[features]
y = df['target']

model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
model.fit(X, y)

joblib.dump(model, "ridge_model_asset_101.pkl")
print("Saved Ridge model")
