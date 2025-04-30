import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from features import add_features
import matplotlib.pyplot as plt

features = [
    'log_return', 'high_low_spread', 'close_open_spread',
    'rolling_mean_3', 'rolling_std_3',
    'rolling_mean_15', 'rolling_std_15',
    'price_to_vwap', 'return_1', 'return_5',
    'momentum_10',
    'volume_log', 'rolling_vol_15'
]


df = pd.read_csv("adabtc_train.csv")
# Normalize column names
df.columns = df.columns.str.strip().str.lower()

df = add_features(df)
df = df.dropna(subset=features + ['target'])

X = df[features]
y = df['target']

model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X, y)

joblib.dump(model, "model_asset_101.pkl")
print("Saved model: model_asset_101.pkl")

# Feature importance as a dict
importances = model.get_booster().get_score(importance_type='gain')

# Convert to DataFrame
importance_df = pd.DataFrame({
    'feature': list(importances.keys()),
    'importance': list(importances.values())
}).sort_values(by='importance', ascending=False)

print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.title("XGBRegressor Feature Importances (Gain)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
