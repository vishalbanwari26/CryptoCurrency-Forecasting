import pandas as pd
import matplotlib.pyplot as plt

preds = pd.read_csv("adabtc_predictions.csv")
true = pd.read_csv("adabtc_test.csv")

preds['datetime'] = pd.to_datetime(preds['timestamp'], unit='ms')
true['datetime'] = pd.to_datetime(true['timestamp'], unit='ms')

df = pd.merge(preds, true[['datetime', 'target', 'close']], on='datetime', how='left')

corr = df['prediction'].corr(df['target'])
if corr < 0:
    print(f"lipping prediction (corr = {corr:.3f})")
    df['prediction'] *= -1
else:
    print(f"Using original prediction (corr = {corr:.3f})")

df['signal'] = df['prediction'].apply(lambda x: 1 if x > 0 else -1)
df['strategy_return'] = df['signal'] * df['target']
df['cumulative_return'] = (1 + df['strategy_return']).cumprod()

print(f"Final cumulative return: {df['cumulative_return'].iloc[-1]:.4f}")

plt.figure(figsize=(14, 6))
plt.plot(df['datetime'], df['cumulative_return'], label='Ridge Strategy Return')
plt.axhline(1.0, linestyle='--', color='gray', linewidth=0.8)
plt.title("Ridge Regression Strategy Backtest")
plt.xlabel("Time")
plt.ylabel("Growth from $1")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
plt.scatter(df['prediction'], df['target'], alpha=0.3)
plt.xlabel("Prediction")
plt.ylabel("Actual Target")
plt.title("Prediction vs Actual Return")
plt.grid(True)
plt.show()

