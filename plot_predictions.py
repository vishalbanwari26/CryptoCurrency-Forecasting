import pandas as pd
import matplotlib.pyplot as plt

preds = pd.read_csv("adabtc_ridge_predictions.csv")
price = pd.read_csv("adabtc_test.csv")

# Normalize column names
price.columns = price.columns.str.strip().str.lower()

preds['datetime'] = pd.to_datetime(preds['timestamp'], unit='ms')
price['datetime'] = pd.to_datetime(price['timestamp'], unit='ms')

df = pd.merge(preds, price[['datetime', 'close']], on='datetime', how='left')

plt.figure(figsize=(14, 6))
ax1 = plt.gca()
ax1.plot(df['datetime'], df['close'], label='ADABTC Close', color='blue')
ax1.set_ylabel('Price', color='blue')

ax2 = ax1.twinx()
ax2.plot(df['datetime'], df['prediction'], label='Predicted Return', color='red', alpha=0.5)
ax2.set_ylabel('Predicted Return', color='red')

plt.title("ADABTC Price vs Predicted Return")
plt.grid(True)
plt.tight_layout()
plt.show()
