import requests
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
from features import add_features
from datetime import datetime

# --- CONFIG ---
MODEL_PATH = "model_asset_101.pkl"
SYMBOL = "ADABTC"
INTERVAL = "1m"
FETCH_LIMIT = 150  # must be ≥ max rolling window (e.g., 15 + buffer)
WINDOW = 30  # how many points to show on plot

# Load trained model
model = joblib.load(MODEL_PATH)

# Set up live plot
plt.ion()
fig, ax = plt.subplots(figsize=(14, 6))
predictions, prices, timestamps = [], [], []

def fetch_latest_klines(symbol=SYMBOL, interval=INTERVAL, limit=FETCH_LIMIT):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    
    # Build DataFrame
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.columns = df.columns.str.lower()
    return df

def prepare_features_for_live(df):
    df = df.copy()
    df['asset_id'] = 101
    df['count'] = 1
    df['vwap'] = (df['high'] + df['low'] + df['close']) / 3

    df = add_features(df, include_target=False)

    df = df.dropna()
    return df

# --- MAIN LOOP ---
while True:
    try:
        print("\nFetching new data...")
        df_raw = fetch_latest_klines(limit=FETCH_LIMIT)
        print("📦 Raw Binance rows:", len(df_raw))
        print(df_raw.tail(2))

        df_live = prepare_features_for_live(df_raw)

        if df_live.empty:
            print("Still not enough valid rows for prediction.")
            time.sleep(15)
            continue

        latest = df_live.iloc[-1:]
        X_latest = latest[[
            'log_return', 'high_low_spread', 'close_open_spread',
            'rolling_mean_3', 'rolling_std_3', 'rolling_mean_15', 'rolling_std_15',
            'price_to_vwap', 'return_1', 'return_5', 'momentum_10',
            'volume_log', 'rolling_vol_15'  # 🧼 removed lag_target_1
        ]]


        y_pred = model.predict(X_latest)[0]
        close_price = latest['close'].values[0]
        ts = latest['timestamp'].iloc[0].to_pydatetime()

        predictions.append(y_pred)
        prices.append(close_price)
        timestamps.append(ts)

        # --- Plot update ---
        ax.clear()
        ax.plot(timestamps[-WINDOW:], prices[-WINDOW:], label="Actual Close", linewidth=2)
        ax.plot(timestamps[-WINDOW:], predictions[-WINDOW:], label="Predicted Return", linestyle='--')
        ax.set_title(f"Live ADABTC Prediction: {close_price:.8f} | Last Pred: {y_pred:.8f}")
        ax.set_ylabel("Price / Predicted Return")
        ax.set_xlabel("Time")
        ax.legend()
        ax.grid(True)

        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate()

        plt.pause(1)

        time.sleep(60)

    except Exception as e:
        print("Error:", str(e))
        time.sleep(15)
