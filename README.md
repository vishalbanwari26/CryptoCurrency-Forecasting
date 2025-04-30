# 🧠 Crypto Forecasting with Traditional Machine Learning

This project builds a short-term return forecasting pipeline for crypto assets using traditional ML models such as Ridge Regression and XGBoost. The focus is on the ADABTC pair at a 1-minute interval, using real-time data from Binance.

I smooth the prediction target using a forward-looking 3-minute return, engineer features using only current/past data, train and evaluate models on historical data, and deploy a live prediction system connected to Binance.

## 🔄 Data Format

I use 1-minute OHLCV candlestick data with the following fields:

- `timestamp`: UNIX timestamp converted to datetime
- `open`, `high`, `low`, `close`: standard OHLC price
- `volume`: base asset volume in that minute
- `vwap`: calculated as `(high + low + close) / 3`
- `target`: smoothed forward return, calculated as `close.pct_change(3).shift(-3)`

## 🧠 Features and Target

To reduce noise, the target is defined as the 3-step forward return:  
`target = close.pct_change(periods=3).shift(-3)`

Features include:

- `log_return`: log(close / open)
- `high_low_spread`: high - low
- `close_open_spread`: close - open
- `rolling_mean_3`, `rolling_mean_15`: rolling mean of close
- `rolling_std_3`, `rolling_std_15`: rolling standard deviation
- `return_1`, `return_5`: past returns
- `momentum_10`: difference in price 10 minutes ago
- `volume_log`: log1p(volume)
- `rolling_vol_15`: rolling average of volume

All features use only current or past data — there is no leakage of future information.

## 🏗️ Project Pipeline

1. **Prepare the dataset with target + features**
    ```bash
    python prepare_asset_data.py
    ```

2. **Train the Ridge regression or XGBoost model**
    ```bash
    python train_model.py
    ```

3. **Evaluate the model's correlation and strategy performance**
    ```bash
    python evaluate_predictions.py
    ```

4. **Start live prediction using Binance API**
    ```bash
    python live_predict_binance.py
    ```


## 📈 Example Output

```
📊 Correlation between prediction and target: 0.860
📈 Final cumulative return: 789029.4036
✅ Live chart updates every 60s using real Binance data
```

## 📦 Requirements

```bash
pip install pandas numpy matplotlib scikit-learn joblib requests
```

## 🚀 Extensions & Ideas

- Add classification mode for up/down predictions
- Include trading signals (long/short threshold)
- Send alerts via Telegram or email
- Switch to Binance WebSocket for tick-level streaming

## 🏁 License

MIT — feel free to adapt and build upon this project.
