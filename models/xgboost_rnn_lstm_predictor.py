import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def run_stock_prediction(csv_path):
    print("\n📈 Running Stock Prediction using XGBoost + LSTM...\n")

    df = pd.read_csv(csv_path)

    # Rename relevant columns
    df.rename(columns={
        'Price': 'Close',
        'Vol.': 'Volume'
    }, inplace=True)

    # Drop 'Change %' column if it exists (non-numeric)
    if 'Change %' in df.columns:
        df.drop(columns=['Change %'], inplace=True)

    # Function to clean numeric values (remove commas)
    def clean_numeric(val):
        if isinstance(val, str):
            return float(val.replace(',', '').strip())
        return val

    # Clean all numeric columns
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)

    # Clean Volume column (handle 'K', 'M', 'B' if needed)
    def parse_volume(val):
        if isinstance(val, str):
            val = val.replace(',', '').strip()
            if 'K' in val:
                return float(val.replace('K', '')) * 1_000
            elif 'M' in val:
                return float(val.replace('M', '')) * 1_000_000
            elif 'B' in val:
                return float(val.replace('B', '')) * 1_000_000_000
            elif val == '-':
                return np.nan
            return float(val)
        return val

    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].apply(parse_volume)

    # Date parsing and sorting
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date', 'Close'], inplace=True)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    # --- XGBoost ---
    X_xgb = df[['Open', 'High', 'Low']].values
    y_xgb = df['Close'].values
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.2, shuffle=False)

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(X_train_xgb, y_train_xgb)
    xgb_preds = xgb_model.predict(X_test_xgb)

    # --- LSTM ---
    close_prices = df[['Close']].values
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(close_prices)

    def create_dataset(data, time_step=10):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:i+time_step, 0])
            y.append(data[i+time_step, 0])
        return np.array(X), np.array(y)

    time_step = 10
    X_lstm, y_lstm = create_dataset(scaled_close, time_step)
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)
    X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], 1)
    X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], 1)

    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=1, batch_size=1, verbose=0)

    lstm_preds_scaled = lstm_model.predict(X_test_lstm)
    lstm_preds = scaler.inverse_transform(lstm_preds_scaled).flatten()

    # Combine Predictions
    min_len = min(len(xgb_preds), len(lstm_preds))
    combined_preds = (xgb_preds[-min_len:] + lstm_preds[-min_len:]) / 2
    test_dates = df.index[-min_len:]

    print("✅ Combined Predictions:\n")
    for i in range(min_len):
        print(f"Date: {test_dates[i].date()} | Predicted Close Price: {combined_preds[i]:.2f}")

    print(f"\n📊 MSE (XGBoost): {mean_squared_error(y_test_xgb[-min_len:], xgb_preds[-min_len:]):.4f}")
    print(f"📊 MSE (LSTM): {mean_squared_error(scaler.inverse_transform(y_test_lstm[:min_len].reshape(-1, 1)), lstm_preds.reshape(-1, 1)):.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_xgb[-min_len:], label="Actual", color='blue')
    plt.plot(test_dates, combined_preds, label="Predicted (Combined)", color='orange')
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("📈 Actual vs Predicted Close Price (XGBoost + LSTM)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

 