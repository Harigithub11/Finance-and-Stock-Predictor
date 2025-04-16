    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    def preprocess_xgboost_lstm(data_path):
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler

        df = pd.read_csv(data_path)

        # Rename columns for consistency
        df.rename(columns={'Price': 'Close', 'Vol.': 'Volume'}, inplace=True)

        # Drop 'Change %' if it exists
        if 'Change %' in df.columns:
            df.drop(columns=['Change %'], inplace=True)

        # Parse Volume
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

        df['Volume'] = df['Volume'].apply(parse_volume)

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index('Date', inplace=True)
        X_xgb = df[['Open', 'High', 'Low']].values

        df['Return'] = df['Close'].pct_change()
        df.dropna(inplace=True)

        # XGBoost features
        X = df[['Open', 'High', 'Low', 'Volume']].values
        y = df['Close'].values

        # LSTM features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_close = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

        def create_lstm_dataset(dataset, time_step=10):
            X_lstm, y_lstm = [], []
            for i in range(len(dataset) - time_step):
                X_lstm.append(dataset[i:i + time_step, 0])
                y_lstm.append(dataset[i + time_step, 0])
            return np.array(X_lstm), np.array(y_lstm)

        X_lstm, y_lstm = create_lstm_dataset(scaled_close)
        X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

        return X, y, X_lstm, y_lstm, scaler
