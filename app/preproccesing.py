import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
import datetime

def fetch_stock_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            raise ValueError(f"No data fetched for {ticker} from {start_date} to {end_date}")
    except Exception as e:
        raise ValueError(f"Error fetching data for {ticker}: {str(e)}")

    print(f"DataFrame columns after fetching: {df.columns.tolist()}")
    print(f"DataFrame shape: {df.shape}")

    if 'Close' not in df.columns and 'close' in df.columns:
        df['Close'] = df['close']
    elif 'Close' not in df.columns:
        raise ValueError(f"'Close' column missing in data for {ticker}")

    df['Ticker'] = ticker

    df['SMA_10'] = df['Close'].rolling(window=10).mean().shift(1)
    df['SMA_50'] = df['Close'].rolling(window=50).mean().shift(1)

    # RSI
    delta = df['Close'].diff().shift(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Drop rows with NaN values
    df.dropna(inplace=True)

    print(f"DataFrame columns after processing: {df.columns.tolist()}")
    print(f"DataFrame shape after dropna: {df.shape}")

    if len(df) < 50:
        raise ValueError(f"Insufficient data for {ticker} after processing (only {len(df)} rows)")

    required_columns = ['Close', 'SMA_10', 'SMA_50', 'RSI']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in DataFrame for {ticker}: {missing_columns}")

    return df

def prepare_datasets(data_dict, seq_length=20, val_size=0.2, test_size=0.2):
    X, y, tickers = [], [], []
    for ticker, df in data_dict.items():
        values = df[['Close', 'SMA_10', 'SMA_50', 'RSI']].values
        for i in range(len(values) - seq_length):
            X.append(values[i:i+seq_length])
            y.append(values[i+seq_length, 0])
            tickers.append(ticker)

    X = np.array(X)
    y = np.array(y)
    tickers = np.array(tickers)

    print(f"Data for {ticker} - X: {X.shape}, y: {y.shape}, tickers: {tickers.shape}")

    train_idx = int(len(X) * (1 - val_size - test_size))
    val_idx = int(len(X) * (1 - test_size))

    X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
    y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]
    tickers_train, tickers_val, tickers_test = tickers[:train_idx], tickers[train_idx:val_idx], tickers[val_idx:]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Feature scaling
    feature_scaler = RobustScaler()
    X_train = feature_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = feature_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = feature_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Target scaling
    target_scaler = RobustScaler()
    y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    print(f"Final X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"Final y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, target_scaler, tickers_train, tickers_val, tickers_test


def prepare_prediction_data(df, seq_length=20):
    required_columns = ['Close', 'SMA_10', 'SMA_50', 'RSI']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in DataFrame: {missing_columns}")

    print(f"Input DataFrame columns: {df.columns.tolist()}")
    print(f"Input DataFrame shape: {df.shape}")

    values = df[required_columns].values

    X, y = [], []
    for i in range(len(values) - seq_length):
        X.append(values[i:i+seq_length])
        y.append(values[i+seq_length, 0])

    X = np.array(X)
    y = np.array(y)

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    feature_scaler = RobustScaler()
    X_scaled = feature_scaler.fit_transform(
        X.reshape(-1, X.shape[-1])).reshape(X.shape)

    target_scaler = RobustScaler()
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled, y, feature_scaler, target_scaler