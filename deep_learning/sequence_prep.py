import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(values, seq_len=24):
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len])
    return np.array(X), np.array(y)

def prepare_data(df, target_col="consumption", seq_len=24):
    scaler = MinMaxScaler()
    values = scaler.fit_transform(df[[target_col]]).flatten()

    X, y = create_sequences(values, seq_len)

    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((-1, 1))

    print(f"[INFO] Prepared sequences: X={X.shape}, y={y.shape}")
    return X, y, scaler
