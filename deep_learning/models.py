import tensorflow as tf
from tensorflow.keras import layers, Sequential

def build_lstm(seq_len=24):
    model = Sequential([
        layers.LSTM(32, return_sequences=False, input_shape=(seq_len, 1)),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])
    return model

def build_gru(seq_len=24):
    model = Sequential([
        layers.GRU(32, return_sequences=False, input_shape=(seq_len, 1)),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])
    return model
