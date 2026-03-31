import tensorflow as tf
import boto3
import os
from data_loader import download_data_from_s3, load_data
from sequence_prep import prepare_data
from models import build_lstm, build_gru
from config import OUTPUT_MODEL_DIR, S3_MODEL_BUCKET, S3_MODEL_KEY

def upload_model_to_s3(local_dir):
    s3 = boto3.client("s3")
    for root, _, files in os.walk(local_dir):
        for f in files:
            path = os.path.join(root, f)
            key = S3_MODEL_KEY + path.replace(local_dir, "")
            s3.upload_file(path, S3_MODEL_BUCKET, key)
    print("[INFO] Uploaded best model to S3.")

def train_models():

    # 1. Load data
    download_data_from_s3()
    df = load_data()

    # 2. Prepare sequences
    X, y, scaler = prepare_data(df)

    # 3. Split data
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # 4. Build models
    lstm = build_lstm()
    gru = build_gru()

    # 5. Train both
    print("[INFO] Training LSTM model...")
    lstm_hist = lstm.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    print("[INFO] Training GRU model...")
    gru_hist = gru.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # 6. Compare metrics
    lstm_val_mae = lstm_hist.history["val_mae"][-1]
    gru_val_mae = gru_hist.history["val_mae"][-1]

    best_model = lstm if lstm_val_mae < gru_val_mae else gru
    best_name = "LSTM" if lstm_val_mae < gru_val_mae else "GRU"

    print(f"[INFO] Best model: {best_name}")

    # 7. Save best model
    if not os.path.exists(OUTPUT_MODEL_DIR):
        os.makedirs(OUTPUT_MODEL_DIR)

    best_model.save(OUTPUT_MODEL_DIR)
    print("[INFO] Saved best model locally.")

    # 8. Upload model to S3
    upload_model_to_s3(OUTPUT_MODEL_DIR)

    print("[INFO] Finished training + export pipeline.")

if __name__ == "__main__":
    train_models()
