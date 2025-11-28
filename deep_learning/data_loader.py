import boto3
import pandas as pd
from config import S3_BUCKET, S3_DATA_KEY, LOCAL_DATA_PATH

def download_data_from_s3():
    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, S3_DATA_KEY, LOCAL_DATA_PATH)
    print("[INFO] Downloaded processed dataset from S3.")

def load_data():
    df = pd.read_parquet(LOCAL_DATA_PATH)
    print(f"[INFO] Loaded dataset with shape {df.shape}")
    return df
