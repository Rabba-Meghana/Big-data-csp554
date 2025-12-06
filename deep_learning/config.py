import os

AWS_REGION = "us-east-1"

S3_BUCKET = "your-cleaned-data-bucket"
S3_DATA_KEY = "processed/final_data.parquet"

LOCAL_DATA_PATH = "data/final_data.parquet"

# Output model
OUTPUT_MODEL_DIR = "saved_model/"
S3_MODEL_BUCKET = "your-model-bucket"
S3_MODEL_KEY = "deep_learning/best_model/"
