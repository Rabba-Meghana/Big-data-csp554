# Electricity Consumption Forecasting – Deep Learning Module  
CSP 554 – Big Data Technologies  
**Author:** Chaitanya Datta MAddukuri
**Task:** Individual Deep Learning Component  

---

## 📌 Overview  
This module focuses on developing **deep learning forecasting models** (LSTM and GRU) using hourly electricity consumption data.  
It is part of an end-to-end big data forecasting pipeline using AWS S3, Spark, and SageMaker.

Your tasks completed in this module:
- Load cleaned, processed dataset from S3  
- Create supervised time-series sequences (past 24 hours → next hour)  
- Build small deep learning models (LSTM, GRU)  
- Train both and compare validation metrics  
- Select the best performing model  
- Export model in TensorFlow SavedModel format  
- Upload model to an S3 bucket for deployment  

---

## 📁 Project Structure

deep_learning/
│── config.py # S3 paths, model directory config
│── data_loader.py # Download + load dataset from S3
│── sequence_prep.py # Create 24-hour input sequences
│── models.py # LSTM and GRU model definitions
│── train.py # Full training pipeline
│── requirements.txt # Python dependencies
└── README.md # This file


---

## 🚀 How It Works

### 1️⃣ **Load Dataset from S3**
The script downloads the final processed dataset (Parquet format) into `data/`.

### 2️⃣ **Prepare Sequences**
- Normalize consumption values  
- Use 24-hour history as input  
- Predict next hour consumption  

### 3️⃣ **Build Models**
Two lightweight models:  
- **LSTM (32 units + dropout)**  
- **GRU (32 units + dropout)**  

### 4️⃣ **Train + Compare**
Both models are trained with:
- Validation split  
- MAE, MSE metrics  
- Dropout to reduce overfitting  

The best model is selected based on **validation MAE**.

### 5️⃣ **Save and Upload**
The best model is saved in: saved_model/

Then uploaded automatically to your S3 model bucket.

---

## 🛠️ Installation

### Install dependencies:
```bash
pip install -r requirements.txt

### Configure AWS Credentials:
aws configure

### Run Training:
python train.py

This will:

-Download data from S3
-Train LSTM & GRU
-Pick the best model
-Save to saved_model/
-Upload to S3 for SageMaker deployment

