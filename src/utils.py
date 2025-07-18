# src/utils.py
import joblib
import numpy as np

# Load Huber model pipeline
def load_model(model_path='models/huber_pipeline.pkl'):
    return joblib.load(model_path)

# Optional: if you don't log-transform target, you can skip this
def inverse_transform_target(y_log):
    return np.expm1(y_log)

# RMSE calculation
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))