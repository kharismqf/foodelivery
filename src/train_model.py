import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor

# 1. Load Data (path relatif ke src/)
df = pd.read_csv("../data/delivery_data.csv", delimiter=";", decimal=",")

# 2. Drop kolom yang tidak dibutuhkan
df = df.drop(columns=["Order_ID", "Condition_Combo"], errors='ignore')

# 3. Pisahkan fitur dan target
X = df.drop(columns=["Delivery_Time_min"])
y = df["Delivery_Time_min"]

# 4. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Deteksi fitur numerik dan kategorikal secara eksplisit
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 6. Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
])

# 7. Pipeline model
huber_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', HuberRegressor())
])

# 8. Fit model
huber_pipeline.fit(X_train, y_train)

# 9. Simpan pipeline
os.makedirs("../models", exist_ok=True)
joblib.dump(huber_pipeline, "../models/huber_pipeline.pkl")

print("✅ Model saved to ../models/huber_pipeline.pkl")
