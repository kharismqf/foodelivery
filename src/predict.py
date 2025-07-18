# src/predict.py

import streamlit as st
import pandas as pd
import joblib

def show_predict_page():
    st.title("🚦Food Delivery Time Prediction")
    st.image("images/image3.jpg", use_container_width=True)

    # Input dari user
    Distance_km = st.slider("Distance (km)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
    Preparation_Time_min = st.slider("Preparation Time (min)", min_value=0, max_value=120, value=10, step=1)
    Courier_Experience_yrs = st.slider("Courier Experience (years)", min_value=0, max_value=20, value=2, step=1)

    Weather = st.selectbox("Weather", ["Sunny", "Rainy", "Stormy", "Clear", "Cloudy", "Fog"])
    Traffic_Level = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
    Time_of_Day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
    Vehicle_Type = st.selectbox("Vehicle", ["Bike", "Scooter", "Car"])

    input_df = pd.DataFrame({
        "Distance_km": [Distance_km],
        "Weather": [Weather],  # ✅ Perbaikan di sini
        "Traffic_Level": [Traffic_Level],
        "Time_of_Day": [Time_of_Day],
        "Vehicle_Type": [Vehicle_Type],
        "Preparation_Time_min": [Preparation_Time_min],
        "Courier_Experience_yrs": [Courier_Experience_yrs]
    })


    if st.button("Predict"):
        pipeline = joblib.load("models/huber_pipeline.pkl")
        prediction = pipeline.predict(input_df)[0]
        st.success(f"Estimated Delivery Time: {prediction:.2f} minutes")
