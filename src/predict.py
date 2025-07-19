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

    Weather = st.selectbox("Weather", ["Clear", "Windy", "Rainy", "Snowy", "Foggy"])
    Traffic_Level = st.selectbox("Traffic", ["Low", "Medium", "High"])
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

    # 🔍 Show Input Summary
    st.markdown("### 🧾 Input Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**📏 Distance (km):** {Distance_km}")
        st.markdown(f"**⏲️ Preparation Time (min):** {Preparation_Time_min}")
        st.markdown(f"**🚚 Courier Experience (yrs):** {Courier_Experience_yrs}")
        st.markdown(f"**☁️ Weather:** {Weather}")

    with col2:
        st.markdown(f"**🚦 Traffic Level:** {Traffic_Level}")
        st.markdown(f"**🕰️ Time of Day:** {Time_of_Day}")
        st.markdown(f"**🛵 Vehicle Type:** {Vehicle_Type}")

    st.markdown("---")
    st.info("📌 *Note: The prediction is based on historical patterns using Huber Regression.*")

