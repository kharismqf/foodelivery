# 🚚 Food Delivery Time Prediction

This project aims to build a predictive model that estimates food delivery time based on historical order and operational data. The objective is to help businesses set accurate expectations, allocate resources efficiently, and improve customer satisfaction.

---

## 📊 Project Overview

Despite having operational data, the company lacks a system that can predict delivery time based on real-world conditions such as:

- Distance (km)
- Food preparation time (min)
- Courier’s delivery experience (years)
- Traffic and weather conditions

This project addresses that gap using a combination of Exploratory Data Analysis (EDA) and Machine Learning (ML) modeling.

---

## 📁 Folder Structure

food_delivery_model/
│
├── data/
│ └── delivery_data.csv # Clean dataset used for modeling
│
├── src/
│ ├── train_model.py # Script for training Ridge & Huber Regression
│ ├── predict.py # Script for making predictions
│ ├── utils.py # Utility functions for preprocessing
│ └── config.toml # Streamlit configuration (theme, etc.)
│
├── models/
│ └── model_huber.pkl # Saved ML model
│
├── app.py # Streamlit web application
├── requirements.txt # Python dependencies
└── README.md # Project documentation
