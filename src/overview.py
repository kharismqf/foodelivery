import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def plot_outlier(df, column):
    # Pastikan kolom bertipe numerik
    data = pd.to_numeric(df[column], errors='coerce').dropna()

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Histogram
    sns.histplot(data, bins=30, kde=True, ax=axes[0], color="#6CC38C")
    axes[0].set_title(f"{column} - Histogram")

    # Q-Q Plot
    res = stats.probplot(data, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_color("#62996e")  # Data points
    axes[1].get_lines()[1].set_color("#FF6C6C")  # Fitted line
    axes[1].set_title(f"{column} - Q-Q Plot")

    # Boxplot
    sns.boxplot(x=data, ax=axes[2], color="#6CC38C")
    axes[2].set_title(f"{column} - Boxplot")

    st.pyplot(fig)
    plt.clf()


def show_overview():
    st.markdown("<h1 style='text-align: center; color: #fffff;'>🚚 Food Delivery Dataset</h1>", unsafe_allow_html=True)
    st.image("images/image1.png", use_container_width=True)
    st.markdown("""
        <div style="background-color: #c8ae57; padding: 15px; border-radius: 10px; font-size: 16px; text-align: justify;">
        This regression-based project aims to predict the total delivery time (in minutes) based on a set of contextual, operational, and environmental features. 
        With better delivery time prediction, platforms can improve customer trust, plan routes effectively, and reduce inefficiencies in high-demand scenarios.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📦 Dataset Preview")
    st.markdown("Below is a quick glimpse into the dataset used in this project:")

    df = pd.read_csv("data/delivery_data.csv", sep=";")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    st.markdown("### 📋 Feature Summary")
    feature_summary = pd.DataFrame({
        "Name": [
            "Order_ID", "Distance_km", "Weather", "Traffic_Level", "Time_of_Day",
            "Vehicle_Type", "Preparation_Time_min", "Courier_Experience_yrs", "Delivery_Time_min"
        ],
        "Description": [
            "Unique identifier for each order",
            "Delivery distance in kilometers",
            "Weather conditions (Clear, Rainy, Snowy, Foggy, Windy)",
            "Traffic intensity (Low, Medium, High)",
            "Time slot of the day (Morning, Afternoon, Evening, Night)",
            "Type of vehicle used (Bike, Scooter, Car)",
            "Time taken by the restaurant to prepare the order",
            "Years of experience of the courier",
            "Total delivery time in minutes (target variable)"
        ]
    })
    st.dataframe(feature_summary, use_container_width=True)

    st.markdown("---")

    st.subheader("📈 Initial Data Overview")

    # Get basic metrics
    total_rows = df.shape[0]
    total_cols = df.shape[1]
    missing_vals = int(df.isnull().sum().sum())

    # Styling with HTML boundaries
    st.markdown("""
    <style>
    .metric-box {
        background-color: #e6ca70;
        padding: 15px;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        text-align: center;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        gap: 10px;
    }
    .metric-divider {
        width: 1px;
        background-color: #e6ca70;
        margin: 0 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
    <div class="metric-box">
        <h5>📊 Total Rows</h5>
        <p style='font-size: 20px;'>{total_rows}</p>
    </div>
    <div class="metric-box">
        <h5>📐 Total Columns</h5>
        <p style='font-size: 20px;'>{total_cols}</p>
    </div>
    <div class="metric-box">
        <h5>⚠️ Missing Values</h5>
        <p style='font-size: 20px;'>{missing_vals}</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("---")

    with st.expander("🧹 Data Preprocessing Summary"):
        st.subheader("🔎 1. Missing Values")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        st.dataframe(missing_data)

        st.markdown("""
            <div style="background-color: #E0F5E8; padding: 12px; border-radius: 8px; font-size: 15px; text-align: justify;">
            Some columns still have missing values. Below are the details:
            <ul>
                <li><b>Weather</b>: 30 missing values</li>
                <li><b>Traffic_Level</b>: 30 missing values</li>
                <li><b>Time_of_Day</b>: 30 missing values</li>
                <li><b>Courier_Experience_yrs</b>: 30 missing values</li>
            </ul>
            Detected missing values in 4 features, which will be imputed before feeding into the model. These will be handled using appropriate imputation methods before modeling (using mode).
            </div>
        """, unsafe_allow_html=True)


        st.subheader("✅ 2. Duplicate Check")
        st.write("No duplicate rows found in the dataset.")

        st.subheader("📉 3. Outlier Inspection")
        st.markdown("Numerical columns were examined for outliers, but no significant outliers were found. You can explore them below:")
    
    with st.expander("🧪 Show Outlier Visualization (Distance)"):
        df["Distance_km"] = df["Distance_km"].str.replace(",", ".").astype(float)
        plot_outlier(df, "Distance_km")

    with st.expander("🧪 Show Outlier Visualization (Delivery Time)"):
        plot_outlier(df, "Delivery_Time_min")



