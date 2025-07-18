import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as stats


# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/delivery_data.csv", sep=";")
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return pd.DataFrame()

# ---------------------------
# EDA Page
# ---------------------------
def show_eda():
    st.title("🔍 Exploratory Data Analysis")
    df = load_data()

    if df.empty:
        st.warning("❌ Dataset is empty after loading. Please check the CSV or path.")
        return

    # Section 1 - Preview
    st.header("📋 Dataset Preview")
    st.dataframe(df.head())
    st.divider()

    # Section 2 - Numerical Feature Distribution
    st.header("📈 Numerical Feature Distribution")
    numerical_cols = [
        'Delivery_Time_min',
        'Preparation_Time_min',
        'Distance_km',
        'Courier_Experience_yrs'
    ]

    # Perbaikan nilai desimal yang pakai koma ke titik, lalu ubah ke float
    df['Courier_Experience_yrs'] = (
        df['Courier_Experience_yrs']
        .astype(str)
        .str.replace(',', '.', regex=False)  # ganti koma ke titik
        .astype(float)                       # konversi ke float
    )

    if numerical_cols:
        selected_num = st.selectbox("Select a numerical column:", numerical_cols)
        fig = px.histogram(
            df, 
            x=selected_num, 
            nbins=30, 
            marginal="box", 
            title=f"Distribution of {selected_num}"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("🔎 No numerical columns found.")



    st.subheader("📊 Categorical Feature Analysis")

    cat_cols = [
        'Weather',
        'Traffic_Level',
        'Vehicle_Type',
        'Time_of_Day'
    ]

    selected_cat = st.selectbox("🔎 Pilih kolom kategorikal untuk dieksplorasi:", cat_cols)

    # Plot histogram interaktif
    fig = px.histogram(
        df,
        x=selected_cat,
        color_discrete_sequence=["#95DCE2"],
        text_auto=True,
        category_orders={selected_cat: sorted(df[selected_cat].unique())},
        labels={selected_cat: selected_cat},
    )
    fig.update_layout(
        title=f"Distribusi dari '{selected_cat}'",
        xaxis_title=selected_cat,
        yaxis_title="Jumlah",
        bargap=0.15
    )
    fig.update_traces(marker_line_width=1, opacity=0.85)

    st.plotly_chart(fig, use_container_width=True)


    st.header("🤔 Insightful Questions")

    # Section 3 - Distribution of Delivery Time
    st.subheader("1️⃣ Deliveries by Weather and Traffic Conditions")

    if 'Weather' in df.columns and 'Traffic_Level' in df.columns:
        delivery_counts = df.groupby(['Weather', 'Traffic_Level']).size().reset_index(name='Count')
        fig = px.bar(
            delivery_counts,
            x='Weather',
            y='Count',
            color='Traffic_Level',
            barmode='stack',
            title="Delivery Counts by Weather and Traffic Conditions"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Weather or Traffic conditions column not found.")


    # Section 4 - Vehicle Type vs Delivery Time
    st.subheader("2️⃣ Vehicle Type vs Delivery Time")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='Vehicle_Type', y='Delivery_Time_min', ax=ax)
    ax.set_xlabel("Vehicle Type")
    ax.set_ylabel("Delivery Time (minutes)")
    st.pyplot(fig)
    st.divider()

    # Section 5 - Weather & Traffic Effects
    st.subheader("3️⃣ Weather & Traffic Effects")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Weather Conditions")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Weather', y='Delivery_Time_min', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

    with col2:
        st.subheader("Traffic Conditions")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Traffic_Level', y='Delivery_Time_min', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    st.divider()

    # Section 6 - Preparation Time vs Delivery Time
    st.subheader("4️⃣ Preparation Time vs Delivery Time")
    fig = px.scatter(df, x="Preparation_Time_min", y="Delivery_Time_min", color="Vehicle_Type")
    st.plotly_chart(fig)
    st.divider()

    # Section 7 - Delivery Time by Time of Day
    st.subheader("5️⃣ Delivery Time by Time of Day")
    if "Time_of_Day" in df.columns:
        time_of_day = st.selectbox("Select Time Segment:", df['Time_of_Day'].unique())
        filtered_df = df[df['Time_of_Day'] == time_of_day]
        fig, ax = plt.subplots()
        sns.histplot(filtered_df['Delivery_Time_min'], kde=True, bins=20, ax=ax)
        ax.set_title(f"Delivery Time during {time_of_day}")
        st.pyplot(fig)
    else:
        st.info("🕒 Column 'Time_of_Day' not available.")
    st.divider()
