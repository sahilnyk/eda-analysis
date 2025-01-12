import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# Page Configuration
st.set_page_config(
    page_title="EDA Tool by Sahil Nayak",
    page_icon=":bar_chart:",
    layout="centered"
)

# Title and Introduction
st.title("Exploratory Data Analysis Tool")
st.write("Upload your dataset and perform step-by-step EDA operations.")

# File Upload Section
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

# Helper Functions
def load_dataset(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def display_basic_info(df):
    st.subheader("Basic Information")
    st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.write("**Data Types:")
    st.write(df.dtypes)
    st.write("**Preview:")
    st.write(df.head())

def handle_missing_data(df):
    st.subheader("Handle Missing Data")
    missing_data = df.isnull().sum()
    st.write("**Columns with Missing Values:**")
    st.write(missing_data[missing_data > 0])

    if st.button("Remove Missing Data"):
        df.dropna(inplace=True)
        st.success("Removed rows with missing data.")
    return df

def remove_duplicates(df):
    st.subheader("Remove Duplicates")
    duplicates = df.duplicated().sum()
    st.write(f"**Duplicate Rows:** {duplicates}")

    if st.button("Remove Duplicate Rows"):
        df.drop_duplicates(inplace=True)
        st.success("Duplicate rows removed.")
    return df

def detect_outliers(df):
    st.subheader("Outlier Detection")
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        st.write(f"**{col}:**")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)

def apply_scaling(df):
    st.subheader("Apply Min-Max Scaling")
    numeric_cols = df.select_dtypes(include=['number']).columns

    if st.button("Apply Scaling"):
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        st.success("Scaling applied.")
    return df

def display_correlation_heatmap(df):
    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=['number'])

    if not numeric_cols.empty:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

def visualize_data(df):
    st.subheader("Data Visualization")

    st.write("**Histograms:**")
    for col in df.select_dtypes(include=['number']).columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    st.write("**Bar Charts:**")
    for col in df.select_dtypes(include=['object', 'category']).columns:
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    st.write("**Scatter Plot:**")
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
        st.plotly_chart(fig)

# Main EDA Workflow
if uploaded_file:
    df = load_dataset(uploaded_file)

    st.header("Step-by-Step Analysis")
    display_basic_info(df)
    df = handle_missing_data(df)
    df = remove_duplicates(df)
    detect_outliers(df)
    df = apply_scaling(df)
    display_correlation_heatmap(df)
    visualize_data(df)

    st.subheader("Download Processed Dataset")
    processed_file = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=processed_file, file_name="processed_data.csv")
else:
    st.info("Upload a file to start EDA.")
