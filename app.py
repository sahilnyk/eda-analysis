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

# Custom CSS for light/dark mode and M PLUS Code font
st.markdown("""
    <style>
        /* Load Google font - M PLUS Code */
        @import url('https://fonts.googleapis.com/css2?family=M+PLUS+Code+Latin:wght@300;400;500&display=swap');

        /* Set the body font to M PLUS Code */
        body {
            font-family: 'M PLUS Code Latin', monospace;
        }

        /* Dark Mode Styling */
        .dark-mode {
            background-color: #121212;
            color: white;
        }

        /* Light Mode Styling */
        .light-mode {
            background-color: #f5f5f5;
            color: black;
        }

        /* Header Styling */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'M PLUS Code Latin', monospace;
        }

    </style>
""", unsafe_allow_html=True)

# Title and Introduction
st.title("Exploratory Data Analysis Tool")
st.write("Upload your dataset and perform step-by-step basic EDA operations.")

# File Upload Section
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

# Helper Functions
def load_dataset(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file, on_bad_lines='skip', encoding_errors='ignore')
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"Failed to load the dataset: {e}")
        return None

def make_arrow_compatible(df):
    """Converts object columns to string to make the DataFrame Arrow-compatible."""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    return df

def display_basic_info(df):
    st.subheader("Basic Information")
    st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    st.write("Data Types:")
    st.write(df.dtypes)
    st.write("Preview:")
    st.write(df.head())

def display_feature_types(df):
    st.subheader("Feature Types")
    categorical_features = df.select_dtypes(include=['object']).columns
    numerical_features = df.select_dtypes(include=['int', 'float']).columns

    st.write("Categorical Features:")
    st.write(categorical_features)

    st.write("Numerical Features:")
    st.write(numerical_features)

def handle_missing_data(df):
    st.subheader("Handle Missing Data")
    missing_data = df.isnull().sum()
    st.write("Columns with Missing Values:")
    st.write(missing_data[missing_data > 0])

    if st.button("Remove Missing Data"):
        df.dropna(inplace=True)
        st.success("Removed rows with missing data.")
    return df

def remove_duplicates(df):
    st.subheader("Remove Duplicates")
    duplicates = df.duplicated().sum()
    st.write(f"Duplicate Rows: {duplicates}")

    if st.button("Remove Duplicate Rows"):
        df.drop_duplicates(inplace=True)
        st.success("Duplicate rows removed.")
    return df

def detect_outliers(df):
    st.subheader("Outlier Detection")
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    for col in numeric_cols:
        st.write(f"{col}:")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)

def apply_min_max_scaling(df):
    st.subheader("Apply Min-Max Scaling")
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns

    if st.button("Apply Scaling"):
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        st.success("Scaling applied.")
    return df

def display_correlation_heatmap(df):
    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=['int', 'float'])

    if not numeric_cols.empty:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

def visualize_data(df):
    st.subheader("Data Visualization")

    st.write("Histograms:")
    for col in df.select_dtypes(include=['int', 'float']).columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    st.write("Bar Charts:")
    for col in df.select_dtypes(include=['object']).columns:
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    st.write("Scatter Plot:")
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
        st.plotly_chart(fig)

# Main EDA Workflow
if uploaded_file:
    df = load_dataset(uploaded_file)

    if df is not None:
        df = make_arrow_compatible(df)
        st.header("Step-by-Step Analysis")
        display_basic_info(df)
        display_feature_types(df)  # Display feature types

        # Apply operations in order for both numerical and categorical features
        df = handle_missing_data(df)
        df = remove_duplicates(df)
        detect_outliers(df)
        df = apply_min_max_scaling(df)  # Scaling applies only to numerical features
        display_correlation_heatmap(df)
        visualize_data(df)

        st.subheader("Download Processed Dataset")
        processed_file = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=processed_file, file_name="processed_data.csv")
else:
    st.info("Upload a file to start EDA.")
