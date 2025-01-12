import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# Page Configuration
st.set_page_config(
    page_title="eda by sahilnyk",
    layout="centered",
    page_icon="assets/favicon1.png"
)

# Title and Introduction
st.title("Exploratory Data Analysis Basic Operations")
st.write("Upload your dataset so that we can perform basic operations and give you more information about the data.")

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
    st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.write("Data Types:")
    st.write(df.dtypes)
    st.write("Preview:")
    st.write(df.head(10))

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
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

def visualize_data(df):
    st.subheader("Data Visualization")

    # Histograms with KDE (Kernel Density Estimation)
    st.write("Histograms with KDE:")
    for col in df.select_dtypes(include=['int', 'float']).columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # Bar Charts for Categorical Features
    st.write("Bar Charts for Categorical Features:")
    for col in df.select_dtypes(include=['object']).columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        df[col].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Frequency of {col}")
        st.pyplot(fig)

    # Scatter Plot with Plotly
    st.write("Scatter Plot:")
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    if len(numeric_cols) >= 2:
        st.write("Choose the X and Y axis for the scatter plot:")
        x_axis = st.selectbox("Select X axis", numeric_cols)
        y_axis = st.selectbox("Select Y axis", numeric_cols)

        fig = px.scatter(df, x=x_axis, y=y_axis, color=df[x_axis], title=f"Scatter Plot of {x_axis} vs {y_axis}")
        st.plotly_chart(fig)

    # Pairplot (Seaborn) for all numerical columns
    st.write("Pairplot for Numerical Features:")
    pairplot_fig = sns.pairplot(df.select_dtypes(include=['int', 'float']))
    st.pyplot(pairplot_fig)

# Main EDA Workflow
if uploaded_file:
    df = load_dataset(uploaded_file)

    if df is not None:
        df = make_arrow_compatible(df)

        # Display basic information and features
        st.header("basics EDA operations")
        display_basic_info(df)
        display_feature_types(df)  # Display feature types

        # Apply operations
        df = handle_missing_data(df)
        df = remove_duplicates(df)
        detect_outliers(df)
        df = apply_min_max_scaling(df)  # Scaling applies only to numerical features
        display_correlation_heatmap(df)
        visualize_data(df)

        # Allow downloading of the processed file
        st.subheader("Download Processed Dataset")
        processed_file = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=processed_file, file_name="processed_data.csv")
else:
    st.info("Upload the dataset for operations.")
