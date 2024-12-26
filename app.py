import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io

# Define the CSS string for Source Code font and Footer Styling
font_css = """
<style>
body {
    font-family: 'Courier New', monospace;  /* Source code font */
}

.stTextInput, .stTextArea, .stNumberInput, .stSelectbox, .stRadio, .stSlider, .stButton, .stDownloadButton {
    background-color: #333333;
    color: #FFFFFF;
    border: 1px solid #444444;
    font-family: 'Courier New', monospace;  /* Source code font */
}

.stMarkdown {
    color: #FFFFFF;
    font-family: 'Courier New', monospace;  /* Source code font */
}

.stPlotlyChart {
    background-color: #121212;
}

.stTitle, .stSubheader, .stHeader, .stText {
    font-family: 'Courier New', monospace;  /* Source code font */
}

/* Footer Styling */
.footer {
    font-family: 'Courier New', monospace;
    font-size: 19px;
    color: #FFFFFF;
    text-align: center;
    padding: 20px;
    # background-color: #333333;
    border-radius: 10px;
    margin-top: 40px;
}

.footer a {
    color: #ff6347;
    text-decoration: none;
}

.footer a:hover {
    text-decoration: underline;
}
</style>
"""

# Apply the CSS
st.markdown(font_css, unsafe_allow_html=True)

# Title and Introduction
st.title("Automated EDA Tool:fire:")
st.write("Upload your dataset and let this tool perform a comprehensive Exploratory Data Analysis (EDA) step by step.")

# File Upload Section
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

# Initialize EDA Report
eda_report = ""

if uploaded_file:
    # Load the dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Convert DataFrame to Arrow-compatible types
    df = df.convert_dtypes()

    # Step 1: Basic Information
    st.write("## Step 1: Basic Dataset Information")
    st.write("### Shape and Structure")
    st.write("**Rows and Columns:**")
    st.write(df.shape)
    eda_report += f"- The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.\n"

    st.write("**Column Data Types:**")
    st.write(df.dtypes)
    eda_report += f"- Column data types:\n{df.dtypes.to_string()}\n"

    st.write("**Dataset Overview:**")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    eda_report += f"- Dataset Info:\n{buffer.getvalue()}\n"

    # Step 2: Null Values
    st.write("## Step 2: Handling Missing Values")
    st.write("Let's identify and handle missing values in the dataset.")
    null_counts = df.isnull().sum()
    st.write("**Missing Values by Column:**")
    st.write(null_counts[null_counts > 0])
    eda_report += f"- Missing Values:\n{null_counts[null_counts > 0].to_string()}\n"

    if null_counts.sum() > 0:
        st.write("There are missing values in the dataset. Removing rows with missing values...")
        df.dropna(inplace=True)
        st.write("Missing values removed.")
        eda_report += "- Missing values have been removed.\n"
    else:
        st.write("No missing values found in the dataset.")
        eda_report += "- No missing values found.\n"

    # Step 3: Mathematical Operations
    st.write("## Step 3: Mathematical Operations")
    st.write("Calculating mean, median, and mode for numeric columns only.")

    numeric_columns = df.select_dtypes(include=["number"])
    if numeric_columns.empty:
        st.write("No numeric columns found for mathematical operations.")
        eda_report += "- No numeric columns for mathematical operations.\n"
    else:
        # Calculate Mean
        st.write("### Mean")
        mean_values = numeric_columns.mean()
        st.write(mean_values)
        eda_report += f"- Mean Values:\n{mean_values.to_string()}\n"

        # Calculate Median
        st.write("### Median")
        median_values = numeric_columns.median()
        st.write(median_values)
        eda_report += f"- Median Values:\n{median_values.to_string()}\n"

        # Calculate Mode
        st.write("### Mode")
        mode_values = numeric_columns.mode().iloc[0]
        st.write(mode_values)
        eda_report += f"- Mode Values:\n{mode_values.to_string()}\n"

    # Step 4: Visualization
    st.write("## Step 4: Visualizing the Data")

    # Box Plot
    st.write("### Box Plots (Outlier Detection)")
    for col in numeric_columns.columns:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)
        eda_report += f"- Box plot generated for column: {col}\n"

    # Histogram
    st.write("### Histogram (Frequency Distribution)")
    for col in numeric_columns.columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
        eda_report += f"- Histogram generated for column: {col}\n"

    # Simplified Scatter Plot
    st.write("### Scatter Plot (Relationships Between Columns)")
    if numeric_columns.shape[1] >= 2:
        x_col = numeric_columns.columns[0]
        y_col = numeric_columns.columns[1]
        st.write(f"### Scatter Plot of {x_col} vs {y_col}")
        fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
        st.plotly_chart(fig)
        eda_report += f"- Scatter plot generated for {x_col} vs {y_col}.\n"
    else:
        st.write("Not enough numeric columns for scatter plot.")
        eda_report += "- Not enough numeric columns for scatter plot.\n"

    # Pie Chart for Categorical Data
    st.write("### Pie Chart (Categorical Data Distribution)")
    categorical_columns = df.select_dtypes(include=["object", "category"])
    if not categorical_columns.empty:
        st.write("### Pie Chart for Categorical Columns")
        for col in categorical_columns.columns:
            pie_data = df[col].value_counts()
            fig, ax = plt.subplots()
            ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")  # Equal aspect ratio ensures that pie chart is circular.
            st.pyplot(fig)
            eda_report += f"- Pie chart generated for column: {col}\n"
    else:
        st.write("No categorical columns available for pie chart.")
        eda_report += "- No categorical columns for pie chart.\n"

    # Step 5: Feature Selection
    st.write("## Step 5: Feature Selection")
    st.write("### Correlation Heatmap")
    if not numeric_columns.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(numeric_columns.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        eda_report += "- Correlation heatmap generated for numeric columns.\n"
    else:
        st.write("No numeric columns available for correlation analysis.")
        eda_report += "- No numeric columns for correlation analysis.\n"

    # Step 6: Skewness and Kurtosis
    st.write("## Step 6: Skewness and Kurtosis")
    for col in numeric_columns.columns:
        skewness = df[col].skew()
        kurtosis = df[col].kurt()
        st.write(f"**{col}:** Skewness = {skewness:.2f}, Kurtosis = {kurtosis:.2f}")
        eda_report += f"- {col}: Skewness = {skewness:.2f}, Kurtosis = {kurtosis:.2f}\n"

    # Documentation Output
    st.write("## EDA Report")
    st.text_area("Generated EDA Report", eda_report, height=400)

    # Downloadable Outputs
    eda_report_file = eda_report.encode("utf-8")
    st.download_button("Download EDA Report", data=eda_report_file, file_name="eda_report.txt", mime="text/plain")

else:
    st.write("Please upload a dataset to start the analysis.")

# Stylish Footer Section with Centered and Source Code Font
st.markdown("""
<div class="footer">
    <p>Created by: <strong>Sahil Nayak</strong></p>
    <p>Visit my <a href="https://sahilnayak.vercel.app" target="_blank">Website</a></p>
</div>
""", unsafe_allow_html=True)
