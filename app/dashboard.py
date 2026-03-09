import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.predict import InsurancePredictor

# Page Configuration
st.set_page_config(
    page_title="Medical Insurance Price Predictor",
    page_icon="🩺",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("🩺 Medical Insurance Price Optimization")
st.markdown("---")

# Load model locally for fallback, or use API if preferred.
# For simplicity, we'll use the predictor directly here as well.
@st.cache_resource
def get_predictor():
    try:
        return InsurancePredictor(model_path='models/insurance_model_pipeline.joblib')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

predictor = get_predictor()

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the App Mode", ["Simulator", "Batch Prediction", "Data Insights"])

if app_mode == "Simulator":
    st.header("💡 Insurance Cost Simulator")
    st.write("Enter the following details to estimate medical insurance charges.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 100, 30)
        sex = st.selectbox("Sex", ["female", "male"])
        bmi = st.slider("BMI", 15.0, 50.0, 25.0)
        
    with col2:
        children = st.number_input("Number of Children", 0, 10, 0)
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

    if st.button("Calculate Estimated Charges"):
        if predictor:
            input_data = {
                'age': age,
                'sex': sex,
                'bmi': bmi,
                'children': children,
                'smoker': smoker,
                'region': region
            }
            
            prediction = predictor.predict(input_data)[0]
            
            st.success(f"### Estimated Annual Charges: ${prediction:,.2f}")
            
            # Additional logic to show risk level
            if smoker == 'yes' and bmi > 30:
                st.warning("⚠️ High Risk Detected: Smoking & high BMI significantly increase costs.")
            elif smoker == 'yes':
                st.info("💡 Note: Smoking is the primary driver for increased insurance premiums.")
        else:
            st.error("Predictor is not available.")

elif app_mode == "Batch Prediction":
    st.header("📂 Batch Prediction")
    st.write("Upload a CSV file containing columns: `age`, `sex`, `bmi`, `children`, `smoker`, `region`.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📋 Input Data Sample:")
            st.dataframe(df.head())
            
            if st.button("Run Batch Inference"):
                if predictor:
                    with st.spinner("Processing..."):
                        predictions = predictor.predict(df)
                        df['predicted_charges'] = predictions
                        
                        st.write("✅ Results:")
                        st.dataframe(df.head(10))
                        
                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Results as CSV",
                            csv,
                            "insurance_predictions.csv",
                            "text/csv",
                            key='download-csv'
                        )
                else:
                    st.error("Predictor is not available.")
        except Exception as e:
            st.error(f"Error: {e}")

elif app_mode == "Data Insights":
    st.header("📊 Data Insights & Trends")
    st.write("Visualizations from the underlying insurance dataset.")
    
    # Load dataset
    @st.cache_data
    def load_data():
        return pd.read_csv('dataset/insurance.csv')
        
    try:
        data = load_data()
        
        # Row 1: Key Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Average Charges", f"${data['charges'].mean():,.0f}")
        m2.metric("Median Age", f"{data['age'].median():.0f}")
        m3.metric("% Smokers", f"{len(data[data['smoker']=='yes'])/len(data)*100:.1f}%")
        
        # Row 2: Distribution Visuals
        st.write("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Charges Distribution")
            fig_hist = px.histogram(data, x="charges", color="smoker", nbins=50, title="Charges by Smoker Status")
            st.plotly_chart(fig_hist)
            
        with col2:
            st.subheader("Age vs Charges")
            fig_scatter = px.scatter(data, x="age", y="charges", color="smoker", trendline="ols", title="Impact of Age on Costs")
            st.plotly_chart(fig_scatter)
            
        # Row 3: Categorical Breakdown
        st.write("---")
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("BMI Impact")
            data['bmi_category'] = pd.cut(data['bmi'], [0, 18.5, 25, 30, 100], labels=['Under', 'Normal', 'Over', 'Obese'])
            fig_box = px.box(data, x="bmi_category", y="charges", color="smoker", title="Charges by BMI Category")
            st.plotly_chart(fig_box)
            
        with col4:
            st.subheader("Regional Trends")
            fig_bar = px.bar(data.groupby('region')['charges'].mean().reset_index(), x='region', y='charges', title="Average Charges by Region")
            st.plotly_chart(fig_bar)

    except Exception as e:
        st.error(f"Unable to load data insights: {e}")
