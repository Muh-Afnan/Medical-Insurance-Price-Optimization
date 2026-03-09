import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import time

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.predict import InsurancePredictor

# --- Page Configuration ---
st.set_page_config(
    page_title="Premium Insurance Optimizer",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern Styling (CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    :root {
        --primary-teal: #004d40;
        --secondary-mint: #e0f2f1;
        --accent-coral: #ff7043;
        --glass-bg: rgba(255, 255, 255, 0.7);
        --glass-border: rgba(255, 255, 255, 0.3);
    }

    html, body, [class*="st-"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #e0f2f1 0%, #ffffff 100%);
    }

    /* Glassmorphic Card */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        margin-bottom: 2rem;
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
    }

    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(45deg, #004d40, #00897b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Custom Metrics */
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid #eee;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--primary-teal);
        color: white;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: rgba(255, 255, 255, 0.8);
    }

</style>
""", unsafe_allow_html=True)

# --- Data & Model Loading ---
@st.cache_resource
def get_predictor():
    try:
        return InsurancePredictor(model_path='models/insurance_model_pipeline.joblib')
    except Exception as e:
        return None

@st.cache_data
def load_data():
    try:
        return pd.read_csv('dataset/insurance.csv')
    except:
        return pd.DataFrame()

predictor = get_predictor()
df_raw = load_data()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062325.png", width=80)
    st.markdown("<h1 style='color: white; margin-bottom: 0;'>Antigravity</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.9rem; margin-top: 0;'>Insurance Intelligence Suite</p>", unsafe_allow_html=True)
    st.write("---")
    
    app_mode = st.radio(
        "Navigate",
        ["✨ Cost Simulator", "📊 Data Insights", "📂 Batch Analysis"],
        index=0
    )
    
    st.write("---")
    st.markdown("### Model Status")
    if predictor:
        st.success("🟢 Model Pipeline Ready")
    else:
        st.error("🔴 Model Not Found")
        st.info("Please run training pipeline first.")

# --- Main Dashboard Logic ---

if app_mode == "✨ Cost Simulator":
    st.markdown("<h1 class='gradient-text'>Insurance Premium Simulator</h1>", unsafe_allow_html=True)
    st.markdown("Estimate your annual medical insurance charges with AI precision.")
    
    col_input, col_viz = st.columns([1, 1.2])
    
    with col_input:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Personal Demographics")
        
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 18, 100, 30, help="User's current age")
            sex = st.selectbox("Biological Sex", ["female", "male"])
            children = st.select_slider("Children/Dependents", options=[0, 1, 2, 3, 4, "5+"], value=0)
        
        with c2:
            bmi = st.number_input("BMI Index", 10.0, 60.0, 25.0, step=0.1)
            smoker = st.radio("Smoking Status", ["no", "yes"], horizontal=True)
            region = st.selectbox("Residential Region", ["southwest", "southeast", "northwest", "northeast"])
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Real-time calculation if model exists
        if predictor:
            input_dict = {
                'age': age,
                'sex': sex,
                'bmi': bmi,
                'children': children if isinstance(children, int) else 5,
                'smoker': smoker,
                'region': region
            }
            prediction = predictor.predict(input_dict)[0]
        else:
            prediction = 0.0

    with col_viz:
        st.markdown("<div class='glass-card' style='text-align: center;'>", unsafe_allow_html=True)
        st.subheader("Estimated Annual Charge")
        
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            number = {'prefix': "$", 'font': {'size': 50, 'color': "#004d40"}},
            gauge = {
                'axis': {'range': [None, 65000], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#004d40"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 15000], 'color': '#e0f2f1'},
                    {'range': [15000, 35000], 'color': '#80cbc4'},
                    {'range': [35000, 65000], 'color': '#4db6ac'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50000
                }
            }
        ))
        fig.update_layout(height=400, margin=dict(l=30, r=30, t=50, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison logic
        if not df_raw.empty:
            avg_all = df_raw['charges'].mean()
            diff = ((prediction - avg_all) / avg_all) * 100
            color = "red" if diff > 0 else "green"
            arrow = "↑" if diff > 0 else "↓"
            st.markdown(f"**{arrow} {abs(diff):.1f}%** vs National Average (${avg_all:,.0f})", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Risk Factors
        with st.expander("🔍 Risk Insights & Interpretation", expanded=True):
            cols = st.columns(3)
            if smoker == 'yes':
                cols[0].warning("🚬 **Smoker Status**\nMajor cost driver (+150% avg)")
            else:
                cols[0].success("🚭 **Non-Smoker**\nSignificant savings profile")
                
            if bmi > 30:
                cols[1].warning("⚖️ **BMI Index**\nHigh BMI impacts risk premiums")
            else:
                cols[1].success("⚖️ **Health Metric**\nBMI within optimal range")
                
            if age > 45:
                cols[2].info("⏳ **Age Factor**\nCost increases with age brackets")
            else:
                cols[2].success("👧 **Age Bracket**\nLower base premium group")
            
            st.write("---")
            st.markdown("#### **Why this price?** (Local Interpretation)")
            
            # Simple Local Interpretation logic: 
            # We compare the current prediction against a 'baseline' (average person)
            # and explain the shifts.
            if predictor and not df_raw.empty:
                baseline_input = {
                    'age': 39, # Median
                    'sex': 'female',
                    'bmi': 30, # Borderline obese
                    'children': 1,
                    'smoker': 'no',
                    'region': 'southeast'
                }
                baseline_pred = predictor.predict(baseline_input)[0]
                
                diff_from_baseline = prediction - baseline_pred
                if diff_from_baseline > 5000:
                    st.markdown(f"Your estimate is **${abs(diff_from_baseline):,.0f} higher** than a standard profile primarily due to " + 
                                ("**Smoking**" if smoker=='yes' else "**higher BMI/Age**") + ".")
                elif diff_from_baseline < -2000:
                    st.markdown(f"Your estimate is **${abs(diff_from_baseline):,.0f} lower** than a standard profile thanks to your **non-smoker status** and **favorable demographic factors**.")
                else:
                    st.markdown("Your estimate is aligned with the standard demographic profile.")

                # Sensitivity visualization
                st.markdown("*Impact of changing your profile:*")
                # Vary age by +/- 10
                alt_input = input_dict.copy()
                alt_input['age'] = max(18, age - 10)
                pred_younger = predictor.predict(alt_input)[0]
                age_saving = prediction - pred_younger
                
                if age_saving > 0:
                    st.caption(f"💡 Being 10 years younger would have reduced your premium by approx. **${age_saving:,.0f}**.")
                
                if smoker == 'yes':
                    alt_input = input_dict.copy()
                    alt_input['smoker'] = 'no'
                    pred_nosmoke = predictor.predict(alt_input)[0]
                    smoke_saving = prediction - pred_nosmoke
                    st.caption(f"🔥 Quitting smoking could save you approx. **${smoke_saving:,.0f}** per year!")

elif app_mode == "📊 Data Insights":
    st.markdown("<h1 class='gradient-text'>Market Insights & Trends</h1>", unsafe_allow_html=True)
    
    if df_raw.empty:
        st.warning("No dataset found for insights.")
    else:
        # Key Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown("<div class='metric-container'><h3>Total Records</h3><h2 style='color:#004d40'>"+str(len(df_raw))+"</h2></div>", unsafe_allow_html=True)
        with m2:
            st.markdown("<div class='metric-container'><h3>Avg Charge</h3><h2 style='color:#004d40'>$"+f"{df_raw['charges'].mean():,.0f}"+"</h2></div>", unsafe_allow_html=True)
        with m3:
            st.markdown("<div class='metric-container'><h3>Smoker %</h3><h2 style='color:#ff7043'>"+f"{(df_raw['smoker']=='yes').mean()*100:.1f}%"+"</h2></div>", unsafe_allow_html=True)
        with m4:
            st.markdown("<div class='metric-container'><h3>Avg BMI</h3><h2 style='color:#004d40'>"+f"{df_raw['bmi'].mean():.1f}"+"</h2></div>", unsafe_allow_html=True)
            
        st.write("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Charges Distribution by Smoker")
            fig = px.histogram(df_raw, x="charges", color="smoker", marginal="box", color_discrete_sequence=['#4db6ac', '#ff7043'], barmode="overlay")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("BMI vs Charges Correlation")
            fig = px.scatter(df_raw, x="bmi", y="charges", color="smoker", size="age", hover_data=['region'], color_discrete_sequence=['#4db6ac', '#ff7043'], opacity=0.6)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        st.write("---")
        st.subheader("Market Benchmarking & Model Transparency")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Regional Price Benchmarking")
            fig = px.box(df_raw, x="region", y="charges", color="sex", points="all", color_discrete_sequence=['#004d40', '#80cbc4'])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("#### Global Feature Importance (AI View)")
            if predictor:
                imp_df = predictor.get_feature_importance()
                if not imp_df.empty:
                    fig = px.bar(imp_df.head(10), x='Importance', y='Feature', orientation='h', 
                                 title="Top Drivers for Insurance Costs",
                                 color_discrete_sequence=['#ff7043'])
                    fig.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type.")
            else:
                st.error("Predictor not loaded.")

elif app_mode == "📂 Batch Analysis":
    st.markdown("<h1 class='gradient-text'>Enterprise Batch Prediction</h1>", unsafe_allow_html=True)
    st.write("Upload high-volume data for instant premium estimation.")
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop CSV file here", type=["csv"])
    
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded {len(batch_df)} records.")
        
        st.markdown("### Data Preview")
        st.dataframe(batch_df.head(), use_container_width=True)
        
        if st.button("🚀 Run Mass Prediction", type="primary"):
            if predictor:
                with st.spinner("Processing large-scale inference..."):
                    results = predictor.predict(batch_df)
                    batch_df['Predicted_Charges'] = results
                    time.sleep(1) # Visual effect
                
                st.markdown("### Processed Results")
                st.dataframe(batch_df, use_container_width=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.histogram(batch_df, x="Predicted_Charges", title="Distribution of Predictions", color_discrete_sequence=['#004d40'])
                    st.plotly_chart(fig)
                with c2:
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Annotated CSV",
                        data=csv,
                        file_name="insurance_batch_results.csv",
                        mime="text/csv",
                    )
            else:
                st.error("Predictor engine not loaded.")
    else:
        # Show example template
        st.info("Template requirement: CSV must contain 'age', 'sex', 'bmi', 'children', 'smoker', 'region'")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Powered by Antigravity AI Engine v1.0 • Medical Insurance Price Optimization Dashboard")
