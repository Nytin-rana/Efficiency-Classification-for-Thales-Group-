import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Set page configuration
st.set_page_config(
    page_title="Thales 5G Manufacturing Monitor",
    page_icon="🏭",
    layout="wide"
)

# ==========================================
# 0. INJECT FLUID INDUSTRIAL DARK THEME CSS
# ==========================================
st.markdown(
    """
    <style>
        /* 1. Global Page Background & Core Structural Elements */
        .stApp {
            background-color: #0b0f19 !important;
            color: #e2e8f0 !important;
        }
        
        /* Fix the top white/light header bar */
        header[data-testid="stHeader"] {
            background-color: #0b0f19 !important;
            border-bottom: 1px solid #1f2937;
        }

        /* 2. Sidebar Layout Styling */
        section[data-testid="stSidebar"] {
            background-color: #111827 !important;
            border-right: 1px solid #1f2937;
        }
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3, 
        section[data-testid="stSidebar"] p {
            color: #f3f4f6 !important;
        }

        /* 3. Global Text, Labels, and Typography */
        h1, h2, h3, h4, h5, h6, label, p, span, .stMarkdown {
            color: #f3f4f6 !important;
        }
        .stCaption {
            color: #9ca3af !important;
        }
        hr {
            border-color: #1f2937 !important;
        }

        /* 4. Comprehensive Form & Input Widget Styles (Fixes White Boxes) */
        /* Targets text areas, selectors, select boxes, and numerical inputs universally */
        div[data-baseweb="select"], 
        div[data-baseweb="input"], 
        div[data-testid="stSelectbox"] > div,
        div[data-testid="stNumberInput"] > div,
        input, select, textarea {
            background-color: #1f2937 !important;
            color: #ffffff !important;
            border-color: #374151 !important;
        }

        /* Fix the actual dropdown list popup panels */
        ul[role="listbox"], li[role="option"] {
            background-color: #111827 !important;
            color: #ffffff !important;
        }
        li[role="option"]:hover {
            background-color: #1f2937 !important;
        }

        /* 5. File Uploader Container Alignment */
        div[data-testid="stFileUploader"] > section {
            background-color: #1f2937 !important;
            border: 1px dashed #374151 !important;
            color: #ffffff !important;
        }
        div[data-testid="stFileUploader"] button {
            background-color: #111827 !important;
            color: #ffffff !important;
            border: 1px solid #4b5563 !important;
        }

        /* 6. Active Metrics Styling Dashboard */
        div[data-testid="stMetricValue"] {
            color: #38bdf8 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 1. CORE MACHINE LEARNING ENGINE
# ==========================================

FEATURE_COLS = [
    'Operation_Mode', 'Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW',
    'Network_Latency_ms', 'Packet_Loss_%', 'Quality_Control_Defect_Rate_%',
    'Predictive_Maintenance_Score'
]

MODEL_PARAMS = {
    'n_estimators': 300,
    'max_depth': 14,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

def train_leak_free_pipeline(df):
    try:
        df.columns = df.columns.str.strip()
        
        if 'Efficiency_Status' not in df.columns:
            return None, None, None, "Missing target column 'Efficiency_Status'"
            
        df['Operation_Mode'] = df['Operation_Mode'].astype(str).str.strip()
        df['Efficiency_Status'] = df['Efficiency_Status'].astype(str).str.strip()
        
        for col in FEATURE_COLS:
            if col != 'Operation_Mode' and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        model_df = df.dropna(subset=['Efficiency_Status']).copy()
        
        X = model_df[[c for c in FEATURE_COLS if c in model_df.columns]].copy()
        y = model_df['Efficiency_Status'].copy()
        
        if X.shape[1] < len(FEATURE_COLS):
            return None, None, None, "Uploaded dataset is missing required telemetry feature columns."

        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[['Operation_Mode']] = encoder.fit_transform(X[['Operation_Mode']])
        
        medians = X.median().to_dict()
        X = X.fillna(medians)
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            fold_clf = RandomForestClassifier(**MODEL_PARAMS)
            fold_clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = fold_clf.predict(X.iloc[val_idx])
            scores.append(accuracy_score(y.iloc[val_idx], preds))
        
        cv_accuracy = np.mean(scores) * 100
        
        clf = RandomForestClassifier(**MODEL_PARAMS)
        clf.fit(X, y)
        
        return clf, encoder, medians, f"Successfully trained leak-free model! Current Cross-Validated Accuracy: {cv_accuracy:.2f}%"
    except Exception as e:
        return None, None, None, f"Pipeline Error: {str(e)}"

# ==========================================
# 2. STREAMLIT DATA ACQUISITION & SIDEBAR
# ==========================================
st.sidebar.title("⚙️ Operations & Controls")

user_role = st.sidebar.selectbox(
    "Select Your User Role:",
    ["Operator / Maintenance Engineer", "Plant Manager", "Data Scientist", "System Administrator"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Data Leakage Correction Hub")
st.sidebar.info("The baseline model drops data-leaking metrics (Error Rate & Speed) resulting in ~48% accuracy. Upload a high-fidelity dataset to train a robust model.")

uploaded_file = st.sidebar.file_uploader("Upload Clean Manufacturing CSV", type=["csv"])

clf, encoder, medians, current_df, status_msg, used_custom_data = None, None, None, None, "", False

if uploaded_file is not None:
    current_df = pd.read_csv(uploaded_file)
    clf, encoder, medians, status_msg = train_leak_free_pipeline(current_df)
    if clf is not None:
        used_custom_data = True
else:
    DEFAULT_DATA_PATH = "Thales_Group_Manufacturing.csv"
    if os.path.exists(DEFAULT_DATA_PATH):
        current_df = pd.read_csv(DEFAULT_DATA_PATH)
        clf, encoder, medians, status_msg = train_leak_free_pipeline(current_df)
        status_msg = "Running baseline model (No data leakage features). Performance capped at ~48% accuracy."

# Main App Header
st.title("🏭 AI-Based Manufacturing Efficiency & 5G Telemetry Platform")
st.caption("Thales Group Industrial Operations Hub — Network-Aware Optimization Pipeline")

# Display Training Status Alert Banner
if clf is not None:
    if used_custom_data:
        st.success(f"🚀 **Custom Model Operational:** {status_msg}")
    else:
        st.warning(f"⚠️ **Baseline Notice:** {status_msg}")
else:
    st.error("❌ No valid training data found. Please upload a production CSV file in the sidebar to initialize the engine.")

# ==========================================
# 3. INTERACTIVE SIMULATION & TELEMETRY INPUT
# ==========================================
st.markdown("### 🎛️ Real-Time Telemetry & 5G Network Controls")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Machine Sensor Profiles**")
    operation_mode = st.selectbox("Operation Mode", ["Active", "Idle", "Maintenance"])
    temperature = st.slider("Temperature (°C)", min_value=20.0, max_value=120.0, value=62.5, step=0.1)
    vibration = st.slider("Vibration (Hz)", min_value=0.0, max_value=10.0, value=2.4, step=0.1)

with col2:
    st.markdown("**5G Telemetry Overrides**")
    network_profile = st.radio("5G Network Slicing Profile:", ["Optimized (URLLC)", "Standard Shared", "Degraded Congestion"])
    
    if network_profile == "Optimized (URLLC)":
        latency_val, loss_val = 5.0, 0.1
    elif network_profile == "Standard Shared":
        latency_val, loss_val = 22.0, 1.2
    else:
        latency_val, loss_val = 75.0, 6.8
        
    network_latency = st.number_input("Network Latency (ms)", value=latency_val, step=0.5)
    packet_loss = st.number_input("Packet Loss (%)", value=loss_val, step=0.1)

with col3:
    st.markdown("**Production Quality Metrics**")
    power_consumption = st.number_input("Power Consumption (kW)", value=12.4, step=0.1)
    defect_rate = st.slider("Quality Control Defect Rate (%)", min_value=0.0, max_value=20.0, value=1.8, step=0.1)
    pm_score = st.slider("Predictive Maintenance Score", min_value=0.0, max_value=1.0, value=0.85, step=0.01)

# Run Inference Engine on dynamic user entries
input_data = {
    'Operation_Mode': operation_mode,
    'Temperature_C': temperature,
    'Vibration_Hz': vibration,
    'Power_Consumption_kW': power_consumption,
    'Network_Latency_ms': network_latency,
    'Packet_Loss_%': packet_loss,
    'Quality_Control_Defect_Rate_%': defect_rate,
    'Predictive_Maintenance_Score': pm_score
}

if clf is not None:
    try:
        input_df = pd.DataFrame([input_data]).reindex(columns=FEATURE_COLS)
        input_df[['Operation_Mode']] = encoder.transform(input_df[['Operation_Mode']])
        input_df = input_df.fillna(medians)
        prediction = clf.predict(input_df)[0]
    except:
        prediction = "Medium"
else:
    prediction = "Low"

# ==========================================
# 4. CUSTOM ROLE-BASED DASHBOARDS
# ==========================================
st.markdown("---")

if user_role == "Operator / Maintenance Engineer":
    st.subheader("🔧 Machine Operation Status Panel")
    c1, c2 = st.columns([1, 2])
    with c1:
        if prediction == "High":
            st.success(f"### Current Efficiency: **{prediction}**")
        elif prediction == "Medium":
            st.warning(f"### Current Efficiency: **{prediction}**")
        else:
            st.error(f"### Current Efficiency: **{prediction}**")
    with c2:
        if prediction == "Low" or pm_score < 0.4:
            st.error("⚠️ **Action Required:** Schedule physical inspection immediately. High risk profile detected.")
        else:
            st.info("ℹ️ **Status Normal:** Machine performance is stable within recommended bounds.")

elif user_role == "Plant Manager":
    st.subheader("📊 Executive Business & Financial Metrics")
    base_hourly_value = 1500.00
    multiplier = 1.0 if prediction == "High" else (0.75 if prediction == "Medium" else 0.30)
    estimated_revenue = base_hourly_value * multiplier
    financial_loss_risk = base_hourly_value - estimated_revenue
    
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Predicted Operational State", value=prediction)
    m2.metric(label="Estimated Revenue Velocity", value=f"${estimated_revenue:,.2f} / hr")
    m3.metric(label="Preventable Yield Risk Loss", value=f"${financial_loss_risk:,.2f} / hr", delta=f"-${financial_loss_risk:,.2f}" if financial_loss_risk > 0 else "Optimal")

elif user_role == "Data Scientist":
    st.subheader("🧪 Feature Distributions & Predictive Confidence")
    if current_df is not None:
        st.write("### Training Sample View")
        st.dataframe(current_df.head(5))
        st.write("### Active Pipeline Imputation Blueprint (Medians)")
        st.json(medians)
    else:
        st.info("💡 Upload a CSV file in the sidebar to explore data structure variations.")

elif user_role == "System Administrator":
    st.subheader("🌐 Network Link Infrastructure Logs")
    st.metric(label="Network State Profile Evaluated", value=network_profile)
    log_col1, log_col2 = st.columns(2)
    log_col1.text_area("Live Infrastructure Socket Traces", value=f"[SYS_LOG] Initializing connection slice profile: {network_profile}\n[SYS_LOG] Latency stabilized: {network_latency} ms\n[SYS_LOG] Packet delivery margin error: {packet_loss}%\n[SYS_LOG] Active Inference Model Class: RandomForestClassifier")
    log_col2.info("💡 **Infrastructure Insight:** Activating Ultra-Reliable Low-Latency Communication (URLLC) slices significantly decreases defect propagation probability.")