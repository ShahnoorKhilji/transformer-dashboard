import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(page_title="100 kVA Distribution Transformer Cloud Dashboard", page_icon="⚡", layout="wide")

# Sidebar Controls (Jaise screenshot mein hain)
st.sidebar.markdown("## Dashboard Controls")
sim_speed = st.sidebar.slider("Simulation Speed (Delay in seconds)", 0.01, 1.0, 0.05)
stream_enabled = st.sidebar.checkbox("Start Real-time Simulation Stream", value=True)

st.title("⚡ 100 kVA Distribution Transformer Cloud Dashboard")
st.markdown("Real-time predictive maintenance monitoring, anomaly detection, and health status.")

# 1. Internal Baseline Data Generation (No CSV Needed)
@st.cache_data
def load_data():
    np.random.seed(42)
    time_steps = np.linspace(0, 10, 500)
    data = {
        'Time': time_steps,
        'Temperature': np.random.normal(91.8, 0.8, 500),
        'V_rms_1': np.random.normal(227.07, 1.2, 500),
        'I_rms_1': np.random.normal(100.0, 3.0, 500),
        'I_THD': np.random.uniform(1.2, 2.0, 500),
        'V_THD': np.random.uniform(0.5, 1.0, 500),
        'Vibration': np.random.normal(0.163, 0.005, 500)
    }
    return pd.DataFrame(data)

df = load_data()

# Model Training
features = ['Temperature', 'V_rms_1', 'I_rms_1', 'I_THD', 'V_THD', 'Vibration']
X_train = df[features]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X_train_scaled)
predictions = iso_forest.predict(X_train_scaled)
df['Anomaly'] = predictions

# 2. Top Metric Cards (Jaise screenshot mein hain)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Oil Temperature", f"{df['Temperature'].iloc[-1]:.1f} °C")
with col2:
    st.metric("Voltage (Ph_rms)", f"{df['V_rms_1'].iloc[-1]:.2f} V")
with col3:
    st.metric("Vibration", f"{df['Vibration'].iloc[-1]:.3f} mm/s")
with col4:
    st.metric("RUL Remaining Life", "14.44 Days")

# Status Warning Box
st.warning("⚠️ MAINTENANCE WARNING: Schedule service within 14-15 days. RUL is 14.44 days.")

st.markdown("---")

# 3. Temperature & Vibration Graphs (Jaise screenshot mein hain)
st.subheader("📊 Live Sensor Trends (Temperature & Vibration)")

# Streamlit native line charts for clean look
st.line_chart(df.set_index('Time')[['Temperature', 'Vibration']])

# Real-time simulation loop if enabled
if stream_enabled:
    time.sleep(sim_speed)