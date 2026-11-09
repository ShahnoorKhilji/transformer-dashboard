import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Page Setup
st.set_page_config(page_title="Transformer Health Dashboard", page_icon="⚡", layout="wide")

st.title("⚡ 100 kVA Distribution Transformer - AI Health Dashboard")
st.markdown("Monitoring transformer baseline health status using **Isolation Forest** (Internal Baseline Simulation).")

# 1. Internal Baseline Data Generate Karein (Bina CSV ke)
@st.cache_data
def load_data():
    np.random.seed(42)
    time_steps = np.linspace(0, 10, 1000)
    
    # Normal baseline data simulation for 3-phase transformer
    data = {
        'Time': time_steps,
        'Temperature': np.random.normal(65.0, 1.5, 1000),
        'V_rms_1': np.random.normal(230.0, 2.0, 1000),
        'V_rms_2': np.random.normal(230.0, 2.0, 1000),
        'V_rms_3': np.random.normal(230.0, 2.0, 1000),
        'I_rms_1': np.random.normal(100.0, 5.0, 1000),
        'I_rms_2': np.random.normal(100.0, 5.0, 1000),
        'I_rms_3': np.random.normal(100.0, 5.0, 1000),
        'I_THD': np.random.uniform(1.0, 2.5, 1000),
        'V_THD': np.random.uniform(0.5, 1.5, 1000),
        'Vibration': np.random.normal(0.05, 0.01, 1000)
    }
    return pd.DataFrame(data)

df = load_data()

# 2. Features & Training
features = ['Temperature', 'V_rms_1', 'V_rms_2', 'V_rms_3', 'I_rms_1', 'I_rms_2', 'I_rms_3', 'I_THD', 'V_THD', 'Vibration']
X_train = df[features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X_train_scaled)
predictions = iso_forest.predict(X_train_scaled)
df['Anomaly_Status'] = predictions

# 3. Top Summary Metrics (Results)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("System Rating", "100 kVA")
with col2:
    st.metric("Avg Temperature", f"{df['Temperature'].mean():.2f} °C")
with col3:
    st.metric("Max I_THD", f"{df['I_THD'].max():.2f} %")
with col4:
    anomalies_count = np.sum(predictions == -1)
    status = "🟢 HEALTHY (Baseline)" if anomalies_count < 50 else "⚠️ CHECK REQUIRED"
    st.metric("Transformer Status", status)

st.markdown("---")

# 4. Data Table Preview
st.subheader("📋 Baseline Dataset & Health Results")
st.dataframe(df.head(100), use_container_width=True)