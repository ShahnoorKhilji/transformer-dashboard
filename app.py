import streamlit as st
import pandas as pd
import numpy as np
import time

# Page Configuration
st.set_page_config(page_title="100 kVA Transformer Digital Twin", layout="wide")

st.title("⚡ 100 kVA Distribution Transformer Cloud Dashboard")
st.markdown("Real-time predictive maintenance monitoring, anomaly detection, and 15-day RUL countdown.")

# Load or Generate Dataset on the Fly (Bina CSV file ke)
@st.cache_data
def get_data():
    # Agar CSV nahi hai toh chota optimized dataset live generate kar lo
    np.random.seed(42)
    n = 1000
    time_arr = np.linspace(0, 100, n)
    temp = np.linspace(90, 165, n) + np.random.normal(0, 0.5, n)
    v_rms = np.random.normal(230, 2, n)
    vib = np.linspace(0.1, 2.5, n) + np.random.normal(0, 0.05, n)
    i_thd = np.linspace(2, 11, n)
    
    df = pd.DataFrame({
        'time': time_arr,
        'Temp': temp,
        'V_rms': v_rms,
        'Vib': vib,
        'I_THD': i_thd,
        'dTemp_dt': np.gradient(temp),
        'Anomaly_Score': [1 if t < 155 else -1 for t in temp],
        'TTF_Days': np.linspace(15, 0, n)
    })
    return df

df = get_data()

# Sidebar Controls
st.sidebar.header("Dashboard Controls")
simulation_speed = st.sidebar.slider("Simulation Speed (Delay in seconds)", 0.01, 0.2, 0.05)
run_live = st.sidebar.checkbox("Start Real-Time Simulation Stream", value=True)

# Main Dashboard Layout
placeholder = st.empty()

if run_live:
    for idx, row in df.iterrows():
        with placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            
            temp = row['Temp']
            v_rms = row['V_rms']
            vib = row['Vib']
            ttf_days = row['TTF_Days']
            anomaly = "Normal" if row['Anomaly_Score'] == 1 else "Anomaly Detected"
            
            col1.metric("Oil Temperature", f"{temp:.1f} °C", delta=f"{row['dTemp_dt']:.2f} °C/s")
            col2.metric("Voltage (V_rms)", f"{v_rms:.2f} V")
            col3.metric("Vibration", f"{vib:.3f} mm/s")
            col4.metric("Est. Remaining Life (RUL)", f"{ttf_days:.2f} Days")
            
            # Alerts & Status Banner
            if ttf_days <= 2.0 or temp >= 160.0:
                st.error(f"🚨 CRITICAL ALERT: Transformer trip imminent! Temperature reached {temp:.1f}°C.")
            elif ttf_days <= 15.0:
                st.warning(f"⚠️ MAINTENANCE WARNING: Schedule service within 10-15 days. RUL is {ttf_days:.1f} days.")
            else:
                st.success("✅ Transformer Operating Status: HEALTHY")
                
            # Real-time Charts
            chart_data = df.iloc[:idx+1][['time', 'Temp', 'Vib']]
            st.line_chart(chart_data.set_index('time'))
            
            time.sleep(simulation_speed)
            if idx >= len(df) - 1:
                break
else:
    st.info("Check 'Start Real-Time Simulation Stream' in the sidebar to begin live monitoring.")