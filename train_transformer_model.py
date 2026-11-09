import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 1. Normal baseline data load karein
df = pd.read_csv('Transformer_Normal_Baseline.csv')

# 2. Features select karein (3-phase parameters)
features = [
    'Temperature', 
    'V_rms_1', 'V_rms_2', 'V_rms_3', 
    'I_rms_1', 'I_rms_2', 'I_rms_3', 
    'I_THD', 'V_THD', 'Vibration'
]
X_train = df[features]

# 3. Scale aur Train (Isolation Forest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X_train_scaled)

# 4. Predictions nikalen
predictions = iso_forest.predict(X_train_scaled)

# Shuruat ke initial transients (energization phase) ko chhod kar steady-state check karte hain
# Misal ke taur par, t > 1.0 second ke baad ka data dekhte hain
steady_state_mask = df['Time'] > 1.0
steady_state_preds = predictions[steady_state_mask]

# Anomalies count karein steady-state mein
anomalies_in_steady_state = np.sum(steady_state_preds == -1)
total_steady_state_points = len(steady_state_preds)

print("="*50)
print(" TRANSFORMER HEALTH MONITORING - BASELINE TEST")
print("="*50)

if anomalies_in_steady_state == 0 or (anomalies_in_steady_state / total_steady_state_points) < 0.005:
    print(" Status: ✅ HEALTHY / NORMAL")
    print(" Message: Transformer is operating under normal conditions.")
    print(" Action Required: 🟢 NO MAINTENANCE NEEDED")
else:
    print(f" Status: ⚠️ Minor fluctuations detected ({anomalies_in_steady_state} points)")
    print(" Action Required: Monitor closely.")

print("="*50)