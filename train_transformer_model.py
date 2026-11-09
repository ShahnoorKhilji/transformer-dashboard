import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Dataset Load Karein
df = pd.read_csv('transformer_degradation_data.csv')
print("Dataset Loaded Successfully. Shape:", df.shape)

# 2. IEEE Threshold Based Failure Condition
failure_condition = (df['Temp'] >= 160.0) | (df['I_THD'] >= 10.0)

if failure_condition.any():
    failure_idx = df[failure_condition].index[0]
    t_failure = df.loc[failure_idx, 'time']
else:
    t_failure = df['time'].iloc[-1]

# 3. Time-To-Failure (TTF) ko Days mein Scale karein 
# (Maante hain ki simulation ka total time span 15 days ko represent kar raha hai)
max_sim_time = df['time'].max()
df['Remaining_Time_Seconds'] = t_failure - df['time']

# Scale seconds to 15 Days maximum lifetime countdown
df['TTF_Days'] = (df['Remaining_Time_Seconds'] / max_sim_time) * 15.0
df['TTF_Days'] = df['TTF_Days'].apply(lambda x: max(0.0, x))

# 4. Feature Engineering
df['dTemp_dt'] = df['Temp'].diff().fillna(0) / df['time'].diff().fillna(1e-5)
df['Rolling_Vib'] = df['Vib'].rolling(window=10, min_periods=1).mean()

# 5. Hybrid Model: Isolation Forest & Random Forest (RUL in Days)
features = ['V_rms', 'I_rms', 'V_THD', 'I_THD', 'Temp', 'Vib', 'dTemp_dt']
X = df[features]

iso_forest = IsolationForest(contamination=0.1, random_state=42)
df['Anomaly_Score'] = iso_forest.fit_predict(X)

y_rul = df['TTF_Days']
X_train, X_test, y_train, y_test = train_test_split(X, y_rul, test_size=0.2, random_state=42)

rul_model = RandomForestRegressor(n_estimators=100, random_state=42)
rul_model.fit(X_train, y_train)

y_pred = rul_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RUL Model (in Days) Trained! Test RMSE: {rmse:.4f} Days")

# Save updated dataset
df.to_csv('transformer_rul_dataset.csv', index=False)
print("Updated dataset saved as transformer_rul_dataset.csv!")