# =========================================================
# SHORT-TERM FERRY DEMAND FORECASTING SYSTEM
# ADVANCED STREAMLIT DASHBOARD
# =========================================================

# RUN USING:
# streamlit run app2.py

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Ferry Demand Forecasting Dashboard",
    page_icon="🚢",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #003366;
}

.metric-container {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("🚢 Ferry Ticket Demand Forecasting Dashboard")

st.markdown("""
### Toronto Government Parks, Forestry & Recreation

This dashboard predicts short-term ferry ticket demand using
Machine Learning and Time-Series Forecasting techniques.
""")

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("Toronto Island Ferry Tickets.csv")

# =========================================================
# DATA PREPROCESSING
# =========================================================

df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df = df.sort_values('Timestamp')

# =========================================================
# FEATURE ENGINEERING
# =========================================================

df['hour'] = df['Timestamp'].dt.hour
df['day_of_week'] = df['Timestamp'].dt.dayofweek
df['month'] = df['Timestamp'].dt.month

df['is_weekend'] = np.where(
    df['day_of_week'] >= 5,
    1,
    0
)

# Peak Hour
df['is_peak_hour'] = np.where(
    (df['hour'] >= 16) & (df['hour'] <= 20),
    1,
    0
)

# Lag Features
df['lag_1'] = df['Sales Count'].shift(1)
df['lag_2'] = df['Sales Count'].shift(2)
df['lag_4'] = df['Sales Count'].shift(4)

# Rolling Mean
df['rolling_mean_4'] = (
    df['Sales Count']
    .rolling(window=4)
    .mean()
)

# Remove Missing Values
df.dropna(inplace=True)

# =========================================================
# FEATURE SELECTION
# =========================================================

features = [
    'hour',
    'day_of_week',
    'month',
    'is_weekend',
    'is_peak_hour',
    'lag_1',
    'lag_2',
    'lag_4',
    'rolling_mean_4'
]

X = df[features]

y = df['Sales Count']

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# =========================================================
# MODEL TRAINING
# =========================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# PREDICTIONS
# =========================================================

predictions = model.predict(X_test)

predictions = np.maximum(predictions, 0)

# =========================================================
# MODEL METRICS
# =========================================================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

forecast_accuracy = round(r2 * 100, 2)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙ Dashboard Controls")

num_rows = st.sidebar.slider(
    "Select Number of Rows",
    100,
    5000,
    500
)

future_steps = st.sidebar.slider(
    "Forecast Horizon (15 min intervals)",
    1,
    8,
    4
)

# =========================================================
# KPI SECTION
# =========================================================

st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

peak_demand = int(df['Sales Count'].max())
avg_demand = int(df['Sales Count'].mean())

col1.metric(
    "Forecast Accuracy",
    f"{forecast_accuracy}%"
)

col2.metric(
    "Peak Demand",
    peak_demand
)

col3.metric(
    "Average Demand",
    avg_demand
)

col4.metric(
    "RMSE",
    round(rmse, 2)
)

# =========================================================
# DEMAND TREND
# =========================================================

st.subheader("📈 Historical Ferry Demand")

historical_fig = px.line(
    df.head(num_rows),
    x='Timestamp',
    y='Sales Count',
    title='Historical Ferry Ticket Demand'
)

st.plotly_chart(
    historical_fig,
    use_container_width=True
)

# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

st.subheader("🤖 Actual vs Predicted Demand")

results_df = pd.DataFrame({
    'Timestamp': df.iloc[split_index:].Timestamp,
    'Actual Demand': y_test.values,
    'Predicted Demand': predictions
})

forecast_fig = px.line(
    results_df.head(num_rows),
    x='Timestamp',
    y=['Actual Demand', 'Predicted Demand'],
    title='Model Forecast Performance'
)

st.plotly_chart(
    forecast_fig,
    use_container_width=True
)

# =========================================================
# FUTURE FORECASTING
# =========================================================

st.subheader("🔮 Future Demand Forecast")

last_row = X.iloc[-1:].copy()

future_predictions = []

for i in range(future_steps):

    future_pred = model.predict(last_row)[0]

    future_pred = max(future_pred, 0)

    future_predictions.append(future_pred)

# =========================================================
# FUTURE TIMESTAMPS
# =========================================================

last_timestamp = df['Timestamp'].iloc[-1]

future_dates = pd.date_range(
    start=last_timestamp,
    periods=future_steps + 1,
    freq='15min'
)[1:]

future_df = pd.DataFrame({
    'Timestamp': future_dates,
    'Forecasted Demand': future_predictions
})

# =========================================================
# CONFIDENCE INTERVALS
# =========================================================

future_df['Upper Bound'] = (
    future_df['Forecasted Demand'] + 10
)

future_df['Lower Bound'] = (
    future_df['Forecasted Demand'] - 10
)

future_df['Lower Bound'] = np.maximum(
    future_df['Lower Bound'],
    0
)

# =========================================================
# FUTURE FORECAST CHART
# =========================================================

fig_future = go.Figure()

fig_future.add_trace(
    go.Scatter(
        x=future_df['Timestamp'],
        y=future_df['Forecasted Demand'],
        mode='lines+markers',
        name='Forecast'
    )
)

fig_future.add_trace(
    go.Scatter(
        x=future_df['Timestamp'],
        y=future_df['Upper Bound'],
        mode='lines',
        name='Upper Bound'
    )
)

fig_future.add_trace(
    go.Scatter(
        x=future_df['Timestamp'],
        y=future_df['Lower Bound'],
        mode='lines',
        fill='tonexty',
        name='Lower Bound'
    )
)

fig_future.update_layout(
    title='Future Ferry Demand Forecast with Confidence Bands',
    xaxis_title='Timestamp',
    yaxis_title='Forecasted Demand'
)

st.plotly_chart(
    fig_future,
    use_container_width=True
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader("⭐ Feature Importance")

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})

importance_df = importance_df.sort_values(
    'Importance',
    ascending=False
)

importance_fig = px.bar(
    importance_df,
    x='Importance',
    y='Feature',
    orientation='h',
    title='Model Feature Importance'
)

st.plotly_chart(
    importance_fig,
    use_container_width=True
)

# =========================================================
# DATASET PREVIEW
# =========================================================

st.subheader("📁 Dataset Preview")

st.dataframe(df.head(20))

# =========================================================
# DOWNLOAD BUTTON
# =========================================================

csv = future_df.to_csv(index=False)

st.download_button(
    label="📥 Download Forecast Report",
    data=csv,
    file_name='future_ferry_forecast.csv',
    mime='text/csv'
)

# =========================================================
# CONCLUSION
# =========================================================

st.markdown("---")

st.markdown("""
# Conclusion

This project developed a predictive ferry demand forecasting system using machine learning and time-series models to improve ferry scheduling, crowd management, and operational planning through proactive decision-making.
""")

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
---
Developed using Streamlit, Machine Learning, and Time-Series Forecasting
""")