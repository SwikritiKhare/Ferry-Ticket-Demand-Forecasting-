import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Ferry Demand Forecasting Dashboard",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================

st.markdown("""
# 🚢 Ferry Ticket Demand Forecasting Dashboard
### Smart Transportation Analytics System
""")

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("ferry_data.csv")

# ==========================================
# DATA CLEANING
# ==========================================

df.columns = df.columns.str.strip()

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df = df.sort_values("Timestamp")

# Reduce memory usage
df = df.tail(2000)

# ==========================================
# KPI SECTION
# ==========================================

total_sales = int(df["Sales Count"].sum())

avg_sales = round(df["Sales Count"].mean(), 2)

max_sales = int(df["Sales Count"].max())

min_sales = int(df["Sales Count"].min())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Ticket Sales", total_sales)

col2.metric("Average Demand", avg_sales)

col3.metric("Peak Demand", max_sales)

col4.metric("Minimum Demand", min_sales)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Forecast Settings")

future_steps = st.sidebar.slider(
    "Select Forecast Horizon",
    5,
    50,
    20
)

# ==========================================
# DEMAND CHART
# ==========================================

st.subheader("Historical Ferry Demand")

fig = px.line(
    df,
    x="Timestamp",
    y="Sales Count",
    title="Ticket Demand Over Time"
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MOVING AVERAGE FORECAST
# ==========================================

st.subheader("Future Demand Forecast")

last_value = df["Sales Count"].rolling(10).mean().iloc[-1]

future_forecast = np.repeat(last_value, future_steps)

future_dates = pd.date_range(
    start=df["Timestamp"].iloc[-1],
    periods=future_steps + 1,
    freq="15min"
)[1:]

forecast_df = pd.DataFrame({
    "Timestamp": future_dates,
    "Forecast": future_forecast
})

fig2 = go.Figure()

# Actual Data
fig2.add_trace(go.Scatter(
    x=df["Timestamp"],
    y=df["Sales Count"],
    mode='lines',
    name='Actual Demand'
))

# Forecast Data
fig2.add_trace(go.Scatter(
    x=forecast_df["Timestamp"],
    y=forecast_df["Forecast"],
    mode='lines',
    name='Forecast Demand'
))

st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# CONFIDENCE INTERVAL
# ==========================================

st.subheader("Confidence Interval Visualization")

forecast_df["Upper Bound"] = forecast_df["Forecast"] + 20

forecast_df["Lower Bound"] = forecast_df["Forecast"] - 20

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=forecast_df["Timestamp"],
    y=forecast_df["Forecast"],
    mode='lines',
    name='Forecast'
))

fig3.add_trace(go.Scatter(
    x=forecast_df["Timestamp"],
    y=forecast_df["Upper Bound"],
    mode='lines',
    name='Upper Bound'
))

fig3.add_trace(go.Scatter(
    x=forecast_df["Timestamp"],
    y=forecast_df["Lower Bound"],
    mode='lines',
    name='Lower Bound'
))

st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# DATA TABLE
# ==========================================

st.subheader("Forecast Data")

st.dataframe(forecast_df)

# ==========================================
# DOWNLOAD BUTTON
# ==========================================

csv = forecast_df.to_csv(index=False).encode('utf-8')

st.download_button(
    "Download Forecast CSV",
    csv,
    "forecast.csv",
    "text/csv"
)

# ==========================================
# PROJECT SUMMARY
# ==========================================

st.markdown("""
---
## 📌 Project Summary

This project predicts short-term ferry ticket demand using time-series analytics and forecasting techniques.  
The dashboard helps support proactive ferry scheduling, crowd management, and transportation planning through interactive visualizations and future demand forecasting.
""")
