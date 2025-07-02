import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Fundraising Dashboard", layout="wide")

# Data
data = {
    'date': ['2025-05-20', '2025-05-27', '2025-05-30', '2025-06-01', '2025-06-02', 
             '2025-06-04', '2025-06-05', '2025-06-07', '2025-06-19', '2025-06-21', 
             '2025-06-22', '2025-06-24', '2025-06-25', '2025-06-26', '2025-06-27', 
             '2025-06-29', '2025-06-30', '2025-07-01', '2025-07-02'],
    'supporters': [0, 1166, 1298, 1371, 1488, 1796, 2000, 2206, 3473, 3533, 3557, 3612, 3646, 3694, 3942, 4015, 4111, 4291, 4522]
}

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df['days_since_start'] = (df['date'] - df['date'].iloc[0]).dt.days

# Campaign parameters
GOAL = 5000
END_DATE = datetime(2025, 7, 6)
TODAY = datetime(2025, 7, 1)
DAYS_REMAINING = (END_DATE - TODAY).days

# Current stats
current_supporters = df['supporters'].iloc[-1]
needed_daily = (GOAL - current_supporters) / DAYS_REMAINING if DAYS_REMAINING > 0 else 0

# Calculate trends
# 1. Overall trend (linear regression on all data)
X_all = df['days_since_start'].values.reshape(-1, 1)
y_all = df['supporters'].values
overall_model = LinearRegression()
overall_model.fit(X_all, y_all)

# 2. Recent trend (linear regression on last 4 data points)
recent_df = df.tail(4)
X_recent = recent_df['days_since_start'].values.reshape(-1, 1)
y_recent = recent_df['supporters'].values
recent_model = LinearRegression()
recent_model.fit(X_recent, y_recent)

# Calculate end day once
end_day = (END_DATE - df['date'].iloc[0]).days

# Header
st.title("🎯 JETZT Fundraising Dashboard")
st.caption("Tracking supporter growth")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Current Supporters", f"{current_supporters:,}")
with col2:
    st.metric("Goal", f"{GOAL:,}", f"{GOAL - current_supporters:,} to go")
with col3:
    st.metric("Days Left", DAYS_REMAINING, f"Need {needed_daily:.0f}/day")
with col4:
    overall_final = overall_model.predict([[end_day]])[0]
    st.metric("Overall Trend Prediction", f"{overall_final:,.0f}", f"{overall_final - GOAL:+,.0f} vs goal")
with col5:
    recent_final = recent_model.predict([[end_day]])[0]
    st.metric("Recent Trend Prediction", f"{recent_final:,.0f}", f"{recent_final - GOAL:+,.0f} vs goal")

# Generate future predictions for chart
future_days = np.arange(0, end_day + 1)
X_future = future_days.reshape(-1, 1)

# Predictions
overall_pred = overall_model.predict(X_future)
recent_pred = recent_model.predict(X_future)

# Create chart
fig = go.Figure()

# Actual data
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['supporters'],
    mode='markers+lines',
    name='Actual Data',
    line=dict(color='blue', width=3),
    marker=dict(size=8, color='blue')
))

# Overall trend
future_dates = [df['date'].iloc[0] + timedelta(days=int(d)) for d in future_days]
fig.add_trace(go.Scatter(
    x=future_dates,
    y=overall_pred,
    mode='lines',
    name='Overall Trend',
    line=dict(color='gold', dash='dash', width=3)
))

# Recent trend (only from recent data point forward)
recent_start_day = recent_df['days_since_start'].iloc[0]
recent_start_date = recent_df['date'].iloc[0]

# Only show recent trend from the recent period forward
recent_future_days = future_days[future_days >= recent_start_day]
recent_future_dates = [df['date'].iloc[0] + timedelta(days=int(d)) for d in recent_future_days]
recent_future_pred = recent_model.predict(recent_future_days.reshape(-1, 1))

fig.add_trace(go.Scatter(
    x=recent_future_dates,
    y=recent_future_pred,
    mode='lines',
    name='Recent Trend (last 4 days)',
    line=dict(color='green', dash='dash', width=3)
))

# Goal line - make it really visible
fig.add_hline(y=GOAL, line_dash="solid", line_color="red", line_width=4)

# End date line
fig.add_shape(
    type="line",
    x0=END_DATE, x1=END_DATE,
    y0=0, y1=GOAL*1.2,
    line=dict(color="gray", dash="dot"),
)

fig.update_layout(
    title="Will They Reach Their Goal?",
    xaxis_title="",
    yaxis_title="Supporters",
    height=500,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.1,
        xanchor="center",
        x=0.5
    )
)

st.plotly_chart(fig, use_container_width=True)

