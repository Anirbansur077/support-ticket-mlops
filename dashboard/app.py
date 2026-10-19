import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Ticket Monitoring Dashboard", layout="wide")

st.title("Support Ticket Monitoring Dashboard")

@st.cache_data
def load_data():
    train = pd.read_parquet('data/processed/train.parquet')
    val = pd.read_parquet('data/processed/val.parquet')
    train['split'] = 'train'
    val['split'] = 'val'
    return pd.concat([train, val], ignore_index=True)

df = load_data()

col1, col2, col3 = st.columns(3)
col1.metric("Total Tickets", len(df))
col2.metric("Escalation Rate", f"{df['escalated'].mean():.1%}")
col3.metric("Date Range", f"{df['created_at'].min().date()} to {df['created_at'].max().date()}")

st.subheader("Ticket Volume Over Time")
daily_volume = df.set_index('created_at').resample('W').size().reset_index(name='count')
fig1 = px.line(daily_volume, x='created_at', y='count', title="Weekly Ticket Volume")
st.plotly_chart(fig1, use_container_width=True)

col4, col5 = st.columns(2)

with col4:
    st.subheader("Priority Distribution")
    priority_counts = df['priority'].value_counts().reset_index()
    fig2 = px.bar(priority_counts, x='priority', y='count')
    st.plotly_chart(fig2, use_container_width=True)

with col5:
    st.subheader("Escalation Rate by Type")
    esc_by_type = df.groupby('type')['escalated'].mean().reset_index()
    fig3 = px.bar(esc_by_type, x='type', y='escalated')
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("Drift Report")
st.info("Latest drift check: run `python src/monitoring/drift_check.py` to regenerate.")
with open('docs/drift_report.html', 'r') as f:
    drift_html = f.read()
st.components.v1.html(drift_html, height=600, scrolling=True)