"""
RiskGuard Monitoring Dashboard.

Real-time visualization of fraud detection results:
- Transaction risk heatmap
- Model performance metrics
- Audit trail viewer
- Per-merchant risk breakdown

Usage:
    streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yaml
import os

st.set_page_config(
    page_title="RiskGuard — Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data
def load_data():
    """Load scored transaction data."""
    df = pd.read_parquet("data/transactions.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_resource
def load_config():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


def main():
    st.title("🛡️ RiskGuard — Fraud Spike Detection Dashboard")
    st.markdown("*Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager*")

    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("No data found. Run `python -m src.data_generator.generator` first.")
        return

    config = load_config()

    # --- Sidebar filters ---
    st.sidebar.header("Filters")
    merchants = st.sidebar.multiselect(
        "Merchants",
        options=sorted(df["merchant_id"].unique()),
        default=sorted(df["merchant_id"].unique())[:5],
    )
    date_range = st.sidebar.date_input(
        "Date range",
        value=(df["timestamp"].min().date(), df["timestamp"].max().date()),
    )

    # Filter data
    filtered = df[
        (df["merchant_id"].isin(merchants))
        & (df["timestamp"].dt.date >= date_range[0])
        & (df["timestamp"].dt.date <= date_range[1])
    ]

    # --- KPI Row ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(filtered):,}")
    with col2:
        n_fraud = filtered["is_fraud"].sum()
        st.metric("Fraud Transactions", f"{n_fraud:,}")
    with col3:
        fraud_rate = filtered["is_fraud"].mean() * 100
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    with col4:
        st.metric("Merchants", f"{filtered['merchant_id'].nunique()}")

    st.markdown("---")

    # --- Transaction Volume Over Time ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Transaction Volume Over Time")
        daily = filtered.set_index("timestamp").resample("D").agg(
            total=("amount", "count"),
            fraud=("is_fraud", "sum"),
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["timestamp"], y=daily["total"],
            name="Total", line=dict(color="#3b82f6"),
        ))
        fig.add_trace(go.Bar(
            x=daily["timestamp"], y=daily["fraud"],
            name="Fraud", marker_color="#ef4444", opacity=0.7,
        ))
        fig.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Fraud Type Distribution")
        fraud_df = filtered[filtered["is_fraud"]]
        if len(fraud_df) > 0:
            fig = px.pie(
                fraud_df, names="fraud_type",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(height=350, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No fraud in selected range")

    # --- Merchant Risk Heatmap ---
    st.subheader("Merchant Risk Heatmap")
    merchant_daily = filtered.groupby([
        filtered["timestamp"].dt.date,
        "merchant_id"
    ]).agg(
        txn_count=("amount", "count"),
        fraud_count=("is_fraud", "sum"),
    ).reset_index()
    merchant_daily["fraud_rate"] = merchant_daily["fraud_count"] / merchant_daily["txn_count"]

    pivot = merchant_daily.pivot_table(
        index="merchant_id", columns="timestamp",
        values="fraud_rate", fill_value=0,
    )
    if not pivot.empty:
        fig = px.imshow(
            pivot.values,
            labels=dict(x="Date", y="Merchant", color="Fraud Rate"),
            y=pivot.index.tolist(),
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --- Hourly Pattern ---
    st.subheader("Hourly Transaction Pattern (Fraud vs Normal)")
    hourly = filtered.groupby([filtered["timestamp"].dt.hour, "is_fraud"]).size().reset_index()
    hourly.columns = ["hour", "is_fraud", "count"]
    hourly["label"] = hourly["is_fraud"].map({True: "Fraud", False: "Normal"})

    fig = px.bar(
        hourly, x="hour", y="count", color="label",
        barmode="group",
        color_discrete_map={"Normal": "#3b82f6", "Fraud": "#ef4444"},
    )
    fig.update_layout(height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # --- Raw Data Explorer ---
    st.subheader("Transaction Explorer")
    show_fraud_only = st.checkbox("Show fraud only")
    display_df = filtered[filtered["is_fraud"]] if show_fraud_only else filtered
    st.dataframe(
        display_df[["txn_id", "timestamp", "merchant_id", "amount", "card_id", "is_fraud", "fraud_type"]]
        .sort_values("timestamp", ascending=False)
        .head(200),
        use_container_width=True,
    )


main()