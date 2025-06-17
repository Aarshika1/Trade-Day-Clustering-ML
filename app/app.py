import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# --- Function: Download and preprocess data ---
@st.cache_data(show_spinner=True)
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    # Flatten columns if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    return data

# --- Function: Feature engineering ---
def feature_engineering(data):
    data["daily_return"] = (data["Close"] - data["Open"]) / data["Open"]
    data["price_range"] = (data["High"] - data["Low"]) / data["Open"]
    data["volatility"] = data[["Open", "High", "Low", "Close"]].std(axis=1)
    data["volume_change"] = data["Volume"].pct_change()
    data["volume_5day_avg"] = data["Volume"].rolling(window=5).mean()
    data["volume_vs_avg"] = data["Volume"] / data["volume_5day_avg"]
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

# --- Function: Clustering ---
def perform_clustering(data, features, n_clusters):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data["cluster"] = kmeans.fit_predict(X_scaled)
    return data, X_scaled

# --- Function: PCA for visualization ---
def get_pca(X_scaled):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca

# --- Streamlit UI ---
st.title("Stock Trading Day Clustering with K-Means")

# Sidebar inputs
st.sidebar.header("User Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker", value="NMR")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-06-01"))
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=6, value=3)

if start_date >= end_date:
    st.error("Error: End date must be after start date.")
else:
    with st.spinner(f"Downloading data for {ticker}..."):
        data = load_data(ticker, start_date, end_date)

    if data.empty:
        st.error("No data found. Please try different parameters.")
    else:
        data = feature_engineering(data)
        features = ["daily_return", "price_range", "volatility", "volume_change", "volume_vs_avg"]

        data, X_scaled = perform_clustering(data, features, n_clusters)
        X_pca = get_pca(X_scaled)

        # Show raw data preview
        st.subheader(f"Raw Data for {ticker}")
        st.dataframe(data.head())

        # Cluster distribution plot
        st.subheader("Cluster Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="cluster", data=data, ax=ax)
        ax.set_title("Number of Days per Cluster")
        st.pyplot(fig)

        # Closing price over time with cluster colors
        st.subheader("Closing Price Over Time by Cluster")
        fig, ax = plt.subplots(figsize=(14, 5))
        for cluster in sorted(data["cluster"].unique()):
            cluster_data = data[data["cluster"] == cluster]
            ax.plot(cluster_data["Date"], cluster_data["Close"], label=f"Cluster {cluster}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.legend()
        st.pyplot(fig)

        # PCA scatter plot
        st.subheader("2D PCA Projection of Clusters")
        fig, ax = plt.subplots(figsize=(10, 6))
        for cluster in sorted(data["cluster"].unique()):
            ax.scatter(X_pca[data["cluster"] == cluster, 0],
                       X_pca[data["cluster"] == cluster, 1],
                       label=f"Cluster {cluster}", alpha=0.6)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend()
        st.pyplot(fig)

        # Cluster feature averages heatmap
        st.subheader("Average Feature Values per Cluster")
        cluster_summary = data.groupby("cluster")[features].mean().round(4)
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.heatmap(cluster_summary, annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

        st.markdown("""
        ---
        ### How to use this app:
        - Select a stock ticker symbol (e.g., NMR, AAPL, MSFT)
        - Pick the date range you want to analyze
        - Choose how many clusters (market regimes) to detect
        - View cluster distributions, price behavior, and feature summaries
        """)

