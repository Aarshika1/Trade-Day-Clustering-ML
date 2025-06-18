import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

st.set_page_config(
    page_title="Stock Trading Day Clustering",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Function: Download and preprocess data ---
@st.cache_data(show_spinner=True)
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
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

# --- Function: Label clusters descriptively ---
def label_clusters(data, features):
    cluster_summary = data.groupby("cluster")[features].mean()
    labels = {}
    neutral_count = 0
    for cluster in cluster_summary.index:
        row = cluster_summary.loc[cluster]

        if row["daily_return"] > 0.02 and row["volume_vs_avg"] > 1.2:
            your_label = "📈 Bullish Spike"
        elif row["daily_return"] < -0.02 and row["volume_vs_avg"] > 1.2:
            your_label = "📉 Bearish Dump"
        elif row["daily_return"] > 0.01 and row["volatility"] > 0.02:
            your_label = "🟢 Volatile Uptrend"
        elif row["daily_return"] < -0.01 and row["volatility"] > 0.02:
            your_label = "🔻 Volatile Downtrend"
        elif row["volatility"] > 0.025 and abs(row["daily_return"]) < 0.005:
            your_label = "⚠️ Sideways with High Volatility"
        elif row["daily_return"] > 0.005:
            your_label = "🟢 Mild Gain"
        elif row["daily_return"] < -0.005:
            your_label = "🔴 Mild Loss"
        else:
            neutral_count += 1
            your_label = f"⚪ Neutral Day {neutral_count}"

        labels[cluster] = f"Cluster {cluster + 1}: {your_label}"
    return labels

# --- Helper: Remove duplicate legend entries ---
def unique_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = dict()
    for handle, label in zip(handles, labels):
        if label not in unique and label != '_nolegend_':
            unique[label] = handle
    ax.legend_.remove() if hasattr(ax, 'legend_') and ax.legend_ else None
    if unique:
        ax.legend(unique.values(), unique.keys())

def remove_emoji(text):
    return re.sub(r'[^\w\s:.,-]', '', text)

# --- Streamlit UI ---
st.title("📊 Stock Trading Day Clustering with K-Means")
st.markdown("""
This app groups trading days of a stock into clusters based on price movement and volume patterns.
It helps identify distinct types of trading days like spikes, dumps, trends, or neutral days.
""")

# Sidebar inputs
st.sidebar.header("User Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker", value="NMR")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-06-01"))
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)
run_button = st.sidebar.button("Run Clustering")

if start_date >= end_date:
    st.error("Error: End date must be after start date.")
elif run_button:
    with st.spinner(f"Downloading data for {ticker}..."):
        data = load_data(ticker, start_date, end_date)

    if data.empty:
        st.error("No data found. Please try different parameters.")
    else:
        data = feature_engineering(data)
        features = ["daily_return", "price_range", "volatility", "volume_change", "volume_vs_avg"]

        data, X_scaled = perform_clustering(data, features, n_clusters)
        X_pca = get_pca(X_scaled)
        cluster_labels = label_clusters(data, features)
        data["cluster_label"] = data["cluster"].map(cluster_labels)
        cluster_summary = data.groupby("cluster_label")[features].mean().round(4)

        # Raw data
        st.subheader("📋 Raw Data & Features")
        st.dataframe(data.head())

        # Cluster summaries
        st.subheader("🗂️ Cluster Summaries")
        for label, row in cluster_summary.iterrows():
            st.markdown(f"### {label}")
            st.markdown(
                f"- **Average Return:** {row['daily_return']:.4f}\n"
                f"- **Price Range:** {row['price_range']:.4f}\n"
                f"- **Volatility:** {row['volatility']:.4f}\n"
                f"- **Volume Change:** {row['volume_change']:.4f}\n"
                f"- **Volume vs Avg:** {row['volume_vs_avg']:.4f}"
            )

        # Cluster distribution
        st.subheader("📊 Cluster Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="cluster_label", data=data, ax=ax, order=cluster_summary.index)
        ax.set_title("Number of Days per Cluster")
        ax.set_xlabel("Cluster Label")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=45)
        clean_labels = [remove_emoji(label) for label in cluster_summary.index]
        ax.set_xticklabels(clean_labels)
        st.pyplot(fig)

        # Closing price over time
        st.subheader("📈 Closing Price Over Time by Cluster")
        fig, ax = plt.subplots(figsize=(14, 5))
        for cluster in sorted(data["cluster"].unique()):
            cluster_data = data[data["cluster"] == cluster]
            ax.plot(cluster_data["Date"], cluster_data["Close"], label=remove_emoji(cluster_labels[cluster]))
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        unique_legend(ax)
        st.pyplot(fig)

        # PCA plot
        st.subheader("🔍 PCA Projection of Clusters")
        fig, ax = plt.subplots(figsize=(10, 6))
        for cluster in sorted(data["cluster"].unique()):
            ax.scatter(
                X_pca[data["cluster"] == cluster, 0],
                X_pca[data["cluster"] == cluster, 1],
                label=remove_emoji(cluster_labels[cluster]),
                alpha=0.6
            )
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        unique_legend(ax)
        st.pyplot(fig)

        # Heatmap
        st.subheader("🔥 Feature Heatmap per Cluster")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(cluster_summary, annot=True, cmap="YlGnBu", ax=ax)
        ax.set_title("Feature Heatmap per Cluster")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Cluster")
        clean_labels = [remove_emoji(label) for label in cluster_summary.index]
        ax.set_yticklabels(clean_labels, rotation=0)
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("""
        ### How to use this app:
        - Enter a stock ticker symbol (e.g., NMR, AAPL).
        - Select the date range to analyze.
        - Choose the number of clusters (types of trading days).
        - Click **Run Clustering** to explore distinct trading day patterns.
        """)
