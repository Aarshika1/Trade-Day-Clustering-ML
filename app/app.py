import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re
import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import ta

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
    data["rsi"] = ta.momentum.RSIIndicator(close=data["Close"]).rsi()
    macd = ta.trend.MACD(close=data["Close"])
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

# --- Function: Clustering ---
def perform_clustering(data, features, n_clusters):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[features])
    pca_all = PCA(n_components=5)
    X_pca_5 = pca_all.fit_transform(X_scaled)
    X_pca_vis = get_pca(X_scaled)
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

    for cluster in cluster_summary.index:
        row = cluster_summary.loc[cluster]

        # Determine volume level
        if row["volume_vs_avg"] > 1.3:
            volume_level = "High Volume"
        elif row["volume_vs_avg"] < 0.7:
            volume_level = "Low Volume"
        else:
            volume_level = "Moderate Volume"

        # Determine volume trend (change)
        if row["volume_change"] > 0.05:
            volume_trend = "Volume Trend Up"
        elif row["volume_change"] < -0.05:
            volume_trend = "Volume Trend Down"
        else:
            volume_trend = "Volume Trend Stable"

        # Determine volatility label
        if row["volatility"] > 0.04:
            volatility = "High Volatility"
        elif row["volatility"] > 0.02:
            volatility = "Moderate Volatility"
        else:
            volatility = "Low Volatility"

        # Determine return label
        if row["daily_return"] > 0.01:
            return_label = "Moderate Gain"
        elif row["daily_return"] < -0.01:
            return_label = "Moderate Loss"
        else:
            return_label = "Neutral Return"

        # Compose the cluster label
        cluster_label = f"{volume_trend}, {volume_level}, {volatility}, {return_label}"

        labels[cluster] = f"Cluster {cluster + 1}: {cluster_label}"

    return labels

def remove_emoji(text):
    return re.sub(r'[^\w\s:.,-]', '', text)

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

# --- Streamlit UI ---
st.title("📊 Stock Trading Day Clustering")
st.markdown("""
This app uses **unsupervised learning (K-Means, DBSCAN and GMM)** to cluster trading days for a selected stock, using return, volatility, and volume-based features.
""")

# Sidebar inputs
st.sidebar.header("User Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker", value="NMR")
st.sidebar.markdown("[🔗 Lookup stock tickers on Yahoo Finance](https://finance.yahoo.com/lookup)")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-06-01"))
clustering_method = st.sidebar.selectbox("Clustering Method", ["KMeans", "DBSCAN", "GMM"])
if clustering_method == "KMeans":
    st.sidebar.info("K-Means partitions data into a user-defined number of clusters by minimizing distances to cluster centers.")
elif clustering_method == "DBSCAN":
    st.sidebar.info("DBSCAN groups points that are close together and marks outliers as noise. No need to predefine number of clusters.")
elif clustering_method == "GMM":
    st.sidebar.info("Gaussian Mixture Models assume the data is generated from a mixture of several Gaussian distributions.")
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3) if clustering_method != "DBSCAN" else None
if clustering_method == "DBSCAN":
    eps = st.sidebar.slider("DBSCAN: eps", 0.1, 5.0, 1.0, 0.1)
    st.sidebar.info("eps controls how close points should be to each other to be considered neighbors. Smaller values make DBSCAN stricter.")
    min_samples = st.sidebar.slider("DBSCAN: min_samples", 2, 10, 3)
    st.sidebar.info("min_samples is the minimum number of nearby points required to form a cluster. Higher values require denser regions.")
else:
    eps = None
    min_samples = None

run_button = st.sidebar.button("Run Analysis")

# --- Handle the logic for button and data loading ---
if run_button:
    st.session_state.run_clustering = True

if start_date >= end_date:
    st.error("Error: End date must be after start date.")
elif st.session_state.get("run_clustering", False) or "data" in st.session_state:
    # Only run clustering if requested or if results already exist
    if st.session_state.get("run_clustering", False):
        with st.spinner(f"Downloading data for {ticker}..."):
            data = load_data(ticker, start_date, end_date)

        if data.empty:
            st.error("No data found. Please try different parameters.")
            st.session_state.run_clustering = False
        else:
            with st.spinner("🔄 Clustering in progress..."):
                data = feature_engineering(data)
            features = ["daily_return", "price_range", "volatility", "volume_change", "volume_vs_avg", "rsi", "macd", "macd_signal"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data[features])
            pca_all = PCA(n_components=5)
            X_pca_5 = pca_all.fit_transform(X_scaled)
            X_pca_vis = get_pca(X_scaled)

            if clustering_method == "KMeans":
                model = KMeans(n_clusters=n_clusters, random_state=42)
                data["cluster"] = model.fit_predict(X_pca_5)
            elif clustering_method == "DBSCAN":
                model = DBSCAN(eps=eps, min_samples=min_samples)
                data["cluster"] = model.fit_predict(X_pca_5)
                data["cluster"] = data["cluster"] + 1  # To avoid -1 for noise
            elif clustering_method == "GMM":
                model = GaussianMixture(n_components=n_clusters, random_state=42)
                data["cluster"] = model.fit_predict(X_pca_5)

            if clustering_method == "DBSCAN" and min_samples == 3:
                with st.expander("Why do some clusters have only 1 point?"):
                    st.markdown("""
                    Even when `min_samples = 3`, DBSCAN can create clusters with just **1 point**.

                    This happens when:
                    - A point has enough neighbors to be a **core point**
                    - But **those neighbors aren't close to each other**, so they can't join the cluster

                    So the cluster is formed by **just the core point itself**.
                    """)

            cluster_labels = label_clusters(data, features)
            data["cluster_label"] = data["cluster"].map(cluster_labels)
            data["cluster_name"] = data["cluster_label"].apply(remove_emoji)
            short_labels = {cluster: f"Cluster {cluster + 1}" for cluster in cluster_labels.keys()}
            cluster_summary = data.groupby("cluster_label")[features].mean().round(4)
            data["cluster_short_label"] = data["cluster"].map(short_labels)

            st.session_state.update({
                "data": data,
                "X_scaled": X_scaled,
                "X_pca": X_pca_vis,
                "cluster_labels": cluster_labels,
                "cluster_summary": cluster_summary,
                "features": features,
                "short_labels": short_labels
            })
            st.session_state.run_clustering = False

    # Always use session_state for display
    if "data" in st.session_state:
        st.markdown("---")
        st.subheader("🧾 View Raw Data")
        if "show_data" not in st.session_state:
            st.session_state["show_data"] = False

        show_data = st.checkbox("Show Raw Data Table", key="show_data")

        if show_data:
            st.subheader("📋 Raw Data & Features")
            st.dataframe(st.session_state["data"].head())

        data = st.session_state["data"]
        X_scaled = st.session_state["X_scaled"]
        X_pca = st.session_state["X_pca"]
        cluster_labels = st.session_state["cluster_labels"]
        cluster_summary = st.session_state["cluster_summary"]
        features = st.session_state["features"]
        short_labels = st.session_state.get("short_labels", {cluster: f"Cluster {cluster + 1}" for cluster in cluster_labels.keys()})

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
        st.markdown("Number of trading days in each cluster.")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x="cluster_short_label", data=data, ax=ax,
                order=[short_labels[cluster] for cluster in sorted(data["cluster"].unique())])
        ax.set_title("Number of Days per Cluster")
        ax.set_xlabel("Cluster Label")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        st.download_button("Download Cluster Distribution", data=fig_to_bytes(fig), file_name="cluster_distribution.png", mime="image/png")

        # PCA plot
        st.subheader("🔍 PCA Projection of Clusters")
        st.markdown("Visualizes clusters in reduced 2D space using PCA. Clusters closer together may share similar trading behavior.")
        fig, ax = plt.subplots(figsize=(8, 5))
        for cluster in sorted(data["cluster"].unique()):
            ax.scatter(
                X_pca[data["cluster"] == cluster, 0],
                X_pca[data["cluster"] == cluster, 1],
                label=short_labels[cluster],
                alpha=0.6
            )
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend(title="Cluster")
        st.pyplot(fig)
        st.download_button("Download PCA Projection", data=fig_to_bytes(fig), file_name="pca_projection.png", mime="image/png")

        # Feature Pairplots
        st.subheader("📊 Feature Correlation by Cluster")
        st.markdown("Understand how feature values relate to each other across clusters.")
        pairplot_fig = sns.pairplot(
            data[features + ["cluster_name"]],
            hue="cluster_name",
            plot_kws={'alpha': 0.5},
            height=2.2
        )
        # Set legend title for pairplot
        if pairplot_fig._legend is not None:
            pairplot_fig._legend.set_title("Cluster")
        st.pyplot(pairplot_fig)
        st.download_button("Download Feature Correlation", data=fig_to_bytes(pairplot_fig.fig), file_name="feature_correlation.png", mime="image/png")

        # Table: Raw average feature values per cluster
        st.subheader("📋 Average Feature Values per Cluster")
        st.markdown("This table shows the raw (non-normalized) average values of each feature for every cluster.")
        st.dataframe(cluster_summary)
        csv_avg = cluster_summary.to_csv().encode()
        st.download_button("Download Cluster Feature Averages", data=csv_avg, file_name="cluster_feature_averages.csv", mime="text/csv")

        # Heatmap - Normalized
        st.subheader("🔥 Feature Heatmap per Cluster (Normalized)")
        st.markdown("Average value of each feature across clusters (normalized and color-scaled). Helps interpret key differentiating factors.")
        fig_norm, ax_norm = plt.subplots(figsize=(8, 5))

        scaler = MinMaxScaler()
        cluster_summary_norm = pd.DataFrame(
            scaler.fit_transform(cluster_summary),
            columns=cluster_summary.columns,
            index=cluster_summary.index
        )

        sns.heatmap(cluster_summary_norm, annot=True, cmap="YlGnBu", ax=ax_norm)
        ax_norm.set_title("Normalized Feature Heatmap per Cluster")
        ax_norm.set_xlabel("Feature")
        ax_norm.set_ylabel("Cluster")
        clean_labels_norm = [remove_emoji(label) for label in cluster_summary.index]
        ax_norm.set_yticklabels(clean_labels_norm, rotation=0)
        st.pyplot(fig_norm)
        st.download_button("Download Normalized Feature Heatmap", data=fig_to_bytes(fig_norm), file_name="feature_heatmap_normalized.png", mime="image/png")

        # Download clustered dataset
        st.subheader("💾 Download Clustered Data")
        csv = data.to_csv(index=False).encode()
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="clustered_trading_days.csv",
            mime="text/csv"
        )
