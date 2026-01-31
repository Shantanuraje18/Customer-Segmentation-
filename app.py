
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --- Page Configuration ---
st.set_page_config(page_title="Customer Segmentation Pro", layout="wide")

st.title("🛍️ Customer Segmentation Dashboard")
st.markdown("Upload your retail dataset to automatically group customers into segments.")

# --- 1. File Upload Section ---
uploaded_file = st.sidebar.file_uploader("Step 1: Upload Customer CSV", type=["csv"])

if uploaded_file is not None:
    # Load and clean data
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File Uploaded Successfully!")

    # Feature Selection (Filter numeric columns only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.sidebar.markdown("### Settings")
    selected_features = st.sidebar.multiselect(
        "Select Features for Clustering",
        numeric_cols,
        default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
    )

    if len(selected_features) >= 2:
        # Preprocessing
        data_to_cluster = df[selected_features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data_to_cluster)

        # Toggle for View Mode
        view_mode = st.sidebar.radio("Select View Type", ["👨‍💼 Shop Owner View", "👨‍💻 Data Science View"])
        k_value = st.sidebar.slider("Select Number of Clusters (K)", 2, 10, 4)

        # --- 2. K-Means Clustering Logic ---
        model = KMeans(n_clusters=k_value, init='k-means++', random_state=42)
        cluster_labels = model.fit_predict(X_scaled)
        data_to_cluster['Cluster'] = cluster_labels

        # Automated Labeling based on Cluster Means
        # (Heuristic: High spending/frequency = Loyal, High Income/Low Spend = Budget, etc.)
        segment_map = {0: 'Budget Shoppers', 1: 'High Spenders', 2: 'Occasional Buyers', 3: 'Loyal Customers'}
        # For K > 4, we just use 'Segment X'
        data_to_cluster['Segment'] = data_to_cluster['Cluster'].apply(lambda x: segment_map.get(x, f"Segment {x}"))

        # --- 3. View Logic ---
        if view_mode == "👨‍💼 Shop Owner View":
            st.header("Business Summary")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Customer Distribution")
                fig_pie = px.pie(data_to_cluster, names='Segment', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.subheader("Segment Characteristics")
                avg_spending = data_to_cluster.groupby('Segment')[selected_features[0]].mean().reset_index()
                fig_bar = px.bar(avg_spending, x='Segment', y=selected_features[0], color='Segment')
                st.plotly_chart(fig_bar, use_container_width=True)

            st.write("### Data Table with Labels")
            st.dataframe(data_to_cluster.head(10))

        else:
            st.header("Technical Clustering Analytics")

            # Elbow Method Plot
            st.subheader("1. Elbow Method (Optimal K)")
            wcss = []
            for i in range(1, 11):
                km = KMeans(n_clusters=i, init='k-means++', random_state=42)
                km.fit(X_scaled)
                wcss.append(km.inertia_)

            fig_elbow, ax_elbow = plt.subplots()
            ax_elbow.plot(range(1, 11), wcss, marker='o', ls='--')
            ax_elbow.set_xlabel('Clusters')
            ax_elbow.set_ylabel('WCSS')
            st.pyplot(fig_elbow)

            # Performance Metrics
            s_score = silhouette_score(X_scaled, cluster_labels)
            st.metric("Silhouette Score", f"{s_score:.4f}")

            # PCA 2D and 3D
            st.subheader("2. Cluster Visualization (PCA)")
            pca_3d = PCA(n_components=3)
            components = pca_3d.fit_transform(X_scaled)

            # 3D Plot
            fig_3d = px.scatter_3d(
                components, x=0, y=1, z=2, color=data_to_cluster['Segment'],
                title='3D PCA Projection', labels={'0': 'PC1', '1': 'PC2', '2': 'PC3'}
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        # --- 4. Export CSV ---
        st.sidebar.markdown("---")
        csv = data_to_cluster.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="⬇️ Download Clustered Data",
            data=csv,
            file_name='clustered_customers.csv',
            mime='text/csv',
        )
    else:
        st.warning("Please select at least 2 features to begin clustering.")
else:
    st.info("Waiting for CSV file upload...")
