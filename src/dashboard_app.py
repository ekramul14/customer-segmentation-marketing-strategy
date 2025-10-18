import pandas as pd
import joblib
import numpy as np
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .cluster-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Cluster descriptions
CLUSTER_DESCRIPTIONS = {
    0: {"name": "Transactors", "description": "Low balance, low cash advance usage, moderate full payment rate. Low-risk customers.", "color": "#1f77b4"},
    1: {"name": "Revolvers", "description": "Moderate balance, low purchases, very low full payment rate. Using credit card as a loan.", "color": "#ff7f0e"},
    2: {"name": "High-Value Customers", "description": "High balance, high purchases, moderate full payment rate. Active purchasers.", "color": "#2ca02c"},
    3: {"name": "Low Tenure Customers", "description": "Very low balance and activity, low tenure. New or inactive customers.", "color": "#d62728"},
    4: {"name": "VIP/Premium Customers", "description": "Very high balance and purchases, high full payment rate. Top-tier customers.", "color": "#9467bd"},
    5: {"name": "Convenience Users", "description": "Low balance, highest full payment rate. Pay in full every month.", "color": "#8c564b"},
    6: {"name": "Cash Advance Seekers", "description": "High balance and very high cash advances. High-risk segment.", "color": "#e377c2"},
}

# Feature names for display
NUMERIC_FEATURES = [
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE"
]

# Load data and models
@st.cache_resource
def load_models():
    """Load all required models and data"""
    try:
        imputer = joblib.load("models/imputer.joblib")
        scaler = joblib.load("models/scaler.joblib")
        kmeans = joblib.load("models/kmeans.joblib")
        pca = joblib.load("models/pca.joblib")
        return imputer, scaler, kmeans, pca
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

@st.cache_data
def load_data():
    """Load processed data"""
    try:
        profiles = pd.read_csv("data/processed/cluster_profiles.csv")
        assignments = pd.read_csv("data/processed/cluster_assignments.csv")
        features = pd.read_csv("data/processed/features.csv")
        return profiles, assignments, features
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Load everything
imputer, scaler, kmeans, pca = load_models()
profiles, assignments, features = load_data()

# Check if data loaded successfully
if imputer is None or scaler is None or kmeans is None or pca is None or profiles is None or assignments is None or features is None:
    st.error("⚠️ Required data files not found. Please run the pipeline: ingest → clean → train → infer")
    st.stop()

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/000000/segmentation.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Data Exploration", "📤 Batch Prediction", "🎯 Single Prediction", "🔍 Cluster Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This interactive dashboard showcases an end-to-end customer segmentation system "
    "using K-Means clustering on credit card customer data."
)
st.sidebar.markdown("### Model Info")
st.sidebar.metric("Number of Clusters", kmeans.n_clusters)
st.sidebar.metric("Total Customers", len(assignments))
st.sidebar.metric("Features", len(NUMERIC_FEATURES))

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">📊 Customer Segmentation Dashboard</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Welcome to the Interactive Customer Segmentation System

    This application demonstrates a **production-ready machine learning system** that segments credit card customers
    into actionable business groups using K-Means clustering.
    """)

    # Key features
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="cluster-card">', unsafe_allow_html=True)
        st.markdown("#### 📤 Batch Prediction")
        st.write("Upload your customer data CSV and get instant segmentation for all customers with downloadable results.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="cluster-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Single Prediction")
        st.write("Input individual customer features and see their segment assignment with AI explainability (SHAP values).")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="cluster-card">', unsafe_allow_html=True)
        st.markdown("#### 🔍 Cluster Analysis")
        st.write("Deep dive into each customer segment with interactive visualizations and business insights.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Cluster overview
    st.markdown('<p class="sub-header">Customer Segments Overview</p>', unsafe_allow_html=True)

    # Calculate cluster sizes
    cluster_counts = assignments['cluster'].value_counts().sort_index()

    col1, col2 = st.columns([1, 1])

    with col1:
        # Pie chart of cluster distribution
        fig = go.Figure(data=[go.Pie(
            labels=[CLUSTER_DESCRIPTIONS[i]["name"] for i in cluster_counts.index],
            values=cluster_counts.values,
            marker=dict(colors=[CLUSTER_DESCRIPTIONS[i]["color"] for i in cluster_counts.index]),
            hole=0.4
        )])
        fig.update_layout(
            title="Customer Distribution by Segment",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📋 Segment Descriptions")
        for cluster_id in sorted(CLUSTER_DESCRIPTIONS.keys()):
            if cluster_id in cluster_counts.index:
                info = CLUSTER_DESCRIPTIONS[cluster_id]
                count = cluster_counts[cluster_id]
                percentage = (count / len(assignments)) * 100

                st.markdown(f"""
                **{info['name']}** ({count:,} customers - {percentage:.1f}%)
                <span style="color: {info['color']};">●</span> {info['description']}
                """, unsafe_allow_html=True)
                st.markdown("---")

    st.markdown("""
    ### 🚀 Quick Start Guide

    1. **Explore Data** - Visualize customer segments in 2D/3D space
    2. **Batch Prediction** - Upload CSV files for bulk customer segmentation
    3. **Single Prediction** - Test individual customer profiles with explainability
    4. **Cluster Analysis** - Compare segments and understand differences

    ### 📊 Dataset Features

    This model uses **17 customer features** including:
    - **Balance & Payments**: Account balance, payment amounts, credit limits
    - **Purchase Behavior**: Total purchases, one-off vs installment purchases
    - **Cash Advances**: Cash advance amounts and frequency
    - **Activity Metrics**: Transaction counts, payment frequency, tenure
    """)

# ============================================================================
# DATA EXPLORATION PAGE
# ============================================================================
elif page == "📊 Data Exploration":
    st.markdown('<p class="main-header">📊 Data Exploration</p>', unsafe_allow_html=True)

    # Prepare data for visualization
    X = features.values

    # 2D PCA visualization
    st.markdown('<p class="sub-header">2D Cluster Visualization (PCA)</p>', unsafe_allow_html=True)

    xy = pca.transform(X)
    plot_df = pd.DataFrame({
        "PC1": xy[:, 0],
        "PC2": xy[:, 1],
        "Cluster": assignments["cluster"],
        "Cluster_Name": [CLUSTER_DESCRIPTIONS[c]["name"] for c in assignments["cluster"]]
    })

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster_Name",
        color_discrete_map={CLUSTER_DESCRIPTIONS[i]["name"]: CLUSTER_DESCRIPTIONS[i]["color"] for i in CLUSTER_DESCRIPTIONS.keys()},
        title="Customer Segments in 2D PCA Space",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
        hover_data=["Cluster"]
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # 3D PCA visualization
    st.markdown('<p class="sub-header">3D Interactive Cluster Visualization</p>', unsafe_allow_html=True)

    # Create 3D PCA
    from sklearn.decomposition import PCA as PCA3D
    pca_3d = PCA3D(n_components=3)
    xyz = pca_3d.fit_transform(X)

    plot_3d_df = pd.DataFrame({
        "PC1": xyz[:, 0],
        "PC2": xyz[:, 1],
        "PC3": xyz[:, 2],
        "Cluster": assignments["cluster"],
        "Cluster_Name": [CLUSTER_DESCRIPTIONS[c]["name"] for c in assignments["cluster"]]
    })

    fig_3d = px.scatter_3d(
        plot_3d_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster_Name",
        color_discrete_map={CLUSTER_DESCRIPTIONS[i]["name"]: CLUSTER_DESCRIPTIONS[i]["color"] for i in CLUSTER_DESCRIPTIONS.keys()},
        title="Customer Segments in 3D PCA Space (Interactive - Rotate & Zoom)",
        labels={"PC1": "PC 1", "PC2": "PC 2", "PC3": "PC 3"},
        hover_data=["Cluster"]
    )
    fig_3d.update_traces(marker=dict(size=3, opacity=0.6))
    fig_3d.update_layout(height=700)
    st.plotly_chart(fig_3d, use_container_width=True)

    st.info("💡 **Tip**: Use your mouse to rotate, zoom, and pan the 3D visualization!")

    # Feature distributions
    st.markdown('<p class="sub-header">Feature Distributions by Cluster</p>', unsafe_allow_html=True)

    selected_feature = st.selectbox(
        "Select a feature to explore",
        NUMERIC_FEATURES,
        index=0
    )

    # Get original data with clusters
    original_data = pd.read_csv("data/processed/live_raw.csv")
    if "CUST_ID" in original_data.columns:
        original_data = original_data.drop("CUST_ID", axis=1)

    if selected_feature in original_data.columns:
        original_data["cluster"] = assignments["cluster"]
        original_data["Cluster_Name"] = [CLUSTER_DESCRIPTIONS[c]["name"] for c in assignments["cluster"]]

        fig_box = px.box(
            original_data,
            x="Cluster_Name",
            y=selected_feature,
            color="Cluster_Name",
            color_discrete_map={CLUSTER_DESCRIPTIONS[i]["name"]: CLUSTER_DESCRIPTIONS[i]["color"] for i in CLUSTER_DESCRIPTIONS.keys()},
            title=f"{selected_feature} Distribution Across Clusters"
        )
        fig_box.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # Cluster profiles table
    st.markdown('<p class="sub-header">Cluster Profiles (Average Values)</p>', unsafe_allow_html=True)

    # Add cluster names to profiles
    profiles_display = profiles.copy()
    profiles_display.insert(1, 'Cluster_Name', [CLUSTER_DESCRIPTIONS[c]["name"] for c in profiles['cluster']])

    st.dataframe(profiles_display, use_container_width=True, height=400)

# ============================================================================
# BATCH PREDICTION PAGE
# ============================================================================
elif page == "📤 Batch Prediction":
    st.markdown('<p class="main-header">📤 Batch Prediction</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Upload Your Customer Data

    Upload a CSV file containing customer features to get instant segmentation for all customers.

    **Required Columns**: Your CSV must contain all 17 feature columns:
    - BALANCE, BALANCE_FREQUENCY, PURCHASES, ONEOFF_PURCHASES, INSTALLMENTS_PURCHASES
    - CASH_ADVANCE, PURCHASES_FREQUENCY, ONEOFF_PURCHASES_FREQUENCY
    - PURCHASES_INSTALLMENTS_FREQUENCY, CASH_ADVANCE_FREQUENCY, CASH_ADVANCE_TRX
    - PURCHASES_TRX, CREDIT_LIMIT, PAYMENTS, MINIMUM_PAYMENTS, PRC_FULL_PAYMENT, TENURE
    """)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with customer data"
    )

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)

            st.success(f"✅ File uploaded successfully! Found {len(df)} customers.")

            # Preview data
            with st.expander("📋 Preview Uploaded Data", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

            # Validate columns
            missing_cols = set(NUMERIC_FEATURES) - set(df.columns)

            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
            else:
                st.success("✅ All required columns present!")

                if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
                    with st.spinner("Generating predictions..."):
                        # Prepare features
                        X = df[NUMERIC_FEATURES].values
                        X_imputed = imputer.transform(X)
                        X_scaled = scaler.transform(X_imputed)

                        # Predict
                        clusters = kmeans.predict(X_scaled)

                        # Add predictions to dataframe
                        df['cluster'] = clusters
                        df['cluster_name'] = [CLUSTER_DESCRIPTIONS[c]["name"] for c in clusters]
                        df['cluster_description'] = [CLUSTER_DESCRIPTIONS[c]["description"] for c in clusters]

                        st.success("✅ Predictions generated successfully!")

                        # Show results
                        st.markdown('<p class="sub-header">Prediction Results</p>', unsafe_allow_html=True)

                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Customers", len(df))
                        with col2:
                            st.metric("Unique Segments", df['cluster'].nunique())
                        with col3:
                            most_common = df['cluster_name'].value_counts().index[0]
                            st.metric("Largest Segment", most_common)

                        # Cluster distribution
                        cluster_dist = df['cluster_name'].value_counts()
                        fig = px.bar(
                            x=cluster_dist.index,
                            y=cluster_dist.values,
                            color=cluster_dist.index,
                            color_discrete_map={CLUSTER_DESCRIPTIONS[i]["name"]: CLUSTER_DESCRIPTIONS[i]["color"] for i in CLUSTER_DESCRIPTIONS.keys()},
                            title="Distribution of Predicted Segments",
                            labels={"x": "Segment", "y": "Number of Customers"}
                        )
                        fig.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig, use_container_width=True)

                        # Show results table
                        with st.expander("📊 View All Predictions", expanded=False):
                            st.dataframe(df, use_container_width=True)

                        # Download options
                        st.markdown('<p class="sub-header">Download Results</p>', unsafe_allow_html=True)

                        col1, col2 = st.columns(2)

                        with col1:
                            # CSV download
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv,
                                file_name="customer_segments.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                        with col2:
                            # Excel download
                            buffer = BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                df.to_excel(writer, index=False, sheet_name='Predictions')

                            st.download_button(
                                label="📥 Download as Excel",
                                data=buffer.getvalue(),
                                file_name="customer_segments.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

    else:
        # Show sample data
        st.info("👆 Upload a file to get started, or download our sample data to test:")

        # Provide sample data
        sample_df = pd.read_csv("data/processed/live_raw.csv").head(100)
        if "CUST_ID" in sample_df.columns:
            sample_df = sample_df.drop("CUST_ID", axis=1)

        csv_sample = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Data",
            data=csv_sample,
            file_name="sample_customers.csv",
            mime="text/csv"
        )

# ============================================================================
# SINGLE PREDICTION PAGE
# ============================================================================
elif page == "🎯 Single Prediction":
    st.markdown('<p class="main-header">🎯 Single Customer Prediction</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Predict Individual Customer Segment

    Enter customer feature values below to see their predicted segment with AI explainability.
    """)

    # Input form
    st.markdown('<p class="sub-header">Customer Features</p>', unsafe_allow_html=True)

    vals = {}

    # Organize inputs in columns
    col1, col2, col3 = st.columns(3)

    for i, feature in enumerate(NUMERIC_FEATURES):
        default_val = float(profiles[feature].mean())

        with [col1, col2, col3][i % 3]:
            vals[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=default_val,
                step=0.1,
                format="%.2f",
                key=feature
            )

    if st.button("🔮 Predict Segment", type="primary", use_container_width=True):
        with st.spinner("Analyzing customer profile..."):
            # Prepare input
            arr = np.array([[vals[c] for c in NUMERIC_FEATURES]])
            arr_imputed = imputer.transform(arr)
            arr_scaled = scaler.transform(arr_imputed)

            # Predict
            cluster = int(kmeans.predict(arr_scaled)[0])
            cluster_info = CLUSTER_DESCRIPTIONS[cluster]

            # Display result
            st.markdown('<p class="sub-header">Prediction Result</p>', unsafe_allow_html=True)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"""
                <div class="cluster-card" style="border-left-color: {cluster_info['color']};">
                    <h2 style="color: {cluster_info['color']};">Cluster {cluster}</h2>
                    <h3>{cluster_info['name']}</h3>
                    <p>{cluster_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # SHAP explanation
                st.markdown("#### 🔍 AI Explainability (SHAP Values)")
                st.info("This shows which features most influenced this prediction.")

                try:
                    # Create SHAP explainer
                    explainer = shap.KernelExplainer(
                        lambda x: kmeans.predict(scaler.transform(imputer.transform(x))),
                        shap.sample(scaler.transform(imputer.transform(features.values[:100])), 50)
                    )

                    # Get SHAP values
                    shap_values = explainer.shap_values(arr)

                    # Create feature importance plot
                    feature_importance = pd.DataFrame({
                        'Feature': NUMERIC_FEATURES,
                        'Importance': np.abs(shap_values[0])
                    }).sort_values('Importance', ascending=True).tail(10)

                    fig = px.bar(
                        feature_importance,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 10 Most Important Features",
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.warning(f"SHAP explanation temporarily unavailable. Showing feature comparison instead.")

                    # Fallback: Show comparison with cluster average
                    cluster_avg = profiles[profiles['cluster'] == cluster].iloc[0]

                    comparison_data = []
                    for feature in NUMERIC_FEATURES[:10]:  # Top 10 features
                        comparison_data.append({
                            'Feature': feature,
                            'Your Value': vals[feature],
                            'Cluster Average': cluster_avg[feature]
                        })

                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)

# ============================================================================
# CLUSTER ANALYSIS PAGE
# ============================================================================
elif page == "🔍 Cluster Analysis":
    st.markdown('<p class="main-header">🔍 Deep Dive: Cluster Analysis</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Compare and Analyze Customer Segments

    Select two clusters to compare their characteristics side-by-side.
    """)

    col1, col2 = st.columns(2)

    with col1:
        cluster1 = st.selectbox(
            "Select First Cluster",
            options=sorted(CLUSTER_DESCRIPTIONS.keys()),
            format_func=lambda x: f"Cluster {x}: {CLUSTER_DESCRIPTIONS[x]['name']}"
        )

    with col2:
        cluster2 = st.selectbox(
            "Select Second Cluster",
            options=sorted(CLUSTER_DESCRIPTIONS.keys()),
            format_func=lambda x: f"Cluster {x}: {CLUSTER_DESCRIPTIONS[x]['name']}",
            index=1 if len(CLUSTER_DESCRIPTIONS) > 1 else 0
        )

    if cluster1 == cluster2:
        st.warning("⚠️ Please select two different clusters for comparison.")
    else:
        # Get cluster data
        c1_data = profiles[profiles['cluster'] == cluster1].iloc[0]
        c2_data = profiles[profiles['cluster'] == cluster2].iloc[0]

        # Cluster info cards
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="cluster-card" style="border-left-color: {CLUSTER_DESCRIPTIONS[cluster1]['color']};">
                <h3 style="color: {CLUSTER_DESCRIPTIONS[cluster1]['color']};">{CLUSTER_DESCRIPTIONS[cluster1]['name']}</h3>
                <p>{CLUSTER_DESCRIPTIONS[cluster1]['description']}</p>
                <p><strong>Size:</strong> {len(assignments[assignments['cluster'] == cluster1])} customers</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="cluster-card" style="border-left-color: {CLUSTER_DESCRIPTIONS[cluster2]['color']};">
                <h3 style="color: {CLUSTER_DESCRIPTIONS[cluster2]['color']};">{CLUSTER_DESCRIPTIONS[cluster2]['name']}</h3>
                <p>{CLUSTER_DESCRIPTIONS[cluster2]['description']}</p>
                <p><strong>Size:</strong> {len(assignments[assignments['cluster'] == cluster2])} customers</p>
            </div>
            """, unsafe_allow_html=True)

        # Radar chart comparison
        st.markdown('<p class="sub-header">Feature Comparison (Normalized)</p>', unsafe_allow_html=True)

        # Select top features for radar chart
        selected_features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT',
                            'PAYMENTS', 'PRC_FULL_PAYMENT', 'PURCHASES_TRX', 'TENURE']

        # Normalize values for visualization
        from sklearn.preprocessing import MinMaxScaler
        scaler_viz = MinMaxScaler()
        all_values = profiles[selected_features].values
        normalized = scaler_viz.fit_transform(all_values)

        c1_idx = profiles[profiles['cluster'] == cluster1].index[0]
        c2_idx = profiles[profiles['cluster'] == cluster2].index[0]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=normalized[c1_idx],
            theta=selected_features,
            fill='toself',
            name=CLUSTER_DESCRIPTIONS[cluster1]['name'],
            line_color=CLUSTER_DESCRIPTIONS[cluster1]['color']
        ))

        fig.add_trace(go.Scatterpolar(
            r=normalized[c2_idx],
            theta=selected_features,
            fill='toself',
            name=CLUSTER_DESCRIPTIONS[cluster2]['name'],
            line_color=CLUSTER_DESCRIPTIONS[cluster2]['color']
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # Detailed comparison table
        st.markdown('<p class="sub-header">Detailed Feature Comparison</p>', unsafe_allow_html=True)

        comparison_data = []
        for feature in NUMERIC_FEATURES:
            c1_val = c1_data[feature]
            c2_val = c2_data[feature]
            diff = c1_val - c2_val
            diff_pct = (diff / c2_val * 100) if c2_val != 0 else 0

            comparison_data.append({
                'Feature': feature,
                CLUSTER_DESCRIPTIONS[cluster1]['name']: f"{c1_val:.2f}",
                CLUSTER_DESCRIPTIONS[cluster2]['name']: f"{c2_val:.2f}",
                'Difference': f"{diff:+.2f}",
                'Difference (%)': f"{diff_pct:+.1f}%"
            })

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, height=600)

        # Business insights
        st.markdown('<p class="sub-header">💡 Business Insights</p>', unsafe_allow_html=True)

        st.markdown(f"""
        **Marketing Strategy Recommendations:**

        **For {CLUSTER_DESCRIPTIONS[cluster1]['name']}:**
        - Focus on their {CLUSTER_DESCRIPTIONS[cluster1]['description'].lower()}
        - Tailor offers based on their spending patterns

        **For {CLUSTER_DESCRIPTIONS[cluster2]['name']}:**
        - Focus on their {CLUSTER_DESCRIPTIONS[cluster2]['description'].lower()}
        - Create targeted campaigns for this segment
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>Customer Segmentation Dashboard v2.0 | Built with Streamlit, scikit-learn, SHAP & Plotly</p>
    <p>End-to-End ML System: Data Ingestion → Preprocessing → Clustering → Deployment</p>
</div>
""", unsafe_allow_html=True)
