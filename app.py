import streamlit as st
import plotly.express as px
from src.engine import generate_psychographic_data, process_and_cluster

st.set_page_config(page_title="Market Segmentation Engine", layout="wide")

st.title("🎯 Enterprise Customer Segmentation Dashboard")
st.markdown("Analyze psychographics and demographics using PCA dimensionality reduction and K-Means clustering.")

st.sidebar.header("Model Parameters")
n_customers = st.sidebar.slider("Number of Customers to Simulate", 500, 5000, 2000)
n_clusters = st.sidebar.slider("Number of Market Segments", 2, 7, 4)

with st.spinner('Generating data and training clustering model...'):
    raw_df = generate_psychographic_data(n_customers)
    processed_df = process_and_cluster(raw_df, n_clusters)

col1, col2 = st.columns(2)

with col1:
    st.subheader("2D PCA Cluster Visualization")
    fig = px.scatter(processed_df, x='PCA1', y='PCA2', color='Segment_Name', 
                     hover_data=['Age', 'Income', 'Brand_Loyalty_Score'],
                     title="Customer Segments in Reduced Dimensional Space")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Segment Demographics: Income vs Age")
    fig2 = px.scatter(processed_df, x='Age', y='Income', color='Segment_Name',
                      title="Distribution by Age and Income")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Raw Segment Data")
st.dataframe(processed_df.head(15))