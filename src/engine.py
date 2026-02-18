import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def generate_psychographic_data(n_customers=2000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(18, 70, n_customers),
        'Income': np.random.randint(30000, 150000, n_customers),
        'Brand_Loyalty_Score': np.random.uniform(1, 10, n_customers),
        'Eco_Conscious_Score': np.random.uniform(1, 10, n_customers),
        'Tech_Savvy_Score': np.random.uniform(1, 10, n_customers),
        'Impulse_Buying_Score': np.random.uniform(1, 10, n_customers)
    }
    return pd.DataFrame(data)

def process_and_cluster(df, n_clusters=4):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    df['PCA1'] = pca_data[:, 0]
    df['PCA2'] = pca_data[:, 1]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Segment'] = kmeans.fit_predict(scaled_data)
    
    segment_names = {0: "Budget-Conscious Traditionalists", 
                     1: "Eco-Friendly Techies", 
                     2: "Impulsive High-Rollers", 
                     3: "Brand-Loyal Moderates"}
    df['Segment_Name'] = df['Segment'].map(segment_names)
    
    return df