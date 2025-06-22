"""
Clustering utilities for hybrid SDG classification
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class ClusterAnalyzer:
    def __init__(self):
        self.kmeans_model = None
        self.dbscan_model = None
        self.pca_model = None
        self.cluster_labels = None
        self.best_n_clusters = None
        
    def find_optimal_clusters(self, X, max_clusters=20, method='kmeans'):
        """Find optimal number of clusters using elbow method and silhouette score"""
        if method == 'kmeans':
            return self._find_optimal_kmeans(X, max_clusters)
        elif method == 'dbscan':
            return self._find_optimal_dbscan(X)
    
    def _find_optimal_kmeans(self, X, max_clusters):
        """Find optimal K for K-means using elbow method and silhouette score"""
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_clusters + 1, X.shape[0] // 2))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, cluster_labels))
        
        # Find elbow point (simplified)
        elbow_k = self._find_elbow_point(k_range, inertias)
        
        # Find best silhouette score
        best_silhouette_k = k_range[np.argmax(silhouette_scores)]
        
        # Use silhouette score as primary metric
        self.best_n_clusters = best_silhouette_k
        
        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'elbow_k': elbow_k,
            'best_silhouette_k': best_silhouette_k
        }
    
    def _find_elbow_point(self, k_range, inertias):
        """Find elbow point using the knee locator method"""
        # Simple elbow detection
        n_points = len(inertias)
        all_coord = np.vstack((k_range, inertias)).T
        first_point = all_coord[0]
        line_vec = all_coord[-1] - all_coord[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        
        vec_from_first = all_coord - first_point
        scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
        vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel
        
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
        elbow_idx = np.argmax(dist_to_line)
        
        return k_range[elbow_idx]
    
    def _find_optimal_dbscan(self, X):
        """Find optimal parameters for DBSCAN"""
        eps_range = np.arange(0.1, 2.0, 0.1)
        min_samples_range = range(5, 20)
        
        best_score = -1
        best_params = None
        results = []
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = dbscan.fit_predict(X)
                
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise = list(cluster_labels).count(-1)
                
                if n_clusters > 1 and n_noise < len(cluster_labels) * 0.5:
                    try:
                        score = silhouette_score(X, cluster_labels)
                        results.append({
                            'eps': eps,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'n_noise': n_noise,
                            'silhouette_score': score
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = {'eps': eps, 'min_samples': min_samples}
                    except:
                        continue
        
        return {'results': results, 'best_params': best_params}
    
    def fit_kmeans(self, X, n_clusters=None):
        """Fit K-means clustering"""
        if n_clusters is None:
            n_clusters = self.best_n_clusters or 8
        
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans_model.fit_predict(X)
        
        return self.cluster_labels
    
    def fit_dbscan(self, X, eps=0.5, min_samples=5):
        """Fit DBSCAN clustering"""
        self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_labels = self.dbscan_model.fit_predict(X)
        
        return self.cluster_labels
    
    def analyze_clusters(self, X, y_multilabel, target_names):
        """Analyze cluster quality and SDG distribution"""
        if self.cluster_labels is None:
            raise ValueError("No clustering performed yet")
        
        # Basic cluster statistics
        unique_clusters = np.unique(self.cluster_labels)
        n_clusters = len(unique_clusters)
        
        cluster_stats = {
            'n_clusters': n_clusters,
            'cluster_sizes': [np.sum(self.cluster_labels == c) for c in unique_clusters],
            'silhouette_score': silhouette_score(X, self.cluster_labels) if n_clusters > 1 else 0
        }
        
        # SDG distribution per cluster
        sdg_distribution = {}
        for cluster_id in unique_clusters:
            cluster_mask = self.cluster_labels == cluster_id
            cluster_sdgs = y_multilabel[cluster_mask]
            
            # Calculate SDG frequency in this cluster
            sdg_freq = np.mean(cluster_sdgs, axis=0)
            sdg_distribution[cluster_id] = dict(zip(target_names, sdg_freq))
        
        return cluster_stats, sdg_distribution
    
    def visualize_clusters(self, X, y_multilabel=None, target_names=None, save_path=None):
        """Visualize clusters using PCA"""
        if self.cluster_labels is None:
            raise ValueError("No clustering performed yet")
        
        # Reduce dimensionality for visualization
        if X.shape[1] > 2:
            self.pca_model = PCA(n_components=2, random_state=42)
            X_pca = self.pca_model.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
        else:
            X_pca = X
        
        plt.figure(figsize=(12, 8))
        
        # Plot clusters
        unique_clusters = np.unique(self.cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = self.cluster_labels == cluster_id
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.6)
        
        plt.title('Document Clustering Visualization (PCA)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # If SDG data is available, create SDG distribution heatmap
        if y_multilabel is not None and target_names is not None:
            self._plot_sdg_distribution_heatmap(y_multilabel, target_names, save_path)
    
    def _plot_sdg_distribution_heatmap(self, y_multilabel, target_names, save_path=None):
        """Plot SDG distribution across clusters"""
        cluster_sdg_matrix = []
        unique_clusters = np.unique(self.cluster_labels)
        
        for cluster_id in unique_clusters:
            cluster_mask = self.cluster_labels == cluster_id
            cluster_sdgs = y_multilabel[cluster_mask]
            sdg_freq = np.mean(cluster_sdgs, axis=0)
            cluster_sdg_matrix.append(sdg_freq)
        
        cluster_sdg_matrix = np.array(cluster_sdg_matrix)
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(cluster_sdg_matrix, 
                   xticklabels=target_names,
                   yticklabels=[f'Cluster {c}' for c in unique_clusters],
                   annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('SDG Distribution Across Clusters')
        plt.xlabel('SDG Categories')
        plt.ylabel('Clusters')
        plt.xticks(rotation=45, ha='right')
        
        if save_path:
            heatmap_path = save_path.replace('.png', '_sdg_heatmap.png')
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_cluster_features(self):
        """Get cluster labels as features for classification"""
        if self.cluster_labels is None:
            raise ValueError("No clustering performed yet")
        
        return self.cluster_labels.reshape(-1, 1)
