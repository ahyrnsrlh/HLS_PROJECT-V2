"""
Fast training script for SDG multi-label classification - simplified version
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Import custom utilities
from utils.preprocessing import DataPreprocessor
from utils.clustering import ClusterAnalyzer

class FastSDGClassifier:
    """Fast SDG classification pipeline for quick testing"""
    
    def __init__(self):
        self.models = {}
        self.hybrid_models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
    
    def train_baseline_models(self, X_train, y_train, X_val=None, y_val=None, fast_mode=True):
        """Train baseline models without clustering"""
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        
        print("Training baseline models...")
        
        # Logistic Regression
        print("  Training Logistic Regression...")
        max_iter = 300 if fast_mode else 1000
        lr_model = MultiOutputClassifier(
            LogisticRegression(random_state=42, max_iter=max_iter, solver='liblinear')
        )
        lr_model.fit(X_train, y_train)
        self.models['logistic_regression'] = lr_model
        
        if X_val is not None and y_val is not None:
            y_pred = lr_model.predict(X_val)
            f1_micro = f1_score(y_val, y_pred, average='micro', zero_division=0)
            print(f"    Validation F1-micro: {f1_micro:.4f}")        # Random Forest
        print("  Training Random Forest...")
        n_estimators = 30 if fast_mode else 100
        rf_model = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        if X_val is not None and y_val is not None:
            y_pred = rf_model.predict(X_val)
            f1_micro = f1_score(y_val, y_pred, average='micro', zero_division=0)
            print(f"    Validation F1-micro: {f1_micro:.4f}")
    
    def train_hybrid_models(self, X_train_hybrid, y_train, X_val_hybrid=None, y_val=None, fast_mode=True):
        """Train hybrid models with cluster features"""
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        
        print("Training hybrid models...")
        
        # Hybrid Logistic Regression
        print("  Training Hybrid Logistic Regression...")
        max_iter = 300 if fast_mode else 1000
        lr_hybrid = MultiOutputClassifier(
            LogisticRegression(random_state=42, max_iter=max_iter, solver='liblinear')
        )
        lr_hybrid.fit(X_train_hybrid, y_train)
        self.hybrid_models['logistic_regression'] = lr_hybrid
        
        if X_val_hybrid is not None and y_val is not None:
            y_pred = lr_hybrid.predict(X_val_hybrid)
            f1_micro = f1_score(y_val, y_pred, average='micro', zero_division=0)
            print(f"    Validation F1-micro: {f1_micro:.4f}")
        
        # Hybrid Random Forest
        print("  Training Hybrid Random Forest...")
        n_estimators = 30 if fast_mode else 100
        rf_hybrid = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        )
        rf_hybrid.fit(X_train_hybrid, y_train)
        self.hybrid_models['random_forest'] = rf_hybrid
        
        if X_val_hybrid is not None and y_val is not None:
            y_pred = rf_hybrid.predict(X_val_hybrid)
            f1_micro = f1_score(y_val, y_pred, average='micro', zero_division=0)
            print(f"    Validation F1-micro: {f1_micro:.4f}")
    
    def evaluate_models(self, X_val, y_val, X_val_hybrid=None):
        """Evaluate all trained models and find the best one"""
        from sklearn.metrics import f1_score
        
        results = {}
        
        # Evaluate baseline models
        for name, model in self.models.items():
            y_pred = model.predict(X_val)
            f1_micro = f1_score(y_val, y_pred, average='micro', zero_division=0)
            f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
            results[name] = {'f1_micro': f1_micro, 'f1_macro': f1_macro}
        
        # Evaluate hybrid models
        if X_val_hybrid is not None:
            for name, model in self.hybrid_models.items():
                y_pred = model.predict(X_val_hybrid)
                f1_micro = f1_score(y_val, y_pred, average='micro', zero_division=0)
                f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
                results[f"{name}_hybrid"] = {'f1_micro': f1_micro, 'f1_macro': f1_macro}
        
        # Find best model
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['f1_micro'])
            self.best_model_name = best_model_name
            self.best_score = results[best_model_name]['f1_micro']
            
            # Set best model reference
            if 'hybrid' in best_model_name:
                base_name = best_model_name.replace('_hybrid', '')
                self.best_model = self.hybrid_models.get(base_name)
            else:
                self.best_model = self.models.get(best_model_name)
        
        return results
    
    def save_models(self, save_dir, save_all=True):
        """Save trained models"""
        import joblib
        import os
        
        os.makedirs(save_dir, exist_ok=True)
        
        if save_all:
            # Save all models
            for name, model in self.models.items():
                model_path = os.path.join(save_dir, f"{name}.joblib")
                joblib.dump(model, model_path)
                print(f"Saved {name} to {model_path}")
            
            for name, model in self.hybrid_models.items():
                model_path = os.path.join(save_dir, f"{name}_hybrid.joblib")
                joblib.dump(model, model_path)
                print(f"Saved {name}_hybrid to {model_path}")
        
        # Save best model info
        if self.best_model is not None:
            best_model_path = os.path.join(save_dir, "best_model.joblib")
            joblib.dump(self.best_model, best_model_path)
            
            best_info = {
                'best_model_name': self.best_model_name,
                'best_score': self.best_score
            }
            info_path = os.path.join(save_dir, "best_model_info.joblib")
            joblib.dump(best_info, info_path)
            
            print(f"Saved best model ({self.best_model_name}) to {best_model_path}")

def main():
    """Fast training pipeline with optimizations"""
    print("="*80)
    print("SDG MULTI-LABEL CLASSIFICATION - FAST TRAINING PIPELINE")
    print("="*80)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Configuration
    DATA_PATH = "data/2503_to_3336_preprocessing_labeling.csv"
    MODEL_SAVE_DIR = "models"
    RESULTS_DIR = "results"
    
    # Fast training options
    SAMPLE_SIZE = 3000  # Use smaller sample for faster training
    MAX_CLUSTERS_TO_TEST = 8  # Reduce cluster search space
    
    # Create directories
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Step 1: Load and explore data
    print("\n1. LOADING DATA...")
    print("-" * 40)
    
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Original data shape: {df.shape}")
        
        # Sample data for faster training
        if len(df) > SAMPLE_SIZE:
            df = df.sample(n=SAMPLE_SIZE, random_state=42)
            print(f"Sampled data for fast training: {df.shape}")
        
        print(f"Columns: {list(df.columns)}")
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return
    
    # Step 2: Data preprocessing
    print("\n2. PREPROCESSING DATA...")
    print("-" * 40)
    
    start_time = time()
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess_data(df)
    
    print(f"Data after preprocessing: {df_processed.shape}")
    print(f"Preprocessing time: {time() - start_time:.2f} seconds")
    
    # Analyze SDG distribution
    print("\nSDG Label Distribution:")
    all_sdgs = []
    for sdg_list in df_processed['SDG_labels']:
        all_sdgs.extend(sdg_list)
    
    sdg_counts = pd.Series(all_sdgs).value_counts()
    print(sdg_counts.head(10))
    
    # Create feature matrix and target matrix
    print("\nCreating feature and target matrices...")
    start_time = time()
    X = preprocessor.create_feature_matrix(df_processed, fit_transform=True)
    y = preprocessor.create_target_matrix(df_processed, fit_transform=True)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")
    print(f"Feature creation time: {time() - start_time:.2f} seconds")
    print(f"Number of unique SDG classes: {len(preprocessor.get_target_names())}")
    
    # Step 3: Train-test split
    print("\n3. SPLITTING DATA...")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Further split training set for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Step 4: Fast clustering analysis
    print("\n4. FAST CLUSTERING ANALYSIS...")
    print("-" * 40)
    
    start_time = time()
    
    # Quick clustering with limited parameter search
    print("Finding optimal number of clusters (fast mode)...")
    
    # Test only a few cluster numbers
    k_range = range(3, MAX_CLUSTERS_TO_TEST + 1)
    silhouette_scores = []
    
    for k in k_range:
        print(f"Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)  # Reduced n_init
        cluster_labels = kmeans.fit_predict(X_train)
        score = silhouette_score(X_train, cluster_labels)
        silhouette_scores.append(score)
        print(f"  Silhouette score: {score:.4f}")
    
    # Find best k
    best_k = k_range[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)
    
    print(f"\nOptimal number of clusters: {best_k}")
    print(f"Best silhouette score: {best_score:.4f}")
    print(f"Clustering analysis time: {time() - start_time:.2f} seconds")
    
    # Fit final clustering model
    print(f"\nFitting final clustering model with k={best_k}...")
    start_time = time()
    
    cluster_analyzer = ClusterAnalyzer()
    cluster_labels_train = cluster_analyzer.fit_kmeans(X_train, n_clusters=best_k)
    cluster_labels_val = cluster_analyzer.kmeans_model.predict(X_val)
    cluster_labels_test = cluster_analyzer.kmeans_model.predict(X_test)
    
    print(f"Final clustering time: {time() - start_time:.2f} seconds")
    
    # Quick cluster analysis
    unique_labels = np.unique(cluster_labels_train)
    cluster_sizes = [np.sum(cluster_labels_train == i) for i in unique_labels]
    
    print(f"Cluster sizes: {cluster_sizes}")
    
    # Step 5: Model training (fast mode)
    print("\n5. TRAINING MODELS (FAST MODE)...")
    print("-" * 40)
    
    start_time = time()
    
    classifier = FastSDGClassifier()
    
    # Prepare hybrid features (combine original features with cluster info)
    X_train_hybrid = np.column_stack([X_train.toarray(), cluster_labels_train.reshape(-1, 1)])
    X_val_hybrid = np.column_stack([X_val.toarray(), cluster_labels_val.reshape(-1, 1)])
    X_test_hybrid = np.column_stack([X_test.toarray(), cluster_labels_test.reshape(-1, 1)])
    
    # Train baseline models
    print("Training baseline models...")
    classifier.train_baseline_models(X_train, y_train, X_val, y_val, fast_mode=True)
    
    print("Training hybrid models...")
    classifier.train_hybrid_models(X_train_hybrid, y_train, X_val_hybrid, y_val, fast_mode=True)
    
    print(f"Model training time: {time() - start_time:.2f} seconds")
    
    # Step 6: Quick evaluation
    print("\n6. QUICK EVALUATION...")
    print("-" * 40)
    
    # Evaluate all models
    results = classifier.evaluate_models(X_val, y_val, X_val_hybrid)
    
    print("\nModel Performance Summary:")
    for model_name, metrics in results.items():
        print(f"{model_name}: F1-micro={metrics['f1_micro']:.4f}, F1-macro={metrics['f1_macro']:.4f}")
    
    # Find best model
    if results:
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_micro'])
        best_f1_micro = results[best_model_name]['f1_micro']
        print(f"\nBest model: {best_model_name} with F1-micro: {best_f1_micro:.4f}")
    else:
        best_model_name = None
        best_f1_micro = 0
    
    # Step 7: Save models
    print("\n7. SAVING MODELS...")
    print("-" * 40)    # Save classifier
    classifier.save_models(MODEL_SAVE_DIR, save_all=True)
    
    # Save preprocessor and cluster analyzer using joblib
    import joblib
    joblib.dump(preprocessor, os.path.join(MODEL_SAVE_DIR, "preprocessor.joblib"))
    print(f"Saved preprocessor to {os.path.join(MODEL_SAVE_DIR, 'preprocessor.joblib')}")
    
    joblib.dump(cluster_analyzer, os.path.join(MODEL_SAVE_DIR, "cluster_model.joblib"))
    print(f"Saved cluster analyzer to {os.path.join(MODEL_SAVE_DIR, 'cluster_model.joblib')}")
    
    # Save results summary
    results_summary = {
        'data_shape': df.shape,
        'processed_shape': df_processed.shape,
        'feature_shape': X.shape,
        'target_shape': y.shape,
        'best_clusters': best_k,
        'best_silhouette': best_score,
        'model_results': results,
        'best_model': best_model_name,
        'best_f1_micro': best_f1_micro
    }
    
    # Save to file
    import json
    with open(os.path.join(RESULTS_DIR, "fast_training_results.json"), 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print("="*80)
    print("FAST TRAINING COMPLETED!")
    print("="*80)
    print(f"Models saved to: {MODEL_SAVE_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")
    
    if best_model_name:
        print(f"Best model: {best_model_name}")
        print(f"Best F1-micro score: {best_f1_micro:.4f}")
    
    print("\nTo run predictions, use: python predict_sample.py")

if __name__ == "__main__":
    main()
