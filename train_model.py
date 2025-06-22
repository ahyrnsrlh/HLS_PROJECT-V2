"""
Main training script for SDG multi-label classification
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Import custom utilities
from utils.preprocessing import DataPreprocessor
from utils.clustering import ClusterAnalyzer
from utils.models import SDGClassificationPipeline
from utils.evaluation import MultiLabelEvaluator

def main():
    """Main training pipeline"""
    print("="*80)
    print("SDG MULTI-LABEL CLASSIFICATION TRAINING PIPELINE")
    print("="*80)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Configuration
    DATA_PATH = "data/2503_to_3336_preprocessing_labeling.csv"
    MODEL_SAVE_DIR = "models"
    RESULTS_DIR = "results"
    
    # Create directories
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Step 1: Load and explore data
    print("\n1. LOADING DATA...")
    print("-" * 40)
    
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Data loaded successfully: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Basic data info
        print(f"\nData Info:")
        print(f"Total entries: {len(df)}")
        print(f"Missing values per column:")
        print(df.isnull().sum())
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return
    
    # Step 2: Data preprocessing
    print("\n2. PREPROCESSING DATA...")
    print("-" * 40)
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess_data(df)
    
    print(f"Data after preprocessing: {df_processed.shape}")
    print(f"Entries with valid SDG labels: {len(df_processed)}")
    
    # Analyze SDG distribution
    print("\nSDG Label Distribution:")
    all_sdgs = []
    for sdg_list in df_processed['SDG_labels']:
        all_sdgs.extend(sdg_list)
    
    sdg_counts = pd.Series(all_sdgs).value_counts()
    print(sdg_counts)
    
    # Create feature matrix and target matrix
    print("\nCreating feature and target matrices...")
    X = preprocessor.create_feature_matrix(df_processed, fit_transform=True)
    y = preprocessor.create_target_matrix(df_processed, fit_transform=True)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")
    print(f"Number of unique SDG classes: {len(preprocessor.get_target_names())}")
    print(f"SDG classes: {preprocessor.get_target_names()}")
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
    
    # Step 4: Clustering analysis
    print("\n4. CLUSTERING ANALYSIS...")
    print("-" * 40)
    
    cluster_analyzer = ClusterAnalyzer()
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
    cluster_optimization = cluster_analyzer.find_optimal_clusters(X_train, max_clusters=15)
    
    print(f"Optimal number of clusters (silhouette): {cluster_optimization['best_silhouette_k']}")
    print(f"Elbow point: {cluster_optimization['elbow_k']}")
    
    # Fit clustering model
    cluster_labels_train = cluster_analyzer.fit_kmeans(X_train, n_clusters=cluster_optimization['best_silhouette_k'])
    cluster_labels_val = cluster_analyzer.kmeans_model.predict(X_val)
    cluster_labels_test = cluster_analyzer.kmeans_model.predict(X_test)
    
    # Analyze clusters
    cluster_stats, sdg_distribution = cluster_analyzer.analyze_clusters(
        X_train, y_train, preprocessor.get_target_names()
    )
    
    print(f"Clustering statistics:")
    print(f"Number of clusters: {cluster_stats['n_clusters']}")
    print(f"Silhouette score: {cluster_stats['silhouette_score']:.4f}")
    print(f"Cluster sizes: {cluster_stats['cluster_sizes']}")
    
    # Visualize clusters
    cluster_viz_path = os.path.join(RESULTS_DIR, "cluster_visualization.png")
    cluster_analyzer.visualize_clusters(
        X_train, y_train, preprocessor.get_target_names(), save_path=cluster_viz_path
    )
    
    # Get cluster features
    cluster_features_train = cluster_analyzer.get_cluster_features()
    cluster_features_val = cluster_labels_val.reshape(-1, 1)
    cluster_features_test = cluster_labels_test.reshape(-1, 1)
    
    # Step 5: Model training
    print("\n5. TRAINING MODELS...")
    print("-" * 40)
    
    pipeline = SDGClassificationPipeline()
    
    # Train models (both regular and hybrid)
    pipeline.train_models(
        X_train, y_train, X_val, y_val,
        cluster_features_train, cluster_features_val,
        tune_hyperparams=True
    )
    
    print(f"\nTrained models: {pipeline.get_model_names()}")
    
    # Step 6: Model evaluation
    print("\n6. MODEL EVALUATION...")
    print("-" * 40)
    
    evaluator = MultiLabelEvaluator()
    target_names = preprocessor.get_target_names()
    
    # Evaluate all models on test set
    test_results = {}
    
    for model_name in pipeline.get_model_names():
        print(f"\nEvaluating {model_name}...")
        
        try:
            # Make predictions
            if model_name.startswith('hybrid_'):
                y_pred = pipeline.predict(X_test, model_name, cluster_features_test)
            else:
                y_pred = pipeline.predict(X_test, model_name)
            
            # Evaluate
            metrics = evaluator.evaluate_model(y_test, y_pred, target_names, model_name)
            test_results[model_name] = metrics
            
            # Print brief results
            print(f"  Exact Match Ratio: {metrics['exact_match_ratio']:.4f}")
            print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
            print(f"  F1-Score (Micro): {metrics['f1_micro']:.4f}")
            print(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
            
        except Exception as e:
            print(f"  Error evaluating {model_name}: {str(e)}")
    
    # Step 7: Results visualization and comparison
    print("\n7. RESULTS VISUALIZATION...")
    print("-" * 40)
    
    # Plot metrics comparison
    comparison_path = os.path.join(RESULTS_DIR, "model_comparison.png")
    evaluator.plot_metrics_comparison(save_path=comparison_path)
    
    # Plot best model detailed results
    best_model_name = pipeline.best_model_name
    if best_model_name and best_model_name in test_results:
        best_metrics = test_results[best_model_name]
        
        # Detailed evaluation report
        print(f"\n{'='*60}")
        print("BEST MODEL DETAILED EVALUATION")
        evaluator.print_evaluation_report(best_metrics, detailed=True)
        
        # Per-class metrics plot
        per_class_path = os.path.join(RESULTS_DIR, f"per_class_metrics_{best_model_name}.png")
        evaluator.plot_per_class_metrics(best_metrics, save_path=per_class_path)
        
        # Confusion matrix
        if best_model_name.startswith('hybrid_'):
            y_pred_best = pipeline.predict(X_test, best_model_name, cluster_features_test)
        else:
            y_pred_best = pipeline.predict(X_test, best_model_name)
        
        cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{best_model_name}.png")
        evaluator.plot_confusion_matrix(y_test, y_pred_best, target_names, save_path=cm_path)
        
        # Save detailed report
        report_path = os.path.join(RESULTS_DIR, f"detailed_report_{best_model_name}.txt")
        evaluator.save_detailed_report(report_path, best_metrics)
    
    # Step 8: Save models and artifacts
    print("\n8. SAVING MODELS AND ARTIFACTS...")
    print("-" * 40)
    
    # Save trained models
    pipeline.save_models(MODEL_SAVE_DIR)
    
    # Save preprocessor and cluster analyzer
    import joblib
    joblib.dump(preprocessor, os.path.join(MODEL_SAVE_DIR, "preprocessor.joblib"))
    joblib.dump(cluster_analyzer, os.path.join(MODEL_SAVE_DIR, "cluster_analyzer.joblib"))
    
    # Save results summary
    results_df = evaluator.get_metrics_dataframe()
    results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    
    # Save feature names for interpretability
    feature_names = preprocessor.get_feature_names()
    joblib.dump(feature_names, os.path.join(MODEL_SAVE_DIR, "feature_names.joblib"))
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Models saved to: {MODEL_SAVE_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Best model: {pipeline.best_model_name}")
    print(f"Best F1-score: {pipeline.best_score:.4f}")
    
    # Performance comparison
    print(f"\nFINAL MODEL COMPARISON:")
    print("-" * 40)
    for model_name, metrics in test_results.items():
        print(f"{model_name:25} | F1-Micro: {metrics['f1_micro']:.4f} | "
              f"F1-Macro: {metrics['f1_macro']:.4f} | "
              f"Exact Match: {metrics['exact_match_ratio']:.4f}")

if __name__ == "__main__":
    main()
