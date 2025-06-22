"""
Model evaluation utilities for multi-label classification
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, jaccard_score, classification_report,
    multilabel_confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

class MultiLabelEvaluator:
    def __init__(self):
        self.metrics_history = []
        
    def evaluate_model(self, y_true, y_pred, target_names=None, model_name="Model"):
        """Comprehensive evaluation of multi-label classification model"""
        
        # Calculate all metrics
        metrics = {
            'model_name': model_name,
            'exact_match_ratio': accuracy_score(y_true, y_pred),
            'hamming_loss': hamming_loss(y_true, y_pred),
            'jaccard_similarity': jaccard_score(y_true, y_pred, average='samples'),
            'precision_micro': precision_score(y_true, y_pred, average='micro'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_micro': recall_score(y_true, y_pred, average='micro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Per-class metrics
        if target_names is not None:
            per_class_metrics = self._calculate_per_class_metrics(y_true, y_pred, target_names)
            metrics['per_class'] = per_class_metrics
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_per_class_metrics(self, y_true, y_pred, target_names):
        """Calculate per-class precision, recall, and F1-score"""
        per_class = {}
        
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        for i, class_name in enumerate(target_names):
            per_class[class_name] = {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1_score': f1_per_class[i],
                'support': np.sum(y_true[:, i])
            }
        
        return per_class
    
    def print_evaluation_report(self, metrics, detailed=True):
        """Print comprehensive evaluation report"""
        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT - {metrics['model_name']}")
        print(f"{'='*60}")
        
        # Overall metrics
        print(f"\nOVERALL METRICS:")
        print(f"Exact Match Ratio (Subset Accuracy): {metrics['exact_match_ratio']:.4f}")
        print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
        print(f"Jaccard Similarity: {metrics['jaccard_similarity']:.4f}")
        
        print(f"\nPRECISION:")
        print(f"  Micro Average: {metrics['precision_micro']:.4f}")
        print(f"  Macro Average: {metrics['precision_macro']:.4f}")
        print(f"  Weighted Average: {metrics['precision_weighted']:.4f}")
        
        print(f"\nRECALL:")
        print(f"  Micro Average: {metrics['recall_micro']:.4f}")
        print(f"  Macro Average: {metrics['recall_macro']:.4f}")
        print(f"  Weighted Average: {metrics['recall_weighted']:.4f}")
        
        print(f"\nF1-SCORE:")
        print(f"  Micro Average: {metrics['f1_micro']:.4f}")
        print(f"  Macro Average: {metrics['f1_macro']:.4f}")
        print(f"  Weighted Average: {metrics['f1_weighted']:.4f}")
        
        # Per-class metrics
        if detailed and 'per_class' in metrics:
            print(f"\nPER-CLASS METRICS:")
            print(f"{'Class':<30} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            print("-" * 70)
            
            for class_name, class_metrics in metrics['per_class'].items():
                print(f"{class_name:<30} {class_metrics['precision']:<10.4f} "
                      f"{class_metrics['recall']:<10.4f} {class_metrics['f1_score']:<10.4f} "
                      f"{class_metrics['support']:<10}")
    
    def plot_metrics_comparison(self, save_path=None):
        """Plot comparison of metrics across different models"""
        if len(self.metrics_history) < 2:
            print("Need at least 2 models to compare")
            return
        
        # Prepare data for plotting
        models = [m['model_name'] for m in self.metrics_history]
        
        metrics_to_plot = [
            'exact_match_ratio', 'jaccard_similarity',
            'precision_micro', 'recall_micro', 'f1_micro',
            'precision_macro', 'recall_macro', 'f1_macro'
        ]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            values = [m[metric] for m in self.metrics_history]
            
            axes[i].bar(models, values)
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, target_names, save_path=None):
        """Plot multi-label confusion matrix"""
        # Calculate confusion matrix for each class
        cm_array = multilabel_confusion_matrix(y_true, y_pred)
        
        n_classes = len(target_names)
        fig, axes = plt.subplots(2, (n_classes + 1) // 2, figsize=(15, 8))
        
        if n_classes == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (class_name, cm) in enumerate(zip(target_names, cm_array)):
            if i >= len(axes):
                break
                
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{class_name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_per_class_metrics(self, metrics, save_path=None):
        """Plot per-class precision, recall, and F1-score"""
        if 'per_class' not in metrics:
            print("Per-class metrics not available")
            return
        
        per_class = metrics['per_class']
        classes = list(per_class.keys())
        
        precision_values = [per_class[c]['precision'] for c in classes]
        recall_values = [per_class[c]['recall'] for c in classes]
        f1_values = [per_class[c]['f1_score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        bars1 = ax.bar(x - width, precision_values, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall_values, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1_values, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('SDG Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_metrics_dataframe(self):
        """Get metrics as pandas DataFrame for easy comparison"""
        if not self.metrics_history:
            return pd.DataFrame()
        
        # Extract main metrics
        main_metrics = []
        for metrics in self.metrics_history:
            main_metric = {k: v for k, v in metrics.items() if k != 'per_class'}
            main_metrics.append(main_metric)
        
        return pd.DataFrame(main_metrics)
    
    def save_detailed_report(self, filepath, metrics):
        """Save detailed evaluation report to file"""
        with open(filepath, 'w') as f:
            f.write(f"DETAILED EVALUATION REPORT - {metrics['model_name']}\n")
            f.write("="*60 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS:\n")
            f.write(f"Exact Match Ratio: {metrics['exact_match_ratio']:.6f}\n")
            f.write(f"Hamming Loss: {metrics['hamming_loss']:.6f}\n")
            f.write(f"Jaccard Similarity: {metrics['jaccard_similarity']:.6f}\n\n")
            
            # Detailed metrics
            f.write("PRECISION:\n")
            f.write(f"  Micro: {metrics['precision_micro']:.6f}\n")
            f.write(f"  Macro: {metrics['precision_macro']:.6f}\n")
            f.write(f"  Weighted: {metrics['precision_weighted']:.6f}\n\n")
            
            f.write("RECALL:\n")
            f.write(f"  Micro: {metrics['recall_micro']:.6f}\n")
            f.write(f"  Macro: {metrics['recall_macro']:.6f}\n")
            f.write(f"  Weighted: {metrics['recall_weighted']:.6f}\n\n")
            
            f.write("F1-SCORE:\n")
            f.write(f"  Micro: {metrics['f1_micro']:.6f}\n")
            f.write(f"  Macro: {metrics['f1_macro']:.6f}\n")
            f.write(f"  Weighted: {metrics['f1_weighted']:.6f}\n\n")
            
            # Per-class metrics
            if 'per_class' in metrics:
                f.write("PER-CLASS METRICS:\n")
                f.write(f"{'Class':<35} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
                f.write("-" * 80 + "\n")
                
                for class_name, class_metrics in metrics['per_class'].items():
                    f.write(f"{class_name:<35} {class_metrics['precision']:<12.6f} "
                           f"{class_metrics['recall']:<12.6f} {class_metrics['f1_score']:<12.6f} "
                           f"{class_metrics['support']:<10}\n")
        
        print(f"Detailed report saved to: {filepath}")
