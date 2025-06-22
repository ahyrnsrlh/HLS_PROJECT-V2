"""
Multi-label classification models for SDG classification
"""

import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
from sklearn.metrics import make_scorer, f1_score
import joblib
import os

def f1_score_multilabel(y_true, y_pred, average='micro'):
    """Calculate F1 score for multilabel classification"""
    return f1_score(y_true, y_pred, average=average, zero_division=0)

class HybridClassifier(BaseEstimator, TransformerMixin):
    """Hybrid classifier that combines clustering features with original features"""
    
    def __init__(self, base_classifier, cluster_analyzer=None):
        self.base_classifier = base_classifier
        self.cluster_analyzer = cluster_analyzer
        
    def fit(self, X, y, cluster_features=None):
        """Fit the hybrid classifier"""
        if cluster_features is not None and self.cluster_analyzer is not None:
            # Combine original features with cluster features
            from scipy.sparse import hstack, csr_matrix
            X_combined = hstack([X, csr_matrix(cluster_features)])
            self.base_classifier.fit(X_combined, y)
        else:
            self.base_classifier.fit(X, y)
        return self
    
    def predict(self, X, cluster_features=None):
        """Predict using the hybrid classifier"""
        if cluster_features is not None and self.cluster_analyzer is not None:
            from scipy.sparse import hstack, csr_matrix
            X_combined = hstack([X, csr_matrix(cluster_features)])
            return self.base_classifier.predict(X_combined)
        else:
            return self.base_classifier.predict(X)
    
    def predict_proba(self, X, cluster_features=None):
        """Predict probabilities using the hybrid classifier"""
        if hasattr(self.base_classifier, 'predict_proba'):
            if cluster_features is not None and self.cluster_analyzer is not None:
                from scipy.sparse import hstack, csr_matrix
                X_combined = hstack([X, csr_matrix(cluster_features)])
                return self.base_classifier.predict_proba(X_combined)
            else:
                return self.base_classifier.predict_proba(X)
        else:
            return None

class SDGClassificationPipeline:
    """Complete pipeline for SDG multi-label classification"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
    def create_base_models(self):
        """Create base classification models"""
        models = {
            'logistic_regression': MultiOutputClassifier(
                LogisticRegression(random_state=42, max_iter=1000)
            ),
            'random_forest': MultiOutputClassifier(
                RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            ),
            'xgboost': MultiOutputClassifier(
                xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            )
        }
        return models
    
    def create_voting_classifier(self, base_models):
        """Create a voting classifier ensemble"""
        estimators = [(name, model) for name, model in base_models.items()]
        voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        return voting_clf
    
    def create_stacking_classifier(self, base_models):
        """Create a stacking classifier ensemble"""
        try:
            from sklearn.ensemble import StackingClassifier
            
            estimators = [(name, model) for name, model in base_models.items()]
            stacking_clf = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(random_state=42),
                cv=3,
                n_jobs=-1
            )
            return stacking_clf
        except ImportError:
            print("StackingClassifier not available, using VotingClassifier instead")
            return self.create_voting_classifier(base_models)
    
    def tune_hyperparameters(self, model, X, y, param_grid, cv=3):
        """Tune hyperparameters using GridSearchCV"""
        scorer = make_scorer(f1_score, average='micro', zero_division=0)
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scorer, n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def get_hyperparameter_grids(self):
        """Get hyperparameter grids for different models"""
        param_grids = {
            'logistic_regression': {
                'estimator__C': [0.1, 1.0, 10.0],
                'estimator__penalty': ['l1', 'l2'],
                'estimator__solver': ['liblinear']
            },
            'random_forest': {
                'estimator__n_estimators': [50, 100, 200],
                'estimator__max_depth': [10, 20, None],
                'estimator__min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'estimator__n_estimators': [50, 100, 200],
                'estimator__max_depth': [3, 6, 10],
                'estimator__learning_rate': [0.01, 0.1, 0.2]
            }
        }
        return param_grids
    
    def train_models(self, X_train, y_train, X_val=None, y_val=None, 
                    cluster_features_train=None, cluster_features_val=None,
                    tune_hyperparams=True, fast_mode=False):
        """Train all models and find the best one"""
        
        print("Creating base models...")
        base_models = self.create_base_models()
        
        # Train base models
        for name, model in base_models.items():
            print(f"\nTraining {name}...")
            
            if tune_hyperparams:
                param_grids = self.get_hyperparameter_grids()
                if name in param_grids:
                    print(f"Tuning hyperparameters for {name}...")
                    tuned_model, best_params, best_score = self.tune_hyperparameters(
                        model, X_train, y_train, param_grids[name]
                    )
                    self.models[name] = tuned_model
                    print(f"Best parameters for {name}: {best_params}")
                    print(f"Best CV score for {name}: {best_score:.4f}")
                else:
                    model.fit(X_train, y_train)
                    self.models[name] = model
            else:
                model.fit(X_train, y_train)
                self.models[name] = model
        
        # Create and train ensemble models
        print("\nTraining ensemble models...")
        
        # Voting classifier
        voting_clf = self.create_voting_classifier(base_models)
        voting_clf.fit(X_train, y_train)
        self.models['voting_classifier'] = voting_clf
        
        # Stacking classifier
        stacking_clf = self.create_stacking_classifier(base_models)
        stacking_clf.fit(X_train, y_train)
        self.models['stacking_classifier'] = stacking_clf
        
        # Hybrid models (if cluster features are provided)
        if cluster_features_train is not None:
            print("\nTraining hybrid models with clustering features...")
            
            for name, base_model in base_models.items():
                hybrid_name = f"hybrid_{name}"
                hybrid_model = HybridClassifier(base_model)
                hybrid_model.fit(X_train, y_train, cluster_features_train)
                self.models[hybrid_name] = hybrid_model
        
        # Evaluate models on validation set if provided
        if X_val is not None and y_val is not None:
            self._evaluate_on_validation(X_val, y_val, cluster_features_val)
    
    def _evaluate_on_validation(self, X_val, y_val, cluster_features_val=None):
        """Evaluate all models on validation set"""
        print("\nEvaluating models on validation set...")
        
        for name, model in self.models.items():
            try:
                if name.startswith('hybrid_') and cluster_features_val is not None:
                    y_pred = model.predict(X_val, cluster_features_val)
                else:
                    y_pred = model.predict(X_val)
                
                # Calculate F1 score
                f1 = f1_score(y_val, y_pred, average='micro', zero_division=0)
                print(f"{name}: F1-score = {f1:.4f}")
                
                if f1 > self.best_score:
                    self.best_score = f1
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
        
        print(f"\nBest model: {self.best_model_name} (F1-score: {self.best_score:.4f})")
    
    def predict(self, X, model_name=None, cluster_features=None):
        """Make predictions using specified model or best model"""
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models.get(model_name)
            
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        if model_name.startswith('hybrid_') and cluster_features is not None:
            return model.predict(X, cluster_features)
        else:
            return model.predict(X)
    
    def predict_proba(self, X, model_name=None, cluster_features=None):
        """Get prediction probabilities"""
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models.get(model_name)
            
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        if model_name.startswith('hybrid_') and cluster_features is not None:
            return model.predict_proba(X, cluster_features)
        else:
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)
            else:
                return None
    
    def save_models(self, save_dir):
        """Save all trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{name}.joblib")
            joblib.dump(model, model_path)
            print(f"Saved {name} to {model_path}")
        
        # Save best model info
        best_model_info = {
            'best_model_name': self.best_model_name,
            'best_score': self.best_score
        }
        info_path = os.path.join(save_dir, "best_model_info.joblib")
        joblib.dump(best_model_info, info_path)
        print(f"Saved best model info to {info_path}")
    
    def load_models(self, save_dir):
        """Load trained models"""
        # Load best model info
        info_path = os.path.join(save_dir, "best_model_info.joblib")
        if os.path.exists(info_path):
            best_model_info = joblib.load(info_path)
            self.best_model_name = best_model_info['best_model_name']
            self.best_score = best_model_info['best_score']
        
        # Load individual models
        model_files = [f for f in os.listdir(save_dir) if f.endswith('.joblib') and f != 'best_model_info.joblib']
        
        for model_file in model_files:
            model_name = model_file.replace('.joblib', '')
            model_path = os.path.join(save_dir, model_file)
            self.models[model_name] = joblib.load(model_path)
            print(f"Loaded {model_name} from {model_path}")
        
        # Set best model
        if self.best_model_name and self.best_model_name in self.models:
            self.best_model = self.models[self.best_model_name]
    
    def get_model_names(self):
        """Get list of trained model names"""
        return list(self.models.keys())
    
    def cross_validate_model(self, model, X, y, cv=5):
        """Perform cross-validation on a specific model"""
        scorer = make_scorer(f1_score, average='micro', zero_division=0)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        return scores
    
    def train_models(self, X_train, y_train, X_val=None, y_val=None, 
                    cluster_features_train=None, cluster_features_val=None,
                    tune_hyperparams=True, fast_mode=False):
        """Train multiple classification models"""
        if models_to_train is None:
            models_to_train = ['logistic_regression', 'random_forest', 'svm']
        
        # Skip SVM in fast mode as it's slow
        if fast_mode and 'svm' in models_to_train:
            models_to_train = [m for m in models_to_train if m != 'svm']
            print("Fast mode: Skipping SVM for speed")
        
        for model_name in models_to_train:
            print(f"Training {model_name}...")
            
            if model_name == 'logistic_regression':
                # Reduce max_iter in fast mode
                max_iter = 500 if fast_mode else 1000
                model = MultiOutputClassifier(
                    LogisticRegression(random_state=42, max_iter=max_iter, solver='liblinear')
                )
            elif model_name == 'random_forest':
                # Reduce n_estimators in fast mode
                n_estimators = 50 if fast_mode else 100
                model = MultiOutputClassifier(
                    RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
                )
            elif model_name == 'svm':
                model = MultiOutputClassifier(
                    SVC(kernel='linear', random_state=42, probability=True)
                )
            else:
                continue
            
            # Train model
            model.fit(X_train, y_train)
            self.models[model_name] = model
            
            # Validate if validation set provided
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                f1_score = f1_score_multilabel(y_val, y_pred, average='micro')
                print(f"{model_name} validation F1-score: {f1_score:.4f}")
    
    def train_hybrid_models(self, X_train_hybrid, y_train, X_val_hybrid=None, y_val=None, 
                          cluster_labels=None, models_to_train=None, fast_mode=False):
        """Train hybrid models with cluster features"""
        if models_to_train is None:
            models_to_train = ['logistic_regression', 'random_forest', 'svm']
        
        # Skip SVM in fast mode
        if fast_mode and 'svm' in models_to_train:
            models_to_train = [m for m in models_to_train if m != 'svm']
        
        for model_name in models_to_train:
            print(f"Training hybrid {model_name}...")
            
            if model_name == 'logistic_regression':
                max_iter = 500 if fast_mode else 1000
                model = MultiOutputClassifier(
                    LogisticRegression(random_state=42, max_iter=max_iter, solver='liblinear')
                )
            elif model_name == 'random_forest':
                n_estimators = 50 if fast_mode else 100
                model = MultiOutputClassifier(
                    RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
                )
            elif model_name == 'svm':
                model = MultiOutputClassifier(
                    SVC(kernel='linear', random_state=42, probability=True)
                )
            else:
                continue
            
            # Train hybrid model
            model.fit(X_train_hybrid, y_train)
            self.hybrid_models[model_name] = model
            
            # Validate if validation set provided
            if X_val_hybrid is not None and y_val is not None:
                y_pred = model.predict(X_val_hybrid)
                f1_score = f1_score_multilabel(y_val, y_pred, average='micro')
                print(f"Hybrid {model_name} validation F1-score: {f1_score:.4f}")
