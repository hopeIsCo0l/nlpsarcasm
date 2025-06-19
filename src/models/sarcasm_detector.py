"""
Main sarcasm detection model class.
Handles training, evaluation, and prediction for various ML models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.pipeline import Pipeline
import joblib
import pickle
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Fix the import to use absolute import
try:
    from preprocessing.text_preprocessor import TextPreprocessor
except ImportError:
    # Fallback for when running from different directory
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from preprocessing.text_preprocessor import TextPreprocessor

class SarcasmDetector:
    """
    Main class for sarcasm detection using various machine learning models.
    """
    
    def __init__(self, model_type: str = 'logistic_regression'):
        """
        Initialize the sarcasm detector.
        
        Args:
            model_type: Type of model to use ('logistic_regression', 'random_forest', 'svm', 'naive_bayes')
        """
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.preprocessor = TextPreprocessor()
        self.pipeline = None
        self.feature_names = None
        self.training_history = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Model configurations
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'model__C': [0.1, 1, 10, 100],
                    'model__penalty': ['l1', 'l2'],
                    'model__solver': ['liblinear', 'saga']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [10, 20, None],
                    'model__min_samples_split': [2, 5, 10]
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['linear', 'rbf'],
                    'model__gamma': ['scale', 'auto']
                }
            },
            'naive_bayes': {
                'model': MultinomialNB(),
                'params': {
                    'model__alpha': [0.1, 0.5, 1.0, 2.0]
                }
            }
        }
    
    def create_pipeline(self, use_tfidf: bool = True, max_features: int = 5000) -> Pipeline:
        """
        Create a scikit-learn pipeline for text classification.
        
        Args:
            use_tfidf: Whether to use TF-IDF vectorization
            max_features: Maximum number of features for vectorization
            
        Returns:
            Scikit-learn pipeline
        """
        # Choose vectorizer
        if use_tfidf:
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
        else:
            vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
        
        # Get model configuration
        model_config = self.model_configs[self.model_type]
        model = model_config['model']
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('model', model)
        ])
        
        return pipeline
    
    def prepare_data(self, df: pd.DataFrame, text_column: str = 'headline', 
                    target_column: str = 'is_sarcastic') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column
            target_column: Name of the target column
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Preprocess text
        self.logger.info("Preprocessing text data...")
        processed_texts = self.preprocessor.preprocess_batch(df[text_column].tolist())
        
        # Extract additional features
        self.logger.info("Extracting additional features...")
        additional_features = self.preprocessor.extract_features_batch(df[text_column].tolist())
        
        # Combine text and additional features
        X_text = np.array(processed_texts)
        X_features = additional_features.values
        
        # For now, we'll use only text features
        # TODO: Implement feature combination
        X = X_text
        y = df[target_column].values
        
        self.logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train(self, df: pd.DataFrame, text_column: str = 'headline', 
              target_column: str = 'is_sarcastic', test_size: float = 0.2,
              random_state: int = 42, use_grid_search: bool = True) -> Dict[str, Any]:
        """
        Train the sarcasm detection model.
        
        Args:
            df: Training DataFrame
            text_column: Name of the text column
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
            use_grid_search: Whether to use grid search for hyperparameter tuning
            
        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Starting training with {self.model_type} model...")
        
        # Prepare data
        X, y = self.prepare_data(df, text_column, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Create pipeline
        self.pipeline = self.create_pipeline()
        
        # Train model
        if use_grid_search:
            self.logger.info("Performing grid search for hyperparameter tuning...")
            model_config = self.model_configs[self.model_type]
            grid_search = GridSearchCV(
                self.pipeline,
                model_config['params'],
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            self.pipeline = grid_search.best_estimator_
            best_params = grid_search.best_params_
            self.logger.info(f"Best parameters: {best_params}")
        else:
            self.logger.info("Training with default parameters...")
            self.pipeline.fit(X_train, y_train)
            best_params = None
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Store training history
        self.training_history = {
            'model_type': self.model_type,
            'training_date': datetime.now().isoformat(),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'best_params': best_params,
            'metrics': metrics
        }
        
        self.logger.info(f"Training completed. Test accuracy: {metrics['accuracy']:.4f}")
        
        return {
            'metrics': metrics,
            'training_history': self.training_history,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate various performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with calculated metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict sarcasm for a single text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with prediction results
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess_text(text)
        
        # Make prediction
        prediction = self.pipeline.predict([processed_text])[0]
        probability = self.pipeline.predict_proba([processed_text])[0]
        
        return {
            'text': text,
            'processed_text': processed_text,
            'is_sarcastic': bool(prediction),
            'confidence': float(max(probability)),
            'sarcastic_probability': float(probability[1]),
            'non_sarcastic_probability': float(probability[0])
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sarcasm for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction results
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Preprocess texts
        processed_texts = self.preprocessor.preprocess_batch(texts)
        
        # Make predictions
        predictions = self.pipeline.predict(processed_texts)
        probabilities = self.pipeline.predict_proba(processed_texts)
        
        # Format results
        results = []
        for i, (text, processed_text, pred, prob) in enumerate(
            zip(texts, processed_texts, predictions, probabilities)
        ):
            results.append({
                'text': text,
                'processed_text': processed_text,
                'is_sarcastic': bool(pred),
                'confidence': float(max(prob)),
                'sarcastic_probability': float(prob[1]),
                'non_sarcastic_probability': float(prob[0])
            })
        
        return results
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation results
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-sarcastic', 'Sarcastic'],
                   yticklabels=['Non-sarcastic', 'Sarcastic'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance for the trained model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Get feature names
        feature_names = self.pipeline.named_steps['vectorizer'].get_feature_names_out()
        
        # Get feature importance
        if hasattr(self.pipeline.named_steps['model'], 'feature_importances_'):
            # For tree-based models
            importance = self.pipeline.named_steps['model'].feature_importances_
        elif hasattr(self.pipeline.named_steps['model'], 'coef_'):
            # For linear models
            importance = np.abs(self.pipeline.named_steps['model'].coef_[0])
        else:
            raise ValueError("Model doesn't support feature importance extraction")
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return feature_importance_df
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Save model
        joblib.dump(self.pipeline, filepath)
        
        # Save training history
        history_path = filepath.replace('.pkl', '_history.pkl')
        joblib.dump(self.training_history, history_path)
        
        self.logger.info(f"Model saved to {filepath}")
        self.logger.info(f"Training history saved to {history_path}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        # Load model
        self.pipeline = joblib.load(filepath)
        
        # Load training history
        history_path = filepath.replace('.pkl', '_history.pkl')
        try:
            self.training_history = joblib.load(history_path)
            self.model_type = self.training_history.get('model_type', 'unknown')
        except FileNotFoundError:
            self.logger.warning("Training history not found")
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        if self.pipeline is None:
            return {'status': 'not_trained'}
        
        info = {
            'status': 'trained',
            'model_type': self.model_type,
            'training_history': self.training_history,
            'vectorizer_type': type(self.pipeline.named_steps['vectorizer']).__name__,
            'model_type_name': type(self.pipeline.named_steps['model']).__name__
        }
        
        return info 