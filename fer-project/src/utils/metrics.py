"""Metrics and evaluation utilities."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """Calculate and track various evaluation metrics."""
    
    def __init__(self, class_names: List[str]):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: List of emotion class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_prob: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        return metrics
    
    def get_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Calculate confusion matrix."""
        return confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = True,
        save_path: str = None,
        title: str = "Confusion Matrix"
    ):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the matrix
            save_path: Path to save the figure
            title: Plot title
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        return_dict: bool = True
    ):
        """Get detailed classification report."""
        return classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names,
            output_dict=return_dict,
            zero_division=0
        )
    
    def calculate_per_class_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate accuracy for each class."""
        cm = confusion_matrix(y_true, y_pred)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        return {
            class_name: acc 
            for class_name, acc in zip(self.class_names, per_class_acc)
        }


class MetricsTracker:
    """Track metrics across training epochs."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
    def update(self, metrics: Dict[str, float], epoch: int):
        """Update metrics for an epoch."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get_best_epoch(self, metric: str = 'val_acc', mode: str = 'max') -> Tuple[int, float]:
        """Get epoch with best metric value."""
        values = self.history.get(metric, [])
        if not values:
            return -1, None
            
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
            
        return best_idx, values[best_idx]
    
    def plot_history(self, metrics: List[str] = None, save_path: str = None):
        """Plot training history."""
        if metrics is None:
            metrics = ['loss', 'acc']
        
        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
        
        if num_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            
            if train_key in self.history:
                ax.plot(self.history[train_key], label=f'Train {metric}')
            if val_key in self.history:
                ax.plot(self.history[val_key], label=f'Val {metric}')
                
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} over Epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save(self, path: str):
        """Save history to file."""
        np.save(path, self.history)
    
    def load(self, path: str):
        """Load history from file."""
        self.history = np.load(path, allow_pickle=True).item()


if __name__ == "__main__":
    # Test metrics
    emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    calc = MetricsCalculator(emotions)
    
    # Dummy data
    y_true = np.random.randint(0, 7, 100)
    y_pred = np.random.randint(0, 7, 100)
    
    metrics = calc.calculate_metrics(y_true, y_pred)
    print("Metrics:", metrics)