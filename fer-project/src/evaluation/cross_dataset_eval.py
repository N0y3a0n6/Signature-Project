"""Cross-dataset evaluation utilities."""

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
from pathlib import Path

from ..utils.metrics import MetricsCalculator


class CrossDatasetEvaluator:
    """Evaluate model performance across multiple datasets."""
    
    def __init__(self, model, class_names: List[str], device: str = 'cuda'):
        """
        Initialize cross-dataset evaluator.
        
        Args:
            model: Trained model
            class_names: List of emotion class names
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.metrics_calculator = MetricsCalculator(class_names)
        
    def evaluate_dataset(
        self,
        data_loader: DataLoader,
        dataset_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model on a single dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        print(f"\nEvaluating on {dataset_name}...")
        
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        # Add dataset name
        metrics['dataset'] = dataset_name
        
        # Print summary
        print(f"\n{dataset_name} Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall:    {metrics['recall_macro']:.4f}")
        
        return metrics, np.array(all_labels), np.array(all_preds)
    
    def evaluate_all_datasets(
        self,
        dataset_loaders: Dict[str, DataLoader],
        save_dir: str = None
    ) -> Dict[str, Dict]:
        """
        Evaluate model on all datasets.
        
        Args:
            dataset_loaders: Dict of dataset_name -> DataLoader
            save_dir: Directory to save results
            
        Returns:
            Dictionary of dataset_name -> metrics
        """
        results = {}
        confusion_matrices = {}
        
        for dataset_name, data_loader in dataset_loaders.items():
            metrics, y_true, y_pred = self.evaluate_dataset(data_loader, dataset_name)
            results[dataset_name] = metrics
            
            # Get confusion matrix
            cm = self.metrics_calculator.get_confusion_matrix(y_true, y_pred)
            confusion_matrices[dataset_name] = cm
            
            # Save confusion matrix plot
            if save_dir:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                
                cm_path = save_path / f'{dataset_name}_confusion_matrix.png'
                self.metrics_calculator.plot_confusion_matrix(
                    y_true, y_pred,
                    normalize=True,
                    save_path=str(cm_path),
                    title=f'{dataset_name} Confusion Matrix'
                )
        
        # Print comparison summary
        print("\n" + "="*70)
        print("Cross-Dataset Evaluation Summary")
        print("="*70)
        print(f"{'Dataset':<20} {'Accuracy':<12} {'F1-Score':<12} {'Samples':<10}")
        print("-"*70)
        
        for dataset_name, metrics in results.items():
            print(f"{dataset_name:<20} {metrics['accuracy']:<12.4f} "
                  f"{metrics['f1_macro']:<12.4f} "
                  f"{len(dataset_loaders[dataset_name].dataset):<10}")
        
        print("="*70)
        
        return results, confusion_matrices
    
    def compare_per_class_performance(
        self,
        dataset_loaders: Dict[str, DataLoader]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare per-class performance across datasets.
        
        Args:
            dataset_loaders: Dict of dataset_name -> DataLoader
            
        Returns:
            Dictionary of class_name -> {dataset_name: f1_score}
        """
        per_class_results = {cls: {} for cls in self.class_names}
        
        for dataset_name, data_loader in dataset_loaders.items():
            metrics, _, _ = self.evaluate_dataset(data_loader, dataset_name)
            
            # Extract per-class F1 scores
            for cls in self.class_names:
                f1_key = f'f1_{cls}'
                if f1_key in metrics:
                    per_class_results[cls][dataset_name] = metrics[f1_key]
        
        # Print per-class comparison
        print("\n" + "="*80)
        print("Per-Class Performance Comparison")
        print("="*80)
        
        # Header
        datasets = list(dataset_loaders.keys())
        header = f"{'Emotion':<15}"
        for ds in datasets:
            header += f"{ds:<15}"
        print(header)
        print("-"*80)
        
        # Per-class rows
        for cls in self.class_names:
            row = f"{cls:<15}"
            for ds in datasets:
                score = per_class_results[cls].get(ds, 0.0)
                row += f"{score:<15.4f}"
            print(row)
        
        print("="*80)
        
        return per_class_results


def load_and_evaluate(
    model_path: str,
    config,
    dataset_loaders: Dict[str, DataLoader],
    device: str = 'cuda'
):
    """
    Load a trained model and evaluate on multiple datasets.
    
    Args:
        model_path: Path to model checkpoint
        config: Configuration object
        dataset_loaders: Dict of dataset_name -> DataLoader
        device: Device to run on
    """
    from ..models.efficientnet_model import create_model
    
    # Load model
    model = create_model(config, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create evaluator
    class_names = config.get('emotions.classes')
    evaluator = CrossDatasetEvaluator(model, class_names, device)
    
    # Evaluate
    results, cms = evaluator.evaluate_all_datasets(dataset_loaders)
    per_class = evaluator.compare_per_class_performance(dataset_loaders)
    
    return results, cms, per_class


if __name__ == "__main__":
    print("Cross-dataset evaluation module ready!")