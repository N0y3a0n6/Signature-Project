"""Visualization utilities for data analysis and results."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Tuple
import cv2
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_class_distribution(
    class_counts: Dict[str, int],
    title: str = "Class Distribution",
    save_path: str = None
):
    """
    Plot class distribution as a bar chart.
    
    Args:
        class_counts: Dictionary of class names to counts
        title: Plot title
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    colors = sns.color_palette("husl", len(classes))
    bars = plt.bar(classes, counts, color=colors, edgecolor='black', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontweight='bold'
        )
    
    plt.xlabel('Emotion Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_multi_dataset_distribution(
    dataset_distributions: Dict[str, Dict[str, int]],
    title: str = "Class Distribution Across Datasets",
    save_path: str = None
):
    """
    Plot class distributions for multiple datasets.
    
    Args:
        dataset_distributions: Dict of dataset_name -> {class: count}
        title: Plot title
        save_path: Path to save the figure
    """
    # Get all unique classes
    all_classes = set()
    for dist in dataset_distributions.values():
        all_classes.update(dist.keys())
    all_classes = sorted(list(all_classes))
    
    # Prepare data
    num_datasets = len(dataset_distributions)
    x = np.arange(len(all_classes))
    width = 0.8 / num_datasets
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = sns.color_palette("husl", num_datasets)
    
    for i, (dataset_name, dist) in enumerate(dataset_distributions.items()):
        counts = [dist.get(cls, 0) for cls in all_classes]
        offset = width * (i - num_datasets/2 + 0.5)
        ax.bar(x + offset, counts, width, label=dataset_name, 
               color=colors[i], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Emotion Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_sample_images(
    images: List[np.ndarray],
    labels: List[str],
    predictions: List[str] = None,
    title: str = "Sample Images",
    save_path: str = None,
    grid_size: Tuple[int, int] = (4, 8)
):
    """
    Plot a grid of sample images.
    
    Args:
        images: List of images (numpy arrays)
        labels: List of true labels
        predictions: List of predicted labels (optional)
        title: Plot title
        save_path: Path to save the figure
        grid_size: (rows, cols) for the grid
    """
    rows, cols = grid_size
    num_images = min(len(images), rows * cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    
    for i in range(num_images):
        img = images[i]
        
        # Handle grayscale or RGB
        if len(img.shape) == 2:
            axes[i].imshow(img, cmap='gray')
        else:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Create label
        label_text = f"True: {labels[i]}"
        if predictions and i < len(predictions):
            label_text += f"\nPred: {predictions[i]}"
            # Color the border based on correctness
            color = 'green' if labels[i] == predictions[i] else 'red'
            for spine in axes[i].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)
        
        axes[i].set_title(label_text, fontsize=8)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_quality_metrics(
    metrics_dict: Dict[str, List[float]],
    title: str = "Image Quality Metrics",
    save_path: str = None
):
    """
    Plot distributions of quality metrics.
    
    Args:
        metrics_dict: Dict of metric_name -> list of values
        title: Plot title
        save_path: Path to save the figure
    """
    num_metrics = len(metrics_dict)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, (metric_name, values) in zip(axes, metrics_dict.items()):
        ax.hist(values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(values), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(values):.2f}')
        ax.axvline(np.median(values), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(values):.2f}')
        
        ax.set_xlabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_data_threshing_results(
    before_counts: Dict[str, int],
    after_counts: Dict[str, int],
    removed_counts: Dict[str, int],
    save_path: str = None
):
    """
    Plot before/after comparison of data threshing.
    
    Args:
        before_counts: Class counts before threshing
        after_counts: Class counts after threshing
        removed_counts: Number of samples removed per class
        save_path: Path to save the figure
    """
    classes = list(before_counts.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Before/After comparison
    x = np.arange(len(classes))
    width = 0.35
    
    before_vals = [before_counts[c] for c in classes]
    after_vals = [after_counts[c] for c in classes]
    
    axes[0].bar(x - width/2, before_vals, width, label='Before', 
                color='lightcoral', edgecolor='black', alpha=0.8)
    axes[0].bar(x + width/2, after_vals, width, label='After', 
                color='lightgreen', edgecolor='black', alpha=0.8)
    
    axes[0].set_xlabel('Emotion Class', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    axes[0].set_title('Dataset Size Before/After Threshing', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Removed samples
    removed_vals = [removed_counts[c] for c in classes]
    removal_pct = [(removed_counts[c] / before_counts[c] * 100) if before_counts[c] > 0 else 0 
                   for c in classes]
    
    bars = axes[1].bar(classes, removed_vals, color='salmon', edgecolor='black', alpha=0.8)
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, removal_pct)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    axes[1].set_xlabel('Emotion Class', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Samples Removed', fontsize=12, fontweight='bold')
    axes[1].set_title('Samples Removed by Class', fontsize=13, fontweight='bold')
    axes[1].set_xticklabels(classes, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cross_dataset_performance(
    results: Dict[str, Dict[str, float]],
    metric: str = 'accuracy',
    save_path: str = None
):
    """
    Plot performance across different datasets.
    
    Args:
        results: Dict of dataset_name -> {metric: value}
        metric: Metric to plot
        save_path: Path to save the figure
    """
    datasets = list(results.keys())
    values = [results[d].get(metric, 0) for d in datasets]
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(datasets))
    bars = plt.bar(datasets, values, color=colors, edgecolor='black', alpha=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Dataset', fontsize=12, fontweight='bold')
    plt.ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
    plt.title(f'{metric.capitalize()} Across Datasets', fontsize=14, fontweight='bold')
    plt.ylim([0, 1.0])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


if __name__ == "__main__":
    # Test visualizations
    emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    
    # Test class distribution
    class_counts = {e: np.random.randint(1000, 5000) for e in emotions}
    plot_class_distribution(class_counts, save_path="test_dist.png")
    print("Visualization tests complete!")