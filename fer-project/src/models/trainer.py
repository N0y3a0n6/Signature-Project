"""Training utilities for FER models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
import time

from ..utils.metrics import MetricsCalculator, MetricsTracker


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1):
        """
        Initialize label smoothing loss.
        
        Args:
            smoothing: Smoothing factor (0.0 = no smoothing)
        """
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate label smoothing loss."""
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss


class FERTrainer:
    """Trainer for facial emotion recognition models."""
    
    def __init__(
        self,
        model: nn.Module,
        config,
        device: str = 'cuda',
        class_names: list = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Configuration object
            device: Device to train on
            class_names: List of emotion class names
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.class_names = class_names or config.get('emotions.classes')
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup loss function
        self.criterion = self._create_criterion()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Metrics
        self.metrics_calculator = MetricsCalculator(self.class_names)
        self.metrics_tracker = MetricsTracker()
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config."""
        lr = self.config.get('model.learning_rate')
        weight_decay = self.config.get('model.weight_decay')
        
        optimizer_name = self.config.get('model.optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function from config."""
        label_smoothing = self.config.get('model.label_smoothing', 0.0)
        
        if label_smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            return nn.CrossEntropyLoss()
    
    def _create_scheduler(self):
        """Create learning rate scheduler from config."""
        patience = self.config.get('model.patience', 10)
        factor = self.config.get('model.factor', 0.5)
        
        return ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=patience,
            factor=factor,
            verbose=True
        )
    
    def calculate_class_weights(self, train_loader: DataLoader) -> torch.Tensor:
        """
        Calculate class weights for handling imbalance.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Class weights tensor
        """
        # Count samples per class
        class_counts = torch.zeros(len(self.class_names))
        
        for _, labels in train_loader:
            for label in labels:
                class_counts[label] += 1
        
        # Calculate weights (inverse frequency)
        total_samples = class_counts.sum()
        class_weights = total_samples / (len(self.class_names) * class_counts)
        
        # Clip weights if specified
        if self.config.get('model.clip_weights', False):
            max_weight = self.config.get('model.max_weight', 5.0)
            class_weights = torch.clamp(class_weights, max=max_weight)
        
        return class_weights.to(self.device)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics — keep on device, move to CPU once at epoch end
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.detach())
            all_labels.append(labels.detach())

            # Update progress bar
            pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
        
        # Calculate epoch metrics — single CPU transfer per epoch
        epoch_loss = running_loss / len(train_loader)
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        metrics = self.metrics_calculator.calculate_metrics(all_labels, all_preds)
        
        metrics['loss'] = epoch_loss
        
        return metrics
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Track metrics — keep on device, move to CPU once at epoch end
                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.detach())
                all_labels.append(labels.detach())

                # Update progress bar
                pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})

        # Calculate epoch metrics — single CPU transfer per epoch
        epoch_loss = running_loss / len(val_loader)
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        metrics = self.metrics_calculator.calculate_metrics(all_labels, all_preds)
        
        metrics['loss'] = epoch_loss
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: str = 'models/checkpoints'
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print("="*60)
        print("Starting Training")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print("="*60)
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Update scheduler
            self.scheduler.step(val_metrics['accuracy'])
            
            # Track metrics
            self.metrics_tracker.update({
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }, epoch)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{num_epochs} - {epoch_time:.1f}s")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                
                checkpoint_path = save_path / 'best_model.pth'
                self.save_checkpoint(checkpoint_path, epoch, val_metrics)
                print(f"  ✓ New best model saved! (Val Acc: {self.best_val_acc:.4f})")
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                checkpoint_path = save_path / f'checkpoint_epoch_{epoch}.pth'
                self.save_checkpoint(checkpoint_path, epoch, val_metrics)
            
            print("-"*60)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        print("="*60)
        
        return self.metrics_tracker
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.config
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']


if __name__ == "__main__":
    print("Trainer module ready!")