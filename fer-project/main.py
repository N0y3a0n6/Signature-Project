"""Main script for running the FER project pipeline."""

import argparse
import torch
from pathlib import Path

from src.utils.config import Config, setup_seed, create_directories
from src.data.dataset_loader import create_dataloaders
from src.models.efficientnet_model import create_model
from src.models.trainer import FERTrainer
from src.evaluation.cross_dataset_eval import load_and_evaluate


def train(config_path: str = 'config/config.yaml'):
    """Run training pipeline."""
    # Load config
    config = Config(config_path)
    setup_seed(config.get('project.random_seed'))
    create_directories(config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(config, dataset_name='fer2013')
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, pretrained=True)
    params = model.count_parameters()
    print(f"Model: {config.get('model.architecture')}")
    print(f"Total parameters: {params['total']:,}")
    
    # Create trainer
    emotions = config.get('emotions.classes')
    trainer = FERTrainer(model, config, device, class_names=emotions)
    
    # Train
    print("\nStarting training...")
    num_epochs = config.get('model.epochs')
    save_dir = config.get('output.models_dir') + '/checkpoints'
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_dir=save_dir
    )
    
    print("\nTraining complete!")
    print(f"Best model saved to: {save_dir}/best_model.pth")


def evaluate(model_path: str, config_path: str = 'config/config.yaml'):
    """Run cross-dataset evaluation."""
    # Load config
    config = Config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders for all datasets
    print("\nLoading datasets for evaluation...")
    dataset_loaders = {}
    
    # FER2013 test
    _, _, test_loader = create_dataloaders(config, dataset_name='fer2013')
    dataset_loaders['FER2013_Test'] = test_loader
    
    # CK+
    try:
        _, val_loader, test_loader = create_dataloaders(config, dataset_name='ckplus')
        dataset_loaders['CK+'] = test_loader
    except:
        print("CK+ dataset not available")
    
    # JAFFE
    try:
        _, val_loader, test_loader = create_dataloaders(config, dataset_name='jaffe')
        dataset_loaders['JAFFE'] = test_loader
    except:
        print("JAFFE dataset not available")
    
    # Evaluate
    print("\nEvaluating model...")
    results, cms, per_class = load_and_evaluate(
        model_path=model_path,
        config=config,
        dataset_loaders=dataset_loaders,
        device=device
    )
    
    print("\nEvaluation complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='FER Project - Training and Evaluation')
    parser.add_argument('mode', choices=['train', 'evaluate'], help='Mode to run')
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                       help='Path to config file')
    parser.add_argument('--model', type=str, help='Path to model checkpoint (for evaluation)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args.config)
    elif args.mode == 'evaluate':
        if not args.model:
            print("Error: --model argument required for evaluation mode")
            return
        evaluate(args.model, args.config)


if __name__ == "__main__":
    main()