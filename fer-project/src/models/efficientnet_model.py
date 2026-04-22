"""EfficientNet-based model for facial emotion recognition."""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from typing import Optional


class EfficientNetFER(nn.Module):
    """EfficientNet-based model for facial emotion recognition."""
    
    def __init__(
        self,
        model_name: str = 'efficientnet-b2',
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize EfficientNet FER model.
        
        Args:
            model_name: EfficientNet variant (b0-b7)
            num_classes: Number of emotion classes
            pretrained: Whether to use pretrained ImageNet weights
            dropout: Dropout rate
        """
        super(EfficientNetFER, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load base EfficientNet
        if pretrained:
            self.base_model = EfficientNet.from_pretrained(model_name)
        else:
            self.base_model = EfficientNet.from_name(model_name)
        
        # Get feature dimension
        in_features = self.base_model._fc.in_features
        
        # Replace classifier
        self.base_model._fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.base_model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before final classification layer."""
        # Extract features
        x = self.base_model.extract_features(x)
        x = self.base_model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x
    
    def freeze_backbone(self):
        """Freeze backbone weights for fine-tuning."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier
        for param in self.base_model._fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze all weights."""
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def count_parameters(self) -> dict:
        """Count trainable and total parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }


class EnsembleFER(nn.Module):
    """Ensemble of multiple FER models."""
    
    def __init__(
        self,
        models: list,
        voting: str = 'soft'
    ):
        """
        Initialize ensemble model.
        
        Args:
            models: List of trained models
            voting: 'soft' (average probabilities) or 'hard' (majority vote)
        """
        super(EnsembleFER, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.voting = voting
        self.num_models = len(models)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        if self.voting == 'soft':
            # Soft voting: average probabilities
            outputs = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    output = torch.softmax(model(x), dim=1)
                    outputs.append(output)
            
            # Average predictions
            ensemble_output = torch.stack(outputs).mean(dim=0)
            return ensemble_output
        
        else:  # hard voting
            # Hard voting: majority vote on predicted classes
            predictions = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    output = model(x)
                    pred = torch.argmax(output, dim=1)
                    predictions.append(pred)
            
            # Stack predictions and take mode
            predictions = torch.stack(predictions, dim=0)
            
            # Get mode (most common prediction)
            ensemble_pred = torch.mode(predictions, dim=0)[0]
            
            # Convert to one-hot for consistency
            num_classes = self.models[0].num_classes
            ensemble_output = torch.zeros(x.size(0), num_classes, device=x.device)
            ensemble_output.scatter_(1, ensemble_pred.unsqueeze(1), 1.0)
            
            return ensemble_output


def create_model(config, pretrained: bool = True) -> EfficientNetFER:
    """
    Create model from configuration.
    
    Args:
        config: Configuration object
        pretrained: Whether to use pretrained weights
        
    Returns:
        EfficientNetFER model
    """
    model = EfficientNetFER(
        model_name=config.get('model.architecture'),
        num_classes=config.get('emotions.num_classes'),
        pretrained=pretrained,
        dropout=config.get('model.dropout')
    )
    
    return model


def load_model(checkpoint_path: str, config, device: str = 'cuda') -> EfficientNetFER:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration object
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    model = create_model(config, pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = EfficientNetFER()
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    print(f"Model: {model.model_name}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    params = model.count_parameters()
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")