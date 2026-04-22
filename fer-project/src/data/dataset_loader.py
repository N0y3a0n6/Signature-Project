"""
Updated dataset loader to match your exact folder structure.

Your structure:
- FER2013/train/, FER2013/test/ with emotion subfolders
- CK+48/ with emotion subfolders  
- jaffe/jaffe/ with .tiff files
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class FER2013Dataset(Dataset):
    """
    FER2013 dataset loader for YOUR structure:
    FER2013/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── sad/
    │   ├── surprise/
    │   └── neutral/
    └── test/
        └── (same structure)
    """
    
    def __init__(
        self, 
        data_dir: str,
        split: str = 'train',
        transform=None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # Emotion name to index mapping
        self.emotion_to_idx = {
            'angry': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'sad': 4,
            'surprise': 5,
            'neutral': 6
        }
        
        # Load data
        self.images, self.labels = self._load_data()
        
    def _load_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Load images and labels from directory structure."""
        images = []
        labels = []
        
        # Path is FER2013/train or FER2013/test
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Directory {split_dir} does not exist!")
        
        print(f"Loading from: {split_dir}")
        
        # Get emotion folders
        emotion_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        for emotion_dir in emotion_dirs:
            emotion_name = emotion_dir.name
            
            if emotion_name not in self.emotion_to_idx:
                print(f"Warning: Unknown emotion '{emotion_name}', skipping...")
                continue
            
            emotion_idx = self.emotion_to_idx[emotion_name]
            
            # Load all images in this emotion folder
            image_files = list(emotion_dir.glob('*.png')) + \
                         list(emotion_dir.glob('*.jpg')) + \
                         list(emotion_dir.glob('*.jpeg'))
            
            for img_path in tqdm(image_files, desc=f"Loading {emotion_name}", leave=False):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, self.image_size)
                    images.append(img)
                    labels.append(emotion_idx)
        
        print(f"Loaded {len(images)} images from {split_dir}")
        return images, labels
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        label = self.labels[idx]
        
        # Convert to RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class CKPlusDataset(Dataset):
    """
    CK+ dataset loader for YOUR structure:
    CK+48/
    ├── anger/
    ├── contempt/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sadness/
    └── surprise/
    """
    
    def __init__(
        self,
        data_dir: str,
        transform=None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_size = image_size
        
        # CK+48 uses different emotion names - map to standard
        self.emotion_to_idx = {
            'anger': 0,      # Maps to 'angry'
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'sadness': 4,    # Maps to 'sad'
            'surprise': 5,
            'contempt': 1,   # Map contempt to disgust (closest)
            # Note: CK+ doesn't have 'neutral'
        }
        
        self.images, self.labels = self._load_data()
    
    def _load_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Load CK+ images and labels."""
        images = []
        labels = []
        
        if not self.data_dir.exists():
            raise ValueError(f"Directory {self.data_dir} does not exist!")
        
        print(f"Loading from: {self.data_dir}")
        
        # Get emotion directories
        emotion_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for emotion_dir in emotion_dirs:
            emotion_name = emotion_dir.name
            
            if emotion_name not in self.emotion_to_idx:
                print(f"Warning: Unknown emotion '{emotion_name}', skipping...")
                continue
            
            emotion_idx = self.emotion_to_idx[emotion_name]
            
            # Load all images
            image_files = list(emotion_dir.glob('*.png')) + \
                         list(emotion_dir.glob('*.jpg'))
            
            for img_path in tqdm(image_files, desc=f"Loading CK+ {emotion_name}", leave=False):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, self.image_size)
                    images.append(img)
                    labels.append(emotion_idx)
        
        print(f"Loaded {len(images)} images from CK+")
        return images, labels
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        label = self.labels[idx]
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class JAFFEDataset(Dataset):
    """
    JAFFE dataset loader for YOUR structure:
    jaffe/
    ├── README_FIRST.txt
    ├── jaffe/ (subfolder with images)
    │   ├── KM.AN2.18.tiff
    │   ├── KM.AN3.19.tiff
    │   └── ...
    
    File naming: PersonID.EmotionCode.Number.tiff
    Example: KM.AN2.18.tiff = Person KM, Angry (AN), image 2.18
    """
    
    def __init__(
        self,
        data_dir: str,
        transform=None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_size = image_size
        
        # JAFFE emotion codes
        self.emotion_map = {
            'AN': 0,  # Angry
            'DI': 1,  # Disgust
            'FE': 2,  # Fear
            'HA': 3,  # Happy
            'SA': 4,  # Sad
            'SU': 5,  # Surprise
            'NE': 6   # Neutral
        }
        
        self.images, self.labels = self._load_data()
    
    def _load_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Load JAFFE images and labels from filenames."""
        images = []
        labels = []
        
        if not self.data_dir.exists():
            raise ValueError(f"Directory {self.data_dir} does not exist!")
        
        print(f"Loading from: {self.data_dir}")
        
        # Load all .tiff files
        image_files = list(self.data_dir.glob('*.tiff')) + \
                     list(self.data_dir.glob('*.tif'))
        
        if len(image_files) == 0:
            print(f"Warning: No .tiff files found in {self.data_dir}")
            return images, labels
        
        for img_path in tqdm(image_files, desc="Loading JAFFE"):
            # Parse emotion from filename
            # Format: KM.AN2.18.tiff -> parts = ['KM', 'AN2', '18', 'tiff']
            filename = img_path.stem  # Get filename without extension
            parts = filename.split('.')
            
            if len(parts) >= 2:
                # Extract emotion code (first 2 letters of second part)
                emotion_code = parts[1][:2].upper()
                
                if emotion_code in self.emotion_map:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, self.image_size)
                        images.append(img)
                        labels.append(self.emotion_map[emotion_code])
                else:
                    print(f"Warning: Unknown emotion code '{emotion_code}' in {img_path.name}")
        
        print(f"Loaded {len(images)} images from JAFFE")
        return images, labels
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        label = self.labels[idx]
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_transforms(config, is_training: bool = True):
    """Get image transforms based on configuration."""
    normalize = transforms.Normalize(
        mean=config.get('image.normalize_mean'),
        std=config.get('image.normalize_std')
    )
    
    if is_training and config.get('augmentation.train.horizontal_flip', 0) > 0:
        transform_list = [
            transforms.Resize(config.get('image.size')),
            transforms.RandomHorizontalFlip(p=config.get('augmentation.train.horizontal_flip')),
            transforms.RandomRotation(config.get('augmentation.train.rotation_range', 0)),
            transforms.ColorJitter(
                brightness=config.get('augmentation.train.brightness_range', [1.0, 1.0]),
                contrast=[0.8, 1.2]
            ),
            transforms.ToTensor(),
            normalize
        ]
    else:
        transform_list = [
            transforms.Resize(config.get('image.size')),
            transforms.ToTensor(),
            normalize
        ]
    
    return transforms.Compose(transform_list)


def create_dataloaders(config, dataset_name: str = 'fer2013'):
    """Create train, validation, and test dataloaders."""
    from torch.utils.data import DataLoader, random_split
    
    train_transform = get_transforms(config, is_training=True)
    test_transform = get_transforms(config, is_training=False)
    
    batch_size = config.get('model.batch_size')
    
    if dataset_name == 'fer2013':
        data_dir = config.get('data.fer2013.path')
        train_dataset = FER2013Dataset(data_dir, split='train', transform=train_transform)
        
        # Check if 'val' folder exists
        val_path = Path(data_dir) / 'val'
        if val_path.exists():
            val_dataset = FER2013Dataset(data_dir, split='val', transform=test_transform)
        else:
            # Split from train if no separate val folder
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        test_dataset = FER2013Dataset(data_dir, split='test', transform=test_transform)
    
    elif dataset_name == 'ckplus':
        data_dir = config.get('data.ckplus.path')
        dataset = CKPlusDataset(data_dir, transform=test_transform)
        # Split into val and test
        val_size = len(dataset) // 2
        test_size = len(dataset) - val_size
        val_dataset, test_dataset = random_split(dataset, [val_size, test_size])
        train_dataset = None
    
    elif dataset_name == 'jaffe':
        data_dir = config.get('data.jaffe.path')
        dataset = JAFFEDataset(data_dir, transform=test_transform)
        # Split into val and test
        val_size = len(dataset) // 2
        test_size = len(dataset) - val_size
        val_dataset, test_dataset = random_split(dataset, [val_size, test_size])
        train_dataset = None
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    ) if train_dataset else None
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    ) if val_dataset else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    ) if test_dataset else None
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Updated dataset loader ready for your folder structure!")