"""Data threshing pipeline for cleaning and filtering datasets."""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN
from PIL import Image


class DataThresher:
    """
    Data threshing pipeline implementing the FIT machine concept.
    Automatically filters out low-quality and mislabeled data.
    """
    
    def __init__(self, config):
        """
        Initialize data thresher.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize face detector
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_detector = MTCNN(
            keep_all=False,
            device=self.device,
            min_face_size=config.get('threshing.face_detection.min_face_size', 48)
        )
        
        # Quality thresholds
        self.min_brightness = config.get('threshing.quality.min_brightness', 20)
        self.max_brightness = config.get('threshing.quality.max_brightness', 235)
        self.min_contrast = config.get('threshing.quality.min_contrast', 0.3)
        self.blur_threshold = config.get('threshing.quality.blur_threshold', 100)
        self.face_confidence_threshold = config.get('threshing.face_detection.min_confidence', 0.95)
        
    def detect_face(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if image contains a valid face.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Tuple of (is_valid, confidence)
        """
        try:
            # Convert to PIL Image
            if len(image.shape) == 2:
                img_pil = Image.fromarray(image).convert('RGB')
            else:
                img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Detect face
            boxes, probs = self.face_detector.detect(img_pil)
            
            if boxes is not None and len(boxes) > 0:
                confidence = float(probs[0])
                is_valid = confidence >= self.face_confidence_threshold
                return is_valid, confidence
            else:
                return False, 0.0
                
        except Exception as e:
            return False, 0.0
    
    def calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate average brightness of image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return np.mean(gray)
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate contrast (standard deviation) of image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return np.std(gray) / 255.0
    
    def calculate_blur(self, image: np.ndarray) -> float:
        """
        Calculate blur score using Laplacian variance.
        Higher values = sharper image.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def check_brightness(self, image: np.ndarray) -> bool:
        """Check if brightness is within acceptable range."""
        brightness = self.calculate_brightness(image)
        return self.min_brightness <= brightness <= self.max_brightness
    
    def check_contrast(self, image: np.ndarray) -> bool:
        """Check if contrast is sufficient."""
        contrast = self.calculate_contrast(image)
        return contrast >= self.min_contrast
    
    def check_blur(self, image: np.ndarray) -> bool:
        """Check if image is not too blurry."""
        blur_score = self.calculate_blur(image)
        return blur_score >= self.blur_threshold
    
    def calculate_quality_score(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive quality metrics for an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            'brightness': self.calculate_brightness(image),
            'contrast': self.calculate_contrast(image),
            'blur_score': self.calculate_blur(image),
        }
        
        # Face detection
        has_face, face_confidence = self.detect_face(image)
        metrics['has_face'] = has_face
        metrics['face_confidence'] = face_confidence
        
        # Overall quality checks
        metrics['brightness_ok'] = self.check_brightness(image)
        metrics['contrast_ok'] = self.check_contrast(image)
        metrics['blur_ok'] = self.check_blur(image)
        
        # Overall quality score (0-1)
        quality_checks = [
            metrics['brightness_ok'],
            metrics['contrast_ok'],
            metrics['blur_ok'],
            metrics['has_face']
        ]
        metrics['quality_score'] = sum(quality_checks) / len(quality_checks)
        
        return metrics
    
    def filter_dataset(
        self,
        images: List[np.ndarray],
        labels: List[int],
        class_names: List[str] = None
    ) -> Tuple[List[np.ndarray], List[int], Dict]:
        """
        Filter dataset based on quality thresholds.
        
        Args:
            images: List of images
            labels: List of labels
            class_names: Optional list of class names
            
        Returns:
            Tuple of (filtered_images, filtered_labels, stats)
        """
        print("Starting data threshing pipeline...")
        
        filtered_images = []
        filtered_labels = []
        
        # Statistics
        total_count = len(images)
        removed_reasons = {
            'no_face': 0,
            'low_brightness': 0,
            'low_contrast': 0,
            'too_blurry': 0,
            'multiple_issues': 0
        }
        
        class_stats = {}
        if class_names:
            for cls in class_names:
                class_stats[cls] = {'before': 0, 'after': 0, 'removed': 0}
        
        # Process each image
        for img, label in tqdm(zip(images, labels), total=len(images), desc="Threshing data"):
            # Track class statistics
            if class_names and label < len(class_names):
                cls_name = class_names[label]
                class_stats[cls_name]['before'] += 1
            
            # Calculate quality metrics
            metrics = self.calculate_quality_score(img)
            
            # Determine if image passes quality checks
            passes_quality = True
            issues = []
            
            if not metrics['has_face']:
                passes_quality = False
                issues.append('no_face')
            
            if not metrics['brightness_ok']:
                passes_quality = False
                issues.append('low_brightness')
            
            if not metrics['contrast_ok']:
                passes_quality = False
                issues.append('low_contrast')
            
            if not metrics['blur_ok']:
                passes_quality = False
                issues.append('too_blurry')
            
            # Keep or discard
            if passes_quality:
                filtered_images.append(img)
                filtered_labels.append(label)
                
                if class_names and label < len(class_names):
                    cls_name = class_names[label]
                    class_stats[cls_name]['after'] += 1
            else:
                # Track removal reasons
                if len(issues) == 1:
                    removed_reasons[issues[0]] += 1
                else:
                    removed_reasons['multiple_issues'] += 1
                
                if class_names and label < len(class_names):
                    cls_name = class_names[label]
                    class_stats[cls_name]['removed'] += 1
        
        # Calculate final statistics
        filtered_count = len(filtered_images)
        removed_count = total_count - filtered_count
        removal_rate = (removed_count / total_count) * 100 if total_count > 0 else 0
        
        stats = {
            'total_before': total_count,
            'total_after': filtered_count,
            'total_removed': removed_count,
            'removal_rate': removal_rate,
            'removed_reasons': removed_reasons,
            'class_stats': class_stats
        }
        
        print(f"\n{'='*60}")
        print(f"Data Threshing Results:")
        print(f"{'='*60}")
        print(f"Total samples before: {total_count}")
        print(f"Total samples after:  {filtered_count}")
        print(f"Samples removed:      {removed_count} ({removal_rate:.1f}%)")
        print(f"\nRemoval reasons:")
        for reason, count in removed_reasons.items():
            print(f"  {reason}: {count}")
        print(f"{'='*60}\n")
        
        return filtered_images, filtered_labels, stats
    
    def balance_classes(
        self,
        images: List[np.ndarray],
        labels: List[int],
        max_samples_per_class: int = None
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Balance classes by undersampling majority classes.
        
        Args:
            images: List of images
            labels: List of labels
            max_samples_per_class: Maximum samples per class
            
        Returns:
            Tuple of (balanced_images, balanced_labels)
        """
        if max_samples_per_class is None:
            max_samples_per_class = self.config.get('threshing.balance.max_samples_per_class', 5000)
        
        print(f"Balancing classes (max {max_samples_per_class} per class)...")
        
        # Group by class
        class_indices = {}
        for idx, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # Undersample each class
        balanced_indices = []
        for class_id, indices in class_indices.items():
            if len(indices) > max_samples_per_class:
                # Randomly sample
                sampled_indices = np.random.choice(indices, max_samples_per_class, replace=False)
                balanced_indices.extend(sampled_indices)
            else:
                balanced_indices.extend(indices)
        
        # Shuffle
        np.random.shuffle(balanced_indices)
        
        # Create balanced dataset
        balanced_images = [images[i] for i in balanced_indices]
        balanced_labels = [labels[i] for i in balanced_indices]
        
        print(f"Balanced dataset size: {len(balanced_images)}")
        
        return balanced_images, balanced_labels


if __name__ == "__main__":
    print("Data thresher module ready!")