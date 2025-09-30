"""
Data preprocessing utilities for Deep Image Homography Estimation
"""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from typing import Tuple, List, Dict, Optional
import random


class HomographyDataset(Dataset):
    """
    Dataset class for homography estimation with known corresponding points
    """
    
    def __init__(self, csv_file: str, patch_size: int = 128, max_perturbation: int = 32,
                 grayscale: bool = True, augment: bool = True):
        """
        Initialize the dataset
        
        Args:
            csv_file: Path to CSV file with image pairs and coordinates
            patch_size: Size of patches to extract (default: 128x128)
            max_perturbation: Maximum random perturbation for patch extraction
            grayscale: Convert images to grayscale
            augment: Apply data augmentation
        """
        self.data = pd.read_csv(csv_file)
        self.patch_size = patch_size
        self.max_perturbation = max_perturbation
        self.grayscale = grayscale
        self.augment = augment
        
        # Verify all image files exist
        self._verify_images()
    
    def _verify_images(self):
        """Verify that all image files in the CSV exist"""
        missing_files = []
        for idx, row in self.data.iterrows():
            if not os.path.exists(row['reference_image_path']):
                missing_files.append(row['reference_image_path'])
            if not os.path.exists(row['tilted_image_path']):
                missing_files.append(row['tilted_image_path'])
        
        if missing_files:
            print(f"Warning: {len(missing_files)} image files not found:")
            for file in missing_files[:5]:  # Show first 5
                print(f"  {file}")
            if len(missing_files) > 5:
                print(f"  ... and {len(missing_files) - 5} more")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            dict: Contains 'patch_pair', 'homography_params', 'corners'
        """
        row = self.data.iloc[idx]
        
        # Load images
        ref_img = self._load_image(row['reference_image_path'])
        tilted_img = self._load_image(row['tilted_image_path'])
        
        # Extract corresponding points
        ref_points = np.array([
            [row['ref_point1_x'], row['ref_point1_y']],
            [row['ref_point2_x'], row['ref_point2_y']],
            [row['ref_point3_x'], row['ref_point3_y']],
            [row['ref_point4_x'], row['ref_point4_y']]
        ], dtype=np.float32)
        
        tilted_points = np.array([
            [row['tilted_point1_x'], row['tilted_point1_y']],
            [row['tilted_point2_x'], row['tilted_point2_y']],
            [row['tilted_point3_x'], row['tilted_point3_y']],
            [row['tilted_point4_x'], row['tilted_point4_y']]
        ], dtype=np.float32)
        
        # Compute ground truth homography
        homography_matrix = cv2.getPerspectiveTransform(tilted_points, ref_points)
        
        # Extract patches and create training data
        patch_pair, corners, adjusted_homography = self._create_training_sample(
            ref_img, tilted_img, ref_points, tilted_points, homography_matrix
        )
        
        # Convert homography matrix to 8-parameter vector
        homography_params = self._matrix_to_params(adjusted_homography)
        
        return {
            'patch_pair': torch.FloatTensor(patch_pair),
            'homography_params': torch.FloatTensor(homography_params),
            'corners': torch.FloatTensor(corners)
        }
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess an image"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            if self.grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return img.astype(np.float32) / 255.0
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image to prevent crashes during development
            if self.grayscale:
                return np.zeros((480, 640), dtype=np.float32)
            else:
                return np.zeros((480, 640, 3), dtype=np.float32)
    
    def _create_training_sample(self, ref_img: np.ndarray, tilted_img: np.ndarray,
                              ref_points: np.ndarray, tilted_points: np.ndarray,
                              homography_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create training sample by extracting patches and computing relative homography
        
        Following the DeTone et al. methodology:
        1. Extract a patch from reference image
        2. Apply random perturbation to create "tilted" patch coordinates
        3. Extract corresponding patch from tilted image
        4. Compute homography between the patches
        """
        h, w = ref_img.shape[:2]
        
        # Define patch extraction region (avoid borders)
        border = self.patch_size // 2 + self.max_perturbation
        
        # Randomly select patch center from reference image
        center_x = random.randint(border, w - border)
        center_y = random.randint(border, h - border)
        
        # Define reference patch corners
        half_patch = self.patch_size // 2
        ref_corners = np.array([
            [center_x - half_patch, center_y - half_patch],  # Top-left
            [center_x + half_patch, center_y - half_patch],  # Top-right
            [center_x + half_patch, center_y + half_patch],  # Bottom-right
            [center_x - half_patch, center_y + half_patch]   # Bottom-left
        ], dtype=np.float32)
        
        # Apply random perturbation to create "tilted" patch corners
        perturbation = np.random.randint(-self.max_perturbation, self.max_perturbation + 1, (4, 2))
        tilted_corners = ref_corners + perturbation.astype(np.float32)
        
        # Extract patches
        ref_patch = self._extract_patch(ref_img, ref_corners)
        tilted_patch = self._extract_patch(tilted_img, tilted_corners)
        
        # Stack patches (reference first, then tilted)
        if len(ref_patch.shape) == 2:  # Grayscale
            patch_pair = np.stack([ref_patch, tilted_patch], axis=0)
        else:  # RGB
            patch_pair = np.concatenate([ref_patch, tilted_patch], axis=2)
            patch_pair = np.transpose(patch_pair, (2, 0, 1))
        
        # Compute homography from tilted patch to reference patch
        # This represents the transformation needed to align the patches
        patch_homography = cv2.getPerspectiveTransform(tilted_corners, ref_corners)
        
        return patch_pair, tilted_corners, patch_homography
    
    def _extract_patch(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Extract a patch from image using perspective transformation"""
        # Define target patch corners (square patch)
        target_corners = np.array([
            [0, 0],
            [self.patch_size - 1, 0],
            [self.patch_size - 1, self.patch_size - 1],
            [0, self.patch_size - 1]
        ], dtype=np.float32)
        
        # Compute transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(corners, target_corners)
        
        # Apply transformation
        patch = cv2.warpPerspective(img, transform_matrix, (self.patch_size, self.patch_size))
        
        return patch
    
    def _matrix_to_params(self, homography_matrix: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 homography matrix to 8-parameter vector
        
        Args:
            homography_matrix: 3x3 homography matrix
            
        Returns:
            8-element parameter vector [h1, h2, h3, h4, h5, h6, h7, h8]
        """
        # Normalize by h9 (bottom-right element)
        normalized_matrix = homography_matrix / homography_matrix[2, 2]
        
        # Extract 8 parameters (excluding h9 which is always 1)
        params = np.array([
            normalized_matrix[0, 0], normalized_matrix[0, 1], normalized_matrix[0, 2],
            normalized_matrix[1, 0], normalized_matrix[1, 1], normalized_matrix[1, 2],
            normalized_matrix[2, 0], normalized_matrix[2, 1]
        ], dtype=np.float32)
        
        return params


def compute_homography_from_points(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    Compute homography matrix from corresponding points
    
    Args:
        src_points: Source points (Nx2)
        dst_points: Destination points (Nx2)
        
    Returns:
        3x3 homography matrix
    """
    return cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points.astype(np.float32))


def create_sample_dataset(output_path: str, num_samples: int = 100):
    """
    Create a sample dataset CSV file for testing
    
    Args:
        output_path: Path to save the CSV file
        num_samples: Number of sample entries to create
    """
    data = []
    
    for i in range(num_samples):
        # Create sample data (you would replace these with actual image paths and coordinates)
        ref_img_path = f"/path/to/reference/image_{i:04d}.jpg"
        tilted_img_path = f"/path/to/tilted/image_{i:04d}.jpg"
        
        # Sample reference points (4 corners of a rectangle)
        ref_points = [
            [100 + i % 50, 100 + i % 30],  # Top-left
            [300 + i % 50, 100 + i % 30],  # Top-right
            [300 + i % 50, 200 + i % 30],  # Bottom-right
            [100 + i % 50, 200 + i % 30]   # Bottom-left
        ]
        
        # Sample tilted points (with some perturbation)
        tilted_points = [
            [ref_points[0][0] + random.randint(-20, 20), ref_points[0][1] + random.randint(-20, 20)],
            [ref_points[1][0] + random.randint(-20, 20), ref_points[1][1] + random.randint(-20, 20)],
            [ref_points[2][0] + random.randint(-20, 20), ref_points[2][1] + random.randint(-20, 20)],
            [ref_points[3][0] + random.randint(-20, 20), ref_points[3][1] + random.randint(-20, 20)]
        ]
        
        row = [ref_img_path, tilted_img_path] + \
              [coord for point in ref_points for coord in point] + \
              [coord for point in tilted_points for coord in point]
        
        data.append(row)
    
    # Create DataFrame and save
    columns = ['reference_image_path', 'tilted_image_path',
               'ref_point1_x', 'ref_point1_y', 'ref_point2_x', 'ref_point2_y',
               'ref_point3_x', 'ref_point3_y', 'ref_point4_x', 'ref_point4_y',
               'tilted_point1_x', 'tilted_point1_y', 'tilted_point2_x', 'tilted_point2_y',
               'tilted_point3_x', 'tilted_point3_y', 'tilted_point4_x', 'tilted_point4_y']
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created: {output_path}")


def test_dataset():
    """Test the dataset class"""
    # Create a sample dataset
    sample_csv = "/tmp/sample_dataset.csv"
    create_sample_dataset(sample_csv, num_samples=10)
    
    # Create dataset instance
    dataset = HomographyDataset(sample_csv)
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test loading a sample (this will fail with dummy paths, but shows the structure)
    try:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Patch pair shape: {sample['patch_pair'].shape}")
        print(f"Homography params shape: {sample['homography_params'].shape}")
        print(f"Corners shape: {sample['corners'].shape}")
    except Exception as e:
        print(f"Expected error with dummy paths: {e}")


if __name__ == "__main__":
    test_dataset()