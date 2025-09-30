"""
Example usage script for Deep Image Homography Estimation
This script demonstrates the complete workflow from data preparation to testing
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.homography_net import HomographyNet
from utils.data_preprocessing import HomographyDataset, create_sample_dataset


def example_workflow():
    """Demonstrate the complete workflow"""
    
    print("=== Deep Image Homography Estimation Example ===\n")
    
    # Step 1: Create sample dataset
    print("1. Creating sample dataset...")
    sample_csv = "data/example_dataset.csv"
    os.makedirs("data", exist_ok=True)
    create_sample_dataset(sample_csv, num_samples=20)
    print(f"   ✓ Sample dataset created: {sample_csv}")
    
    # Step 2: Load dataset
    print("\n2. Loading dataset...")
    try:
        dataset = HomographyDataset(sample_csv, patch_size=128, max_perturbation=32)
        print(f"   ✓ Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        print(f"   ✗ Dataset loading failed: {e}")
        print("   Note: This is expected with dummy image paths")
        return
    
    # Step 3: Create model
    print("\n3. Creating HomographyNet model...")
    model = HomographyNet(input_channels=2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   ✓ Model created")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    
    # Step 4: Test forward pass
    print("\n4. Testing model forward pass...")
    model.eval()
    
    # Create dummy input (batch_size=2, channels=2, height=128, width=128)
    dummy_input = torch.randn(2, 2, 128, 128)
    
    with torch.no_grad():
        output = model(dummy_input)
        homography_matrices = model.predict_homography_matrix(output)
    
    print(f"   ✓ Forward pass successful")
    print(f"   - Input shape: {dummy_input.shape}")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Homography matrices shape: {homography_matrices.shape}")
    
    # Step 5: Show example homography matrix
    print(f"\n5. Example homography matrix:")
    print(f"   H = {homography_matrices[0].numpy()}")
    
    # Step 6: Training command example
    print(f"\n6. To train the model, run:")
    print(f"   python scripts/train.py \\")
    print(f"       --dataset_csv {sample_csv} \\")
    print(f"       --batch_size 32 \\")
    print(f"       --epochs 50 \\")
    print(f"       --lr 0.005 \\")
    print(f"       --output_dir results/example_training")
    
    # Step 7: Testing command example  
    print(f"\n7. To test the trained model, run:")
    print(f"   python scripts/test_and_visualize.py \\")
    print(f"       --checkpoint results/example_training/best_model.pth \\")
    print(f"       --dataset_csv {sample_csv} \\")
    print(f"       --output_dir results/test_results \\")
    print(f"       --num_samples 5")
    
    print(f"\n=== Example completed successfully! ===")
    
    # Step 8: Show next steps
    print(f"\nNext steps for your car calibration project:")
    print(f"1. Replace the sample dataset with your actual car images")
    print(f"2. Manually annotate the fixed points on car bodies in the CSV")
    print(f"3. Run the training script with your dataset")
    print(f"4. Evaluate results and adjust hyperparameters as needed")
    print(f"5. Use the trained model for real-time calibration")


def show_csv_format():
    """Show the expected CSV format"""
    print("\n=== CSV Format Example ===")
    print("Your CSV file should have these columns:")
    
    columns = [
        'reference_image_path', 'tilted_image_path',
        'ref_point1_x', 'ref_point1_y', 'ref_point2_x', 'ref_point2_y',
        'ref_point3_x', 'ref_point3_y', 'ref_point4_x', 'ref_point4_y',
        'tilted_point1_x', 'tilted_point1_y', 'tilted_point2_x', 'tilted_point2_y',
        'tilted_point3_x', 'tilted_point3_y', 'tilted_point4_x', 'tilted_point4_y'
    ]
    
    for i, col in enumerate(columns):
        print(f"{i+1:2d}. {col}")
    
    print(f"\nExample row:")
    example_row = [
        "/path/to/ref_car_001.jpg", "/path/to/tilted_car_001.jpg",
        120, 150, 350, 145, 355, 280, 125, 285,  # Reference points
        115, 155, 355, 140, 350, 285, 130, 290   # Tilted points
    ]
    
    for i, (col, val) in enumerate(zip(columns, example_row)):
        print(f"   {col}: {val}")


if __name__ == "__main__":
    # Show CSV format first
    show_csv_format()
    
    # Run example workflow
    example_workflow()