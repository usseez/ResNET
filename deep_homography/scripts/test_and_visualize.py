"""
Testing and visualization script for Deep Image Homography Estimation
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.homography_net import HomographyNet
from utils.data_preprocessing import HomographyDataset, compute_homography_from_points


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    model = HomographyNet(input_channels=2)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Model validation loss: {checkpoint.get('loss', 'unknown'):.6f}")
    
    return model


def params_to_matrix(params):
    """Convert 8-parameter vector to 3x3 homography matrix"""
    if torch.is_tensor(params):
        params = params.cpu().numpy()
    
    H = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [params[6], params[7], 1.0]
    ])
    return H


def apply_homography_to_image(image, homography_matrix, output_shape=None):
    """Apply homography transformation to an image"""
    if output_shape is None:
        output_shape = image.shape[:2][::-1]  # (width, height)
    
    transformed = cv2.warpPerspective(image, homography_matrix, output_shape)
    return transformed


def compute_corner_error(pred_corners, true_corners):
    """Compute average corner error in pixels"""
    if torch.is_tensor(pred_corners):
        pred_corners = pred_corners.cpu().numpy()
    if torch.is_tensor(true_corners):
        true_corners = true_corners.cpu().numpy()
    
    error = np.linalg.norm(pred_corners - true_corners, axis=1)
    return np.mean(error)


def visualize_homography_result(ref_img, tilted_img, pred_homography, true_homography, 
                               ref_points, tilted_points, save_path=None):
    """
    Visualize the homography estimation results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original images
    axes[0, 0].imshow(ref_img, cmap='gray' if len(ref_img.shape) == 2 else None)
    axes[0, 0].set_title('Reference Image')
    axes[0, 0].plot(ref_points[:, 0], ref_points[:, 1], 'ro', markersize=8)
    for i, (x, y) in enumerate(ref_points):
        axes[0, 0].annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', color='red')
    
    axes[0, 1].imshow(tilted_img, cmap='gray' if len(tilted_img.shape) == 2 else None)
    axes[0, 1].set_title('Tilted Image')
    axes[0, 1].plot(tilted_points[:, 0], tilted_points[:, 1], 'bo', markersize=8)
    for i, (x, y) in enumerate(tilted_points):
        axes[0, 1].annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', color='blue')
    
    # Apply predicted homography
    pred_corrected = apply_homography_to_image(tilted_img, pred_homography)
    axes[0, 2].imshow(pred_corrected, cmap='gray' if len(pred_corrected.shape) == 2 else None)
    axes[0, 2].set_title('Predicted Correction')
    
    # Apply true homography
    true_corrected = apply_homography_to_image(tilted_img, true_homography)
    axes[1, 0].imshow(true_corrected, cmap='gray' if len(true_corrected.shape) == 2 else None)
    axes[1, 0].set_title('Ground Truth Correction')
    
    # Overlay comparison
    overlay = cv2.addWeighted(ref_img.astype(np.uint8), 0.5, pred_corrected.astype(np.uint8), 0.5, 0)
    axes[1, 1].imshow(overlay, cmap='gray' if len(overlay.shape) == 2 else None)
    axes[1, 1].set_title('Reference + Predicted (Overlay)')
    
    # Difference image
    if len(ref_img.shape) == 2:  # Grayscale
        diff = np.abs(ref_img.astype(np.float32) - pred_corrected.astype(np.float32))
    else:  # Color
        diff = np.abs(ref_img.astype(np.float32) - pred_corrected.astype(np.float32))
        diff = np.mean(diff, axis=2)
    
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title('Absolute Difference')
    axes[1, 2].colorbar = plt.colorbar(axes[1, 2].imshow(diff, cmap='hot'), ax=axes[1, 2])
    
    # Remove axis ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
    
    plt.show()


def test_on_dataset(model, dataset, device, num_samples=10, output_dir=None):
    """Test model on dataset samples and compute metrics"""
    model.eval()
    
    results = {
        'sample_id': [],
        'param_mse': [],
        'corner_error': [],
        'ref_image_path': [],
        'tilted_image_path': []
    }
    
    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Testing samples"):
            try:
                sample = dataset[i]
                
                # Get data
                patch_pair = sample['patch_pair'].unsqueeze(0).to(device)
                true_params = sample['homography_params'].numpy()
                corners = sample['corners'].numpy()
                
                # Predict
                pred_params = model(patch_pair).cpu().numpy().squeeze()
                
                # Convert to matrices
                pred_matrix = params_to_matrix(pred_params)
                true_matrix = params_to_matrix(true_params)
                
                # Compute metrics
                param_mse = np.mean((pred_params - true_params) ** 2)
                
                # Transform corners using predicted homography
                corners_homogeneous = np.column_stack([corners, np.ones(len(corners))])
                pred_corners = (pred_matrix @ corners_homogeneous.T).T
                pred_corners = pred_corners[:, :2] / pred_corners[:, 2:3]
                
                # True transformed corners (should be close to reference patch corners)
                true_corners_homogeneous = np.column_stack([corners, np.ones(len(corners))])
                true_corners = (true_matrix @ true_corners_homogeneous.T).T
                true_corners = true_corners[:, :2] / true_corners[:, 2:3]
                
                corner_error = compute_corner_error(pred_corners, true_corners)
                
                # Store results
                results['sample_id'].append(i)
                results['param_mse'].append(param_mse)
                results['corner_error'].append(corner_error)
                results['ref_image_path'].append(dataset.data.iloc[i]['reference_image_path'])
                results['tilted_image_path'].append(dataset.data.iloc[i]['tilted_image_path'])
                
                print(f"Sample {i}: Param MSE = {param_mse:.6f}, Corner Error = {corner_error:.2f} pixels")
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    # Compute overall metrics
    if results['param_mse']:
        avg_param_mse = np.mean(results['param_mse'])
        avg_corner_error = np.mean(results['corner_error'])
        
        print(f"\n--- Overall Results ---")
        print(f"Average Parameter MSE: {avg_param_mse:.6f}")
        print(f"Average Corner Error: {avg_corner_error:.2f} pixels")
        print(f"Samples processed: {len(results['param_mse'])}")
        
        # Save results
        if output_dir:
            results_df = pd.DataFrame(results)
            results_path = os.path.join(output_dir, 'test_results.csv')
            results_df.to_csv(results_path, index=False)
            print(f"Results saved: {results_path}")
    
    return results


def test_on_full_images(model, csv_file, device, output_dir=None, num_samples=5):
    """Test model on full images (not just patches)"""
    data = pd.read_csv(csv_file)
    model.eval()
    
    for i in range(min(num_samples, len(data))):
        row = data.iloc[i]
        
        try:
            # Load images
            ref_img_path = row['reference_image_path']
            tilted_img_path = row['tilted_image_path']
            
            if not os.path.exists(ref_img_path) or not os.path.exists(tilted_img_path):
                print(f"Skipping sample {i}: Images not found")
                continue
            
            ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            tilted_img = cv2.imread(tilted_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            
            # Get corresponding points
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
            true_homography = cv2.getPerspectiveTransform(tilted_points, ref_points)
            
            # Extract patch pair around center of the image for prediction
            h, w = ref_img.shape
            center_x, center_y = w // 2, h // 2
            patch_size = 128
            half_patch = patch_size // 2
            
            # Extract patches
            ref_patch = ref_img[center_y-half_patch:center_y+half_patch, 
                               center_x-half_patch:center_x+half_patch]
            tilted_patch = tilted_img[center_y-half_patch:center_y+half_patch,
                                     center_x-half_patch:center_x+half_patch]
            
            if ref_patch.shape != (128, 128) or tilted_patch.shape != (128, 128):
                print(f"Skipping sample {i}: Invalid patch size")
                continue
            
            # Prepare input
            patch_pair = np.stack([ref_patch, tilted_patch], axis=0)
            patch_tensor = torch.FloatTensor(patch_pair).unsqueeze(0).to(device)
            
            # Predict homography
            with torch.no_grad():
                pred_params = model(patch_tensor).cpu().numpy().squeeze()
                pred_homography = params_to_matrix(pred_params)
            
            # Visualize results
            print(f"\nProcessing sample {i}:")
            print(f"Reference: {os.path.basename(ref_img_path)}")
            print(f"Tilted: {os.path.basename(tilted_img_path)}")
            
            save_path = None
            if output_dir:
                save_path = os.path.join(output_dir, f'visualization_sample_{i}.png')
            
            visualize_homography_result(
                ref_img, tilted_img, pred_homography, true_homography,
                ref_points, tilted_points, save_path
            )
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description='Test and visualize HomographyNet results')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset_csv', type=str, required=True,
                        help='Path to CSV file with test dataset')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Directory to save test results')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to test')
    parser.add_argument('--test_full_images', action='store_true',
                        help='Test on full images instead of patches')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print('Loading model...')
    model = load_model(args.checkpoint, device)
    
    if args.test_full_images:
        # Test on full images
        print('Testing on full images...')
        test_on_full_images(model, args.dataset_csv, device, args.output_dir, args.num_samples)
    else:
        # Test on dataset patches
        print('Loading dataset...')
        dataset = HomographyDataset(
            csv_file=args.dataset_csv,
            patch_size=128,
            max_perturbation=32,
            grayscale=True,
            augment=False
        )
        
        print('Testing on dataset patches...')
        results = test_on_dataset(model, dataset, device, args.num_samples, args.output_dir)
    
    print(f'\nTesting completed! Results saved in: {args.output_dir}')


if __name__ == '__main__':
    main()