"""
Helper script to create dataset CSV from image directories
"""

import os
import argparse
import pandas as pd
import cv2
import numpy as np
from pathlib import Path


def find_image_pairs(ref_dir, tilted_dir, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """
    Find matching image pairs between reference and tilted directories
    
    Args:
        ref_dir: Directory containing reference images
        tilted_dir: Directory containing tilted images
        extensions: Valid image file extensions
        
    Returns:
        List of (ref_path, tilted_path) tuples
    """
    ref_files = {}
    tilted_files = {}
    
    # Scan reference directory
    for file_path in Path(ref_dir).rglob('*'):
        if file_path.suffix.lower() in extensions:
            base_name = file_path.stem
            ref_files[base_name] = str(file_path)
    
    # Scan tilted directory
    for file_path in Path(tilted_dir).rglob('*'):
        if file_path.suffix.lower() in extensions:
            base_name = file_path.stem
            tilted_files[base_name] = str(file_path)
    
    # Find matching pairs
    pairs = []
    for base_name in ref_files:
        if base_name in tilted_files:
            pairs.append((ref_files[base_name], tilted_files[base_name]))
    
    return pairs


def get_default_points(image_path, margin_ratio=0.2):
    """
    Generate default corner points for an image
    
    Args:
        image_path: Path to image file
        margin_ratio: Margin from edges as ratio of image dimensions
        
    Returns:
        4x2 array of corner points
    """
    img = cv2.imread(image_path)
    if img is None:
        # Default image size if can't load
        h, w = 480, 640
    else:
        h, w = img.shape[:2]
    
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    
    points = np.array([
        [margin_x, margin_y],                    # Top-left
        [w - margin_x, margin_y],                # Top-right
        [w - margin_x, h - margin_y],            # Bottom-right
        [margin_x, h - margin_y]                 # Bottom-left
    ])
    
    return points


def add_random_perturbation(points, max_perturbation=20):
    """Add random perturbation to points"""
    perturbation = np.random.randint(-max_perturbation, max_perturbation + 1, points.shape)
    return points + perturbation


def create_dataset_csv(ref_dir, tilted_dir, output_csv, use_default_points=True, 
                      perturbation_range=20):
    """
    Create dataset CSV from image directories
    
    Args:
        ref_dir: Directory with reference images
        tilted_dir: Directory with tilted images  
        output_csv: Output CSV file path
        use_default_points: Generate default corner points
        perturbation_range: Random perturbation range for tilted points
    """
    # Find image pairs
    pairs = find_image_pairs(ref_dir, tilted_dir)
    print(f"Found {len(pairs)} image pairs")
    
    if not pairs:
        print("No matching image pairs found!")
        return
    
    data = []
    
    for ref_path, tilted_path in pairs:
        print(f"Processing: {os.path.basename(ref_path)}")
        
        if use_default_points:
            # Generate default reference points
            ref_points = get_default_points(ref_path)
            
            # Add perturbation for tilted points
            tilted_points = add_random_perturbation(ref_points, perturbation_range)
        else:
            # Placeholder points - user needs to fill these manually
            ref_points = np.array([[100, 100], [300, 100], [300, 200], [100, 200]])
            tilted_points = np.array([[95, 105], [305, 95], [295, 205], [105, 195]])
        
        # Create row data
        row = [ref_path, tilted_path] + \
              [coord for point in ref_points for coord in point] + \
              [coord for point in tilted_points for coord in point]
        
        data.append(row)
    
    # Create DataFrame
    columns = ['reference_image_path', 'tilted_image_path',
               'ref_point1_x', 'ref_point1_y', 'ref_point2_x', 'ref_point2_y',
               'ref_point3_x', 'ref_point3_y', 'ref_point4_x', 'ref_point4_y',
               'tilted_point1_x', 'tilted_point1_y', 'tilted_point2_x', 'tilted_point2_y',
               'tilted_point3_x', 'tilted_point3_y', 'tilted_point4_x', 'tilted_point4_y']
    
    df = pd.DataFrame(data, columns=columns)
    
    # Save CSV
    df.to_csv(output_csv, index=False)
    print(f"Dataset CSV created: {output_csv}")
    print(f"Total samples: {len(df)}")
    
    if not use_default_points:
        print("\nWARNING: Default placeholder points were used.")
        print("Please manually edit the CSV file to add correct corresponding points.")


def main():
    parser = argparse.ArgumentParser(description='Create dataset CSV from image directories')
    parser.add_argument('--ref_dir', type=str, required=True,
                        help='Directory containing reference images')
    parser.add_argument('--tilted_dir', type=str, required=True,
                        help='Directory containing tilted images')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--use_default_points', action='store_true',
                        help='Generate default corner points based on image dimensions')
    parser.add_argument('--perturbation_range', type=int, default=20,
                        help='Random perturbation range for tilted points')
    
    args = parser.parse_args()
    
    # Verify directories exist
    if not os.path.exists(args.ref_dir):
        print(f"Error: Reference directory not found: {args.ref_dir}")
        return
    
    if not os.path.exists(args.tilted_dir):
        print(f"Error: Tilted directory not found: {args.tilted_dir}")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create dataset CSV
    create_dataset_csv(
        args.ref_dir,
        args.tilted_dir, 
        args.output_csv,
        args.use_default_points,
        args.perturbation_range
    )


if __name__ == '__main__':
    main()