# Deep Image Homography Estimation for Car Side Camera Calibration

This project implements the "Deep Image Homography Estimation" method by Daniel DeTone et al. (2016) specifically adapted for car side camera calibration tasks. The implementation allows you to train a neural network to estimate homography transformations between reference and tilted car images using known corresponding points on the car body.

## Overview

The system consists of:
- **HomographyNet**: A 10-layer CNN that estimates 8-parameter homography transformations
- **Data preprocessing pipeline**: Handles image pairs and coordinate extraction
- **Training framework**: Complete training pipeline with loss functions and optimization
- **Testing and visualization tools**: Evaluate model performance and visualize results

## Project Structure

```
deep_homography/
├── data/
│   └── dataset_template.csv          # Template CSV file format
├── models/
│   ├── __init__.py
│   └── homography_net.py            # HomographyNet architecture
├── utils/
│   ├── __init__.py
│   └── data_preprocessing.py        # Dataset and preprocessing utilities
├── scripts/
│   ├── create_dataset_csv.py        # Helper to create dataset CSV
│   ├── train.py                     # Training script
│   └── test_and_visualize.py        # Testing and visualization
├── results/                         # Training results and checkpoints
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Prepare Your Dataset

#### Option A: Manual CSV Creation
Create a CSV file with the following columns:
- `reference_image_path`: Path to reference image
- `tilted_image_path`: Path to tilted image  
- `ref_point1_x`, `ref_point1_y`: First reference point coordinates
- `ref_point2_x`, `ref_point2_y`: Second reference point coordinates
- `ref_point3_x`, `ref_point3_y`: Third reference point coordinates
- `ref_point4_x`, `ref_point4_y`: Fourth reference point coordinates
- `tilted_point1_x`, `tilted_point1_y`: First tilted point coordinates
- `tilted_point2_x`, `tilted_point2_y`: Second tilted point coordinates
- `tilted_point3_x`, `tilted_point3_y`: Third tilted point coordinates
- `tilted_point4_x`, `tilted_point4_y`: Fourth tilted point coordinates

See `data/dataset_template.csv` for the exact format.

#### Option B: Automatic CSV Generation
If you have organized your images in separate directories:

```bash
python scripts/create_dataset_csv.py \
    --ref_dir /path/to/reference/images \
    --tilted_dir /path/to/tilted/images \
    --output_csv data/my_dataset.csv \
    --use_default_points \
    --perturbation_range 20
```

**Note**: The automatic generation creates default corner points. For car calibration, you should manually edit the CSV to specify the exact coordinates of fixed points on the car body (e.g., door handles, wheel centers, body edges).

### Step 2: Train the Model

```bash
python scripts/train.py \
    --dataset_csv data/my_dataset.csv \
    --batch_size 64 \
    --epochs 100 \
    --lr 0.005 \
    --output_dir results/experiment_1
```

Key training parameters:
- `--dataset_csv`: Path to your dataset CSV file
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.005)
- `--momentum`: SGD momentum (default: 0.9)
- `--train_split`: Fraction for training vs validation (default: 0.8)
- `--patch_size`: Size of image patches (default: 128)
- `--max_perturbation`: Maximum random perturbation for augmentation (default: 32)

The training script will:
- Split your dataset into training and validation sets
- Save checkpoints every 10 epochs
- Save the best model based on validation loss
- Log training progress to TensorBoard
- Save all results in the specified output directory

### Step 3: Test and Visualize Results

#### Test on Dataset Patches
```bash
python scripts/test_and_visualize.py \
    --checkpoint results/experiment_1/best_model.pth \
    --dataset_csv data/my_dataset.csv \
    --output_dir results/test_results \
    --num_samples 10
```

#### Test on Full Images
```bash
python scripts/test_and_visualize.py \
    --checkpoint results/experiment_1/best_model.pth \
    --dataset_csv data/my_dataset.csv \
    --output_dir results/test_results \
    --num_samples 5 \
    --test_full_images
```

The testing script will:
- Load the trained model
- Apply homography corrections to test images
- Generate visualizations showing:
  - Original reference and tilted images
  - Predicted and ground truth corrections
  - Overlay comparisons
  - Difference maps
- Compute quantitative metrics (MSE, corner errors)
- Save results and visualizations

## Key Features for Car Calibration

### 1. Known Corresponding Points
The system is designed to work with known fixed points on car bodies:
- Door handles
- Wheel centers  
- Body panel edges
- License plate corners
- Any other consistent car features

### 2. Robust Data Augmentation
- Random patch extraction with perturbation
- Grayscale conversion for lighting invariance
- Batch normalization for stable training

### 3. Comprehensive Evaluation
- Parameter MSE loss
- Corner reprojection error
- Visual overlay comparisons
- Difference map analysis

## Model Architecture

HomographyNet follows the DeTone et al. (2016) architecture:

```
Input: 2-channel 128x128 patches (reference + tilted)
├── Conv2d(2→64) + BatchNorm + ReLU
├── Conv2d(64→64) + BatchNorm + ReLU  
├── Conv2d(64→64, stride=2) + BatchNorm + ReLU    # 64x64
├── Conv2d(64→64) + BatchNorm + ReLU
├── Conv2d(64→128, stride=2) + BatchNorm + ReLU   # 32x32
├── Conv2d(128→128) + BatchNorm + ReLU
├── Conv2d(128→128, stride=2) + BatchNorm + ReLU  # 16x16
├── Conv2d(128→128) + BatchNorm + ReLU
├── Flatten + Dropout(0.5)
├── Linear(128*16*16→1024) + ReLU + Dropout
└── Linear(1024→8)  # 8 homography parameters
```

## Tips for Car Calibration

### 1. Point Selection
- Choose 4+ fixed points that are clearly visible in both reference and tilted images
- Distribute points across the car body for better stability
- Avoid points that might be occluded or move between shots

### 2. Data Collection
- Ensure consistent lighting conditions
- Maintain similar distance and angle for reference shots
- Capture various tilt angles for robust training

### 3. Training Optimization
- Start with a smaller learning rate (0.001) if training is unstable
- Use more epochs (200+) for better convergence
- Monitor validation loss to avoid overfitting

### 4. Evaluation
- Test on diverse tilt angles not seen during training
- Check corner reprojection errors (should be <5 pixels for good calibration)
- Visually inspect overlay results for alignment quality

## Monitoring Training

The training script includes TensorBoard logging:

```bash
tensorboard --logdir results/experiment_1/tensorboard
```

Monitor these metrics:
- Training and validation loss curves
- Parameter MSE trends
- Learning rate schedule

## Troubleshooting

### Common Issues

1. **"Images not found" errors**
   - Check that all image paths in your CSV are correct and accessible
   - Use absolute paths if relative paths cause issues

2. **CUDA out of memory**
   - Reduce batch size (try 32 or 16)
   - Reduce patch size to 96 or 64

3. **Poor convergence**
   - Lower learning rate (0.001 or 0.0005)
   - Check that your corresponding points are accurate
   - Ensure sufficient training data (100+ image pairs recommended)

4. **High corner errors**
   - Verify the accuracy of your manual point annotations
   - Check for consistent point ordering (clockwise/counterclockwise)
   - Ensure points are on truly fixed parts of the car

### Performance Expectations

- **Training time**: ~2-4 hours for 100 epochs on GPU
- **Corner error**: <5 pixels for well-calibrated models
- **Parameter MSE**: <0.01 for good convergence

## References

1. DeTone, D., Malisiewicz, T., & Rabinovich, A. (2016). Deep image homography estimation. arXiv preprint arXiv:1606.03798.

## License

This implementation is provided for research and educational purposes. Please cite the original paper if you use this work in your research.

## Contributing

Feel free to submit issues and enhancement requests. Contributions are welcome!