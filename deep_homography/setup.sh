#!/bin/bash

# Setup script for Deep Image Homography Estimation
echo "=== Setting up Deep Image Homography Estimation ==="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed"
    exit 1
fi

echo "Python3 found: $(python3 --version)"

# Try to install python3-venv if on Ubuntu/Debian
if command -v apt &> /dev/null; then
    echo "Installing python3-venv..."
    sudo apt update
    sudo apt install -y python3-venv python3-pip
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install torch torchvision numpy pandas matplotlib opencv-python scikit-image pillow tqdm tensorboard

echo "=== Setup completed! ==="
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  python example_usage.py"
echo ""
echo "To train a model, run:"
echo "  python scripts/train.py --dataset_csv data/your_dataset.csv --output_dir results/training"
echo ""
echo "See README.md for detailed usage instructions."