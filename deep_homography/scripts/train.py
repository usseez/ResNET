"""
Training script for Deep Image Homography Estimation
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.homography_net import HomographyNet
from utils.data_preprocessing import HomographyDataset


class HomographyLoss(nn.Module):
    """
    Loss function for homography estimation
    Combines L2 loss on parameters with corner loss
    """
    
    def __init__(self, corner_loss_weight=1.0):
        super(HomographyLoss, self).__init__()
        self.corner_loss_weight = corner_loss_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred_params, target_params, pred_corners=None, target_corners=None):
        """
        Compute loss between predicted and target homography parameters
        
        Args:
            pred_params: Predicted homography parameters (batch_size, 8)
            target_params: Target homography parameters (batch_size, 8)
            pred_corners: Predicted corner positions (optional)
            target_corners: Target corner positions (optional)
        """
        # L2 loss on homography parameters
        param_loss = self.mse_loss(pred_params, target_params)
        
        total_loss = param_loss
        
        # Add corner loss if corners are provided
        if pred_corners is not None and target_corners is not None:
            corner_loss = self.mse_loss(pred_corners, target_corners)
            total_loss += self.corner_loss_weight * corner_loss
        
        return total_loss, param_loss


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_param_loss = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        patch_pairs = batch['patch_pair'].to(device)
        target_params = batch['homography_params'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_params = model(patch_pairs)
        
        # Compute loss
        loss, param_loss = criterion(pred_params, target_params)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_param_loss += param_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'Param Loss': f'{param_loss.item():.6f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_param_loss = total_param_loss / num_batches
    
    return avg_loss, avg_param_loss


def validate_epoch(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_param_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            patch_pairs = batch['patch_pair'].to(device)
            target_params = batch['homography_params'].to(device)
            
            # Forward pass
            pred_params = model(patch_pairs)
            
            # Compute loss
            loss, param_loss = criterion(pred_params, target_params)
            
            total_loss += loss.item()
            total_param_loss += param_loss.item()
    
    avg_loss = total_loss / num_batches
    avg_param_loss = total_param_loss / num_batches
    
    return avg_loss, avg_param_loss


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f'New best model saved with loss: {loss:.6f}')


def main():
    parser = argparse.ArgumentParser(description='Train HomographyNet')
    parser.add_argument('--dataset_csv', type=str, required=True,
                        help='Path to CSV file with dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for regularization')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='Size of image patches')
    parser.add_argument('--max_perturbation', type=int, default=32,
                        help='Maximum perturbation for patch extraction')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data to use for training')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(args.output_dir, f'homography_training_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save training arguments
    with open(os.path.join(experiment_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create dataset
    print('Loading dataset...')
    dataset = HomographyDataset(
        csv_file=args.dataset_csv,
        patch_size=args.patch_size,
        max_perturbation=args.max_perturbation,
        grayscale=True,
        augment=True
    )
    
    # Split dataset
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print('Creating model...')
    model = HomographyNet(input_channels=2)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Create optimizer and loss function
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    criterion = HomographyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(experiment_dir, 'tensorboard'))
    
    # Resume training if checkpoint provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f'Resuming training from {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
    
    # Training loop
    print('Starting training...')
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch}/{args.epochs-1}')
        print('-' * 50)
        
        # Train
        train_loss, train_param_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validate
        val_loss, val_param_loss = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('ParamLoss/Train', train_param_loss, epoch)
        writer.add_scalar('ParamLoss/Validation', val_param_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        print(f'Train Param Loss: {train_param_loss:.6f}, Val Param Loss: {val_param_loss:.6f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            experiment_dir, is_best=is_best
        )
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(experiment_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, checkpoint_path)
    
    print(f'\nTraining completed! Best validation loss: {best_val_loss:.6f}')
    print(f'Results saved in: {experiment_dir}')
    
    writer.close()


if __name__ == '__main__':
    main()