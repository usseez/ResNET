"""
HomographyNet implementation based on:
"Deep Image Homography Estimation" by Daniel DeTone et al. (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HomographyNet(nn.Module):
    """
    HomographyNet: A 10-layer CNN for estimating homography parameters.
    
    Input: Stacked grayscale image patches (2 channels, 128x128)
    Output: 8-parameter homography vector (h1, h2, h3, h4, h5, h6, h7, h8)
    where h9 = 1 (normalized homography)
    """
    
    def __init__(self, input_channels=2):
        super(HomographyNet, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # Downsample
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Downsample
        self.bn5 = nn.BatchNorm2d(128)
        
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)  # Downsample
        self.bn7 = nn.BatchNorm2d(128)
        
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions: 128x128 -> 64x64 -> 32x32 -> 16x16
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 8)  # Output 8 homography parameters
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 2, 128, 128)
            
        Returns:
            homography_params: Tensor of shape (batch_size, 8)
        """
        # Convolutional layers with ReLU activation
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # 64x64
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))  # 32x32
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))  # 16x16
        x = F.relu(self.bn8(self.conv8(x)))
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
    
    def predict_homography_matrix(self, homography_params):
        """
        Convert 8-parameter vector to 3x3 homography matrix
        
        Args:
            homography_params: Tensor of shape (batch_size, 8)
            
        Returns:
            homography_matrix: Tensor of shape (batch_size, 3, 3)
        """
        batch_size = homography_params.size(0)
        
        # Create 3x3 homography matrix
        # H = [[h1, h2, h3],
        #      [h4, h5, h6],
        #      [h7, h8, 1 ]]
        homography_matrix = torch.zeros(batch_size, 3, 3, device=homography_params.device)
        
        homography_matrix[:, 0, 0] = homography_params[:, 0]  # h1
        homography_matrix[:, 0, 1] = homography_params[:, 1]  # h2
        homography_matrix[:, 0, 2] = homography_params[:, 2]  # h3
        homography_matrix[:, 1, 0] = homography_params[:, 3]  # h4
        homography_matrix[:, 1, 1] = homography_params[:, 4]  # h5
        homography_matrix[:, 1, 2] = homography_params[:, 5]  # h6
        homography_matrix[:, 2, 0] = homography_params[:, 6]  # h7
        homography_matrix[:, 2, 1] = homography_params[:, 7]  # h8
        homography_matrix[:, 2, 2] = 1.0  # h9 = 1
        
        return homography_matrix


def test_homography_net():
    """Test the HomographyNet model"""
    model = HomographyNet()
    
    # Create dummy input (batch_size=2, channels=2, height=128, width=128)
    dummy_input = torch.randn(2, 2, 128, 128)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # Test homography matrix conversion
    homography_matrices = model.predict_homography_matrix(output)
    print(f"Homography matrices shape: {homography_matrices.shape}")
    print(f"First homography matrix:\n{homography_matrices[0]}")


if __name__ == "__main__":
    test_homography_net()