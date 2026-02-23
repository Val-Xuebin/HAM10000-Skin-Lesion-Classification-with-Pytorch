"""
Custom CNN model for skin lesion classification
Optimized architecture with carefully designed layers, BatchNorm, and Dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    """
    Optimized Custom CNN model for HAM10000 skin lesion classification
    
    Architecture Design:
    - 5 convolutional blocks with progressive filter increase (32→64→128→256→512)
    - Each block: Conv -> BatchNorm -> LeakyReLU -> Conv -> BatchNorm -> LeakyReLU -> MaxPool
    - Dropout (0.3) after each pooling layer to prevent overfitting
    - 3 fully connected layers (2048 → 1024 → 512) with BatchNorm and Dropout
    - Output layer for 7-class classification
    
    Key Features:
    - BatchNorm after every convolution for stable training
    - LeakyReLU activation (negative slope=0.01) for better gradient flow
    - Dropout (0.3) for regularization
    - Progressive filter increase for hierarchical feature learning
    """
    
    def __init__(self, num_classes=7, input_size=224, dropout_rate=0.3):
        super(CustomCNN, self).__init__()
        
        # Convolutional Block 1: 32 filters
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Convolutional Block 2: 64 filters
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Convolutional Block 3: 128 filters
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # Convolutional Block 4: 256 filters
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        # Convolutional Block 5: 512 filters
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.dropout5 = nn.Dropout2d(dropout_rate)
        
        # Calculate feature map size after 5 pooling layers
        # Input: 224x224 -> after 5 pooling: 224/32 = 7x7
        feature_map_size = (input_size // 32) ** 2
        
        # Fully connected layers with BatchNorm and Dropout
        self.fc1 = nn.Linear(512 * feature_map_size, 2048)
        self.bn_fc1 = nn.BatchNorm1d(2048)
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(2048, 1024)
        self.bn_fc2 = nn.BatchNorm1d(1024)
        self.dropout_fc2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(1024, 512)
        self.bn_fc3 = nn.BatchNorm1d(512)
        self.dropout_fc3 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc4 = nn.Linear(512, num_classes)
        
        # LeakyReLU activation (negative slope=0.01)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
    def forward(self, x):
        # Block 1: 32 filters
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2: 64 filters
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3: 128 filters
        x = self.leaky_relu(self.bn5(self.conv5(x)))
        x = self.leaky_relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Block 4: 256 filters
        x = self.leaky_relu(self.bn7(self.conv7(x)))
        x = self.leaky_relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Block 5: 512 filters
        x = self.leaky_relu(self.bn9(self.conv9(x)))
        x = self.leaky_relu(self.bn10(self.conv10(x)))
        x = self.pool5(x)
        x = self.dropout5(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with BatchNorm and Dropout
        x = self.leaky_relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        
        x = self.leaky_relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        
        x = self.leaky_relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout_fc3(x)
        
        # Output layer (no activation, will be handled by CrossEntropyLoss)
        x = self.fc4(x)
        
        return x


def create_custom_cnn(num_classes=7, input_size=224, feature_extract=False, use_pretrained=False, dropout_rate=0.3):
    """
    Create and return a CustomCNN model
    
    Args:
        num_classes: Number of output classes
        input_size: Input image size (default 224)
        feature_extract: Whether to freeze parameters (for compatibility)
        use_pretrained: Whether to use pretrained weights (for compatibility, not used)
        dropout_rate: Dropout probability (default 0.3)
    
    Returns:
        model: CustomCNN model instance
    """
    model = CustomCNN(num_classes=num_classes, input_size=input_size, dropout_rate=dropout_rate)
    
    # For custom CNN, feature_extract means we can freeze early layers if needed
    if feature_extract:
        # Freeze convolutional layers, only train FC layers
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    return model
