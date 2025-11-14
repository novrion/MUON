import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Block 1: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 2: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Block 3: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        # Pooling and dropout
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout2d = nn.Dropout2d(0.2)
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Block 1: 32x32 -> 16x16
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout2d(x)

        # Block 2: 16x16 -> 8x8
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)
        x = self.dropout2d(x)

        # Block 3: 8x8 -> 4x4
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.maxpool(x)
        x = self.dropout2d(x)

        # Fully connected layers
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
