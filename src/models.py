import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    """
    Custom Convolutional Neural Network architecture for classification.
    """
    def __init__(self, num_classes=3):
        super(SmallCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # (3 → 32)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (32 → 64)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)   # downsampling

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # (64 → 128)
        self.bn3 = nn.BatchNorm2d(128)

        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(2048, 256) 
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32 → 16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16 → 8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8 → 4
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x