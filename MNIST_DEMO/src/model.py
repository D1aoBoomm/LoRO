"""
MNIST 3-Layer MLP Model Definition
Layer 1: 784 -> 256
Layer 2: 256 -> 128
Layer 3: 128 -> 10
"""

import torch
import torch.nn as nn

class MNISTNet(nn.Module):
    """Simple 3-layer MLP for MNIST classification"""
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_layer_weights(self):
        """Return weights and biases for each layer"""
        return [
            (self.fc1.weight.data, self.fc1.bias.data),
            (self.fc2.weight.data, self.fc2.bias.data),
            (self.fc3.weight.data, self.fc3.bias.data)
        ]

    def set_layer_weights(self, weights_list):
        """Set weights and biases for each layer"""
        for i, (w, b) in enumerate(weights_list):
            if i == 0:
                self.fc1.weight.data = w
                self.fc1.bias.data = b
            elif i == 1:
                self.fc2.weight.data = w
                self.fc2.bias.data = b
            elif i == 2:
                self.fc3.weight.data = w
                self.fc3.bias.data = b
