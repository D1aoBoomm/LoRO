"""
Train MNIST 3-Layer MLP and save weights.

This script trains a simple 3-layer MLP on MNIST dataset
and saves the weights for later use in secure inference demo.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTNet
import os

def train_model(epochs=5, batch_size=64, lr=0.001, save_path=None):
    """Train the MNIST model"""

    # Default save path relative to this file
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), '..', 'weights')

    # Create weights directory
    os.makedirs(save_path, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data directory relative to this file
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')

        print(f'Epoch {epoch+1} completed. Avg Loss: {total_loss/len(train_loader):.4f}, '
              f'Train Acc: {100.*correct/total:.2f}%')

    # Testing
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_acc = 100. * correct / total
    print(f'\nTest Accuracy: {test_acc:.2f}%')

    # Save weights
    weights_list = model.get_layer_weights()

    # Save each layer's weights
    for i, (w, b) in enumerate(weights_list):
        torch.save({
            'weight': w.cpu(),
            'bias': b.cpu(),
            'shape': w.shape
        }, os.path.join(save_path, f'layer_{i+1}.pt'))
        print(f"Saved layer {i+1}: weight shape {w.shape}, bias shape {b.shape}")

    # Save complete model
    torch.save(model.state_dict(), os.path.join(save_path, 'mnist_model.pt'))
    print(f"\nModel saved to {save_path}")

    return model, test_acc

def load_weights(save_path='weights/'):
    """Load saved weights"""
    weights_list = []
    for i in range(3):
        checkpoint = torch.load(os.path.join(save_path, f'layer_{i+1}.pt'))
        weights_list.append((checkpoint['weight'], checkpoint['bias']))
    return weights_list

if __name__ == '__main__':
    train_model(epochs=2, batch_size=64, lr=0.001)
