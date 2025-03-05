import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from aliyah import trainingmonitor

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
        
        # Track layers for visualization
        self.layers = [self.fc1, self.fc2, self.fc3]
        
        # Register hooks for visualization
        self.activation_values = {}
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self.get_activation_hook(f"layer_{i}"))

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_activation_hook(self, name):
        def hook(module, input, output):
            # Get post-activation values
            if isinstance(module, nn.Linear):
                self.activation_values[name] = output
        return hook

def train_step(model, data, target, optimizer, criterion, monitor, batch_idx):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    pred = output.argmax(dim=1, keepdim=True)
    accuracy = pred.eq(target.view_as(pred)).float().mean().item() * 100

    # Log metrics (no rate limiting)
    monitor.log_batch(batch_idx, loss.item(), accuracy)

    # Visualize network state
    for idx, (name, activations) in enumerate(model.activation_values.items()):
        # Get post-activation values and normalize
        act_mean = activations.abs().mean(dim=0).detach().cpu().numpy()
        act_norm = (act_mean - act_mean.min()) / (act_mean.max() - act_mean.min() + 1e-8)
        monitor.log_layer_state(idx, act_norm.tolist())

    return loss.item(), accuracy

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load the training data
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Training loop with visualization
    with trainingmonitor() as monitor:
        
        for epoch in range(10):
            model.train()
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            batch_count = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Training step
                loss, accuracy = train_step(model, data, target, optimizer, 
                                         criterion, monitor, batch_idx)
                
                epoch_loss += loss
                epoch_accuracy += accuracy
                batch_count += 1

                # Log epoch stats periodically
                if batch_idx > 0 and batch_idx % 100 == 0:
                    avg_loss = epoch_loss / batch_count
                    avg_accuracy = epoch_accuracy / batch_count
                    monitor.log_epoch(epoch, avg_loss, avg_accuracy)

                # Check for pause/stop
                if not monitor.check_control():
                    print("Training stopped by user")
                    return

            # Final epoch stats
            avg_loss = epoch_loss / batch_count
            avg_accuracy = epoch_accuracy / batch_count
            monitor.log_epoch(epoch, avg_loss, avg_accuracy)

            if not monitor.check_control():
                break

if __name__ == "__main__":
    main()
