import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from aliyah import trainingmonitor, monitor

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

def train():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # MNIST Dataset setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', 
                                 train=True,
                                 transform=transform,
                                 download=True)
    
    test_dataset = datasets.MNIST(root='./data',
                                train=False,
                                transform=transform)
    
    # Data loader
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=100,
                            shuffle=True)
    
    test_loader = DataLoader(dataset=test_dataset,
                           batch_size=100,
                           shuffle=False)
    
    # Model, loss and optimizer
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    total_steps = len(train_loader)
    
    with trainingmonitor():
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (images, labels) in enumerate(train_loader):
                # Check for pause/stop commands
                if not monitor.check_control():
                    print("Training stopped by user")
                    return
                
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update running metrics
                running_loss += loss.item()
                
                if (i + 1) % 100 == 0:
                    batch_loss = running_loss / 100
                    batch_acc = 100 * correct / total
                    
                    # Log batch metrics
                    monitor.log_batch(i + 1, batch_loss, batch_acc)

                    
                    # Print for TUI parsing
                    print(f'Training epoch {epoch}, Loss: {batch_loss:.4f}, Accuracy: {batch_acc:.1f}%')
                    
                    running_loss = 0.0
                    correct = 0
                    total = 0
            
            # Evaluate the model
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                test_loss = 0
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                accuracy = 100 * correct / total
                avg_loss = test_loss / len(test_loader)
                
                # Log epoch metrics
                monitor.log_epoch(epoch, avg_loss, accuracy)
                
                print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.1f}%')

if __name__ == "__main__":
    train()
