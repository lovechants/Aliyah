import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import defaultdict
import time
from aliyah import trainingmonitor

class DetailedNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Deliberately verbose architecture for testing
        self.features = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
        
        # Register hooks to capture activations and gradients
        self.activations = {}
        self.gradients = {}
        self._register_hooks()
    
    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
            
        def get_gradient(name):
            def hook(grad):
                self.gradients[name] = grad.detach()
            return hook
        
        # Register hooks for each layer
        for name, layer in self.named_modules():
            if isinstance(layer, (nn.Linear, nn.BatchNorm1d)):
                layer.register_forward_hook(get_activation(name))
                if hasattr(layer, 'weight') and layer.weight is not None:
                    layer.weight.register_hook(get_gradient(f"{name}_grad"))

    def forward(self, x):
        x = x.view(-1, 784)
        features = self.features(x)
        output = self.classifier(features)
        return output

def generate_dummy_data():
    # Generate dummy MNIST-like data
    X = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    return X, y

def log_detailed_metrics(monitor, model, optimizer, loss_val, acc, batch_idx=None, epoch=None):
    metrics = {
        "loss": float(loss_val),
        "accuracy": float(acc),
        "learning_rate": optimizer.param_groups[0]["lr"],
    }
    
    # Log layer-specific metrics
    layer_metrics = {}
    
    # Activation statistics
    for name, activation in model.activations.items():
        if activation.numel() > 0:  # Check if activation is not empty
            layer_metrics[f"{name}_mean_activation"] = float(activation.mean())
            layer_metrics[f"{name}_max_activation"] = float(activation.max())
            layer_metrics[f"{name}_min_activation"] = float(activation.min())
    
    # Gradient statistics
    for name, gradient in model.gradients.items():
        if gradient.numel() > 0:  # Check if gradient is not empty
            layer_metrics[f"{name}_mean_grad"] = float(gradient.mean())
            layer_metrics[f"{name}_max_grad"] = float(gradient.max())
            layer_metrics[f"{name}_min_grad"] = float(gradient.min())
            layer_metrics[f"{name}_grad_norm"] = float(gradient.norm())
    
    # Parameter statistics
    for name, param in model.named_parameters():
        if param.numel() > 0:  # Check if parameter is not empty
            layer_metrics[f"{name}_weight_mean"] = float(param.data.mean())
            layer_metrics[f"{name}_weight_std"] = float(param.data.std())
            layer_metrics[f"{name}_weight_norm"] = float(param.data.norm())
    
    # Combine all metrics
    metrics.update(layer_metrics)
    
    # Log based on context (batch or epoch)
    if batch_idx is not None:
        monitor.log_batch(batch_idx, metrics)
    if epoch is not None:
        monitor.log_epoch(epoch, metrics)
    
    # Also log individual metrics for more granular tracking
    for name, value in metrics.items():
        monitor.log_metric(name, value)

def train_model():
    with trainingmonitor() as monitor:
        print("Starting detailed training test...")
        
        # Initialize model and data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DetailedNet().to(device)
        X, y = generate_dummy_data()
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
        
        # Log initial model architecture
        monitor.send_update("model_architecture", {
            "framework": "pytorch",
            "layers": [
                {
                    "name": name,
                    "type": type(module).__name__,
                    "parameters": sum(p.numel() for p in module.parameters()),
                    "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad)
                }
                for name, module in model.named_modules()
                if not isinstance(module, (DetailedNet, nn.Sequential))
            ]
        })
        
        for epoch in range(5):
            if not monitor.check_control():
                print("Training stopped by user")
                break
            
            model.train()
            epoch_loss = 0
            epoch_acc = 0
            total_batches = len(dataloader)
            
            for batch_idx, (data, target) in enumerate(dataloader):
                if not monitor.check_control():
                    break
                
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                pred = output.argmax(dim=1)
                acc = (pred == target).float().mean() * 100
                
                # Log detailed metrics
                log_detailed_metrics(
                    monitor, model, optimizer, loss.item(), acc.item(),
                    batch_idx=batch_idx,
                    epoch=None  # Don't log epoch metrics here
                )
                
                # Update epoch metrics
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                
                # Simulate some processing time
                time.sleep(0.1)
            
            # Calculate and log epoch metrics
            epoch_loss /= total_batches
            epoch_acc /= total_batches
            
            log_detailed_metrics(
                monitor, model, optimizer, epoch_loss, epoch_acc,
                batch_idx=None,  # Don't log batch metrics here
                epoch=epoch
            )
            
            # Update learning rate
            scheduler.step(epoch_loss)
            
            print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.1f}%")

if __name__ == "__main__":
    train_model()
