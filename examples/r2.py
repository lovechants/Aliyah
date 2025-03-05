import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from aliyah import trainingmonitor

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Small network for clearer visualization
        self.layers = nn.ModuleList([
            nn.Linear(2, 4),
            nn.Linear(4, 3),
            nn.Linear(3, 1)
        ])
        
        # Register hooks for all layers
        self.activation_values = {}
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self.get_activation_hook(f"layer_{i}"))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))  # Using tanh for bounded activations
        x = self.layers[-1](x)
        return x
    
    def get_activation_hook(self, name):
        def hook(module, input, output):
            self.activation_values[name] = torch.tanh(output)  # Normalize activations
        return hook

def generate_spiral_data(samples=100):
    """Generate two interleaving spirals for binary classification"""
    theta = np.sqrt(np.random.rand(samples)) * 2 * np.pi
    
    # First spiral
    r1 = theta + np.pi
    x1 = np.c_[r1 * np.cos(theta), r1 * np.sin(theta)]
    x1 += np.random.randn(samples, 2) * 0.1
    
    # Second spiral
    r2 = -theta - np.pi
    x2 = np.c_[r2 * np.cos(theta), r2 * np.sin(theta)]
    x2 += np.random.randn(samples, 2) * 0.1
    
    # Combine data
    X = np.vstack([x1, x2])
    y = np.hstack([np.ones(samples), np.zeros(samples)])
    
    return torch.FloatTensor(X), torch.FloatTensor(y).reshape(-1, 1)

def visualize_network(model, monitor, x_batch):
    """Send detailed visualization updates"""
    # Forward pass to get activations
    with torch.no_grad():
        _ = model(x_batch)
    
    # Visualize each layer's activations
    for layer_idx, (name, activations) in enumerate(model.activation_values.items()):
        # Get activation values and normalize
        act_values = activations.abs().mean(dim=0).numpy()
        act_norm = (act_values - act_values.min()) / (act_values.max() - act_values.min() + 1e-8)
        
        # Send individual node activations
        for node_idx, act_val in enumerate(act_norm):
            monitor.log_layer_activation(layer_idx, node_idx, float(act_val))
        
        # Send layer state
        monitor.log_layer_state(layer_idx, act_norm.tolist())
        
        # Visualize connections if not first layer
        if layer_idx > 0:
            prev_acts = model.activation_values[f"layer_{layer_idx-1}"].abs().mean(dim=0).numpy()
            prev_norm = (prev_acts - prev_acts.min()) / (prev_acts.max() - prev_acts.min() + 1e-8)
            
            # Show connections for strong activations
            for i, prev_act in enumerate(prev_norm):
                for j, curr_act in enumerate(act_norm):
                    connection_strength = prev_act * curr_act
                    if connection_strength > 0.3:  # Only show strong connections
                        monitor.log_connection_flow(layer_idx-1, i, layer_idx, j, True)

def train():
    # Generate dataset
    X, y = generate_spiral_data(samples=100)
    
    # Create model and optimizer
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training parameters
    n_epochs = 50
    batch_size = 20
    
    with trainingmonitor() as monitor:
        for epoch in range(n_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            # Process in batches
            for i in range(0, len(X), batch_size):
                if not monitor.check_control():
                    return
                
                # Get batch
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
                total_loss += loss.item()
                
                # Visualize network state
                visualize_network(model, monitor, batch_X[0:1])
                
                # Log metrics
                accuracy = 100 * correct / total
                monitor.log_batch(i//batch_size, loss.item(), accuracy)
                
                # Small delay for better visualization
                time.sleep(0.05)
            
            # Log epoch metrics
            epoch_accuracy = 100 * correct / total
            epoch_loss = total_loss / (len(X) / batch_size)
            monitor.log_epoch(epoch, epoch_loss, epoch_accuracy)
            
            if not monitor.check_control():
                break

if __name__ == "__main__":
    train()
