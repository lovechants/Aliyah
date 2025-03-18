import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from aliyah import trainingmonitor, monitor

# Create synthetic dataset
def generate_synthetic_data(samples=1000, features=8):
    # Generate random weights
    true_weights = np.random.randn(features, 3)
    
    # Generate random features
    X = np.random.randn(samples, features)
    
    # Generate outputs with some noise
    logits = np.dot(X, true_weights) + np.random.randn(samples, 3) * 0.1
    
    # Apply softmax to get probabilities
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Get class labels
    y = np.argmax(probs, axis=1)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    return X_tensor, y_tensor, ["Class A", "Class B", "Class C"]

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size=8, hidden_sizes=[16, 8], output_size=3):
        super(SimpleNet, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)
        
        # For activation visualization
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Register hooks for all layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(get_activation(name))
    
    def forward(self, x):
        return self.model(x)

def send_model_architecture(model):
    """Send model architecture to Aliyah"""
    layers = []
    total_params = 0
    
    # Add input layer
    layers.append({
        "name": "input",
        "layer_type": "Input",
        "input_size": [model.input_size],
        "output_size": [model.input_size],
        "parameters": 0
    })
    
    # Add hidden layers
    for i, module in enumerate(model.model):
        if isinstance(module, nn.Linear):
            layer_info = {
                "name": f"layer_{i}",
                "layer_type": "Linear",
                "input_size": [module.in_features],
                "output_size": [module.out_features],
                "parameters": module.weight.numel() + module.bias.numel()
            }
            layers.append(layer_info)
            total_params += layer_info["parameters"]
        elif isinstance(module, nn.ReLU):
            layer_info = {
                "name": f"relu_{i}",
                "layer_type": "ReLU",
                "parameters": 0
            }
            layers.append(layer_info)
    
    # Send to monitor
    monitor.send_update("model_architecture", {
        "framework": "pytorch",
        "layers": layers,
        "total_parameters": total_params
    })

def visualize_layer_activations(model):
    """Visualize layer activations for network diagram"""
    for i, (name, activation) in enumerate(model.activations.items()):
        # Only visualize Linear layer activations
        if "model" in name and isinstance(activation, torch.Tensor):
            # Get layer index (for proper ordering in visualization)
            layer_idx = int(name.split(".")[1]) // 2  # 2 modules per layer (Linear + ReLU)
            
            # Get mean activation values
            act_mean = activation.abs().mean(dim=0).cpu().numpy()
            
            # Log activations
            monitor.log_layer_state(layer_idx, act_mean.tolist())
            
            # Show some connections between layers
            if layer_idx > 0:
                for _ in range(3):  # Add a few active connections for visualization
                    from_idx = np.random.randint(0, model.hidden_sizes[layer_idx-1])
                    to_idx = np.random.randint(0, 
                             model.hidden_sizes[layer_idx] if layer_idx < len(model.hidden_sizes) else model.output_size)
                    monitor.log_connection_flow(layer_idx-1, from_idx, layer_idx, to_idx, True)

def train():
    # Generate synthetic data
    X, y, class_names = generate_synthetic_data(samples=1000)
    
    # Split into train/test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create model, loss function, and optimizer
    model = SimpleNet(input_size=8, hidden_sizes=[16, 8], output_size=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Send model architecture to monitor
    send_model_architecture(model)
    
    # Training parameters
    num_epochs = 20
    batch_size = 32
    num_batches = len(X_train) // batch_size
    
    # Start training with Aliyah monitoring
    with trainingmonitor():
        for epoch in range(num_epochs):
            # Check if we should continue
            if not monitor.check_control():
                print("Training stopped by user")
                break
            
            # Shuffle data
            indices = torch.randperm(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Train for one epoch
            model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            
            for batch in range(num_batches):
                # Get batch
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == y_batch).float().mean().item() * 100
                
                # Accumulate metrics
                epoch_loss += loss.item()
                epoch_acc += accuracy
                
                # Log batch metrics
                if batch % 5 == 0 or batch == num_batches - 1:
                    batch_loss = epoch_loss / (batch + 1)
                    batch_acc = epoch_acc / (batch + 1)
                    
                    monitor.log_batch(batch, float(batch_loss), float(batch_acc))
                    
                    # Visualize layer activations
                    visualize_layer_activations(model)
                    
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch+1}/{num_batches}, Loss: {batch_loss:.4f}, Accuracy: {batch_acc:.2f}%")
            
            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                test_loss = criterion(outputs, y_test).item()
                _, predicted = torch.max(outputs.data, 1)
                test_acc = (predicted == y_test).float().mean().item() * 100
            
            # Log epoch metrics
            monitor.log_epoch(epoch, float(test_loss), float(test_acc))
            
            # Visualize prediction at the end of each epoch
            with torch.no_grad():
                # Get a random test sample
                sample_idx = np.random.randint(0, len(X_test))
                sample_X = X_test[sample_idx:sample_idx+1]
                sample_y = y_test[sample_idx].item()
                
                # Get prediction
                outputs = model(sample_X)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0].numpy()
                
                # Log prediction
                monitor.log_prediction(
                    probs.tolist(),
                    class_names,
                    f"Prediction (Epoch {epoch+1}, True: {int(sample_y)})"
                )
            
            print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Show some final predictions
    model.eval()
    with torch.no_grad():
        for i in range(3):
            # Get a random test sample
            sample_idx = np.random.randint(0, len(X_test))
            sample_X = X_test[sample_idx:sample_idx+1]
            sample_y = y_test[sample_idx].item()
            
            # Get prediction
            outputs = model(sample_X)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0].numpy()
            
            # Log prediction
            monitor.log_prediction(
                probs.tolist(),
                class_names,
                f"Final Example {i+1} (True: {sample_y})"
            )
            
            # Space out predictions
            time.sleep(0.5)

if __name__ == "__main__":
    train()
