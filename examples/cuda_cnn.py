import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import numpy as np
from aliyah import trainingmonitor, monitor

"""
Same example as the Metal / MPS stress test with CUDA 
"""

class VisualConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a more memory-efficient network with distinct layers for visualization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.flatten = nn.Flatten()
        # Reduced size fully connected layers - using a smaller input image size
        self.fc1 = nn.Linear(128 * 7 * 7, 1024)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 10)
        self.softmax = nn.Softmax(dim=1)
        
        # Store activations
        self.activations = {}
        
        # Register hooks to capture activations for visualization
        self.hooks = []
        for name, layer in self.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU)):
                self.hooks.append(
                    layer.register_forward_hook(self.make_hook(name))
                )
        
        # List of layers for easy iteration
        self.layers = [
            self.conv1, self.relu1, self.conv2, self.relu2, self.pool1,
            self.conv3, self.relu3, self.conv4, self.relu4, self.pool2,
            self.flatten, self.fc1, self.relu5, self.fc2, self.softmax
        ]
    
    def make_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output
        return hook
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.relu5(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x

def gpu_stress_test():
    """
    Function to stress test the GPU with a model and visualize the network
    """
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS (Metal Performance Shaders) available: {torch.backends.mps.is_available()}")
    
    # Use Metal if available, otherwise use CPU
    device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model and move to device
    model = VisualConvNet().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Create a batch of colorful input images (more visually interesting)
    batch_size = 16
    input_size = 28  # Smaller input size to save memory
    inputs = []
    
    # Generate synthetic images with patterns
    for i in range(batch_size):
        # Create a base pattern (checkerboard, radial, etc.)
        pattern_type = i % 4
        
        # Create RGB channels
        r = torch.zeros(input_size, input_size)
        g = torch.zeros(input_size, input_size)
        b = torch.zeros(input_size, input_size)
        
        if pattern_type == 0:  # Checkerboard
            for x in range(input_size):
                for y in range(input_size):
                    if (x // 4 + y // 4) % 2 == 0:
                        r[x, y] = 0.8
                        g[x, y] = 0.2
                        b[x, y] = 0.2
                    else:
                        r[x, y] = 0.2
                        g[x, y] = 0.8
                        b[x, y] = 0.2
        
        elif pattern_type == 1:  # Radial
            cx, cy = input_size // 2, input_size // 2
            for x in range(input_size):
                for y in range(input_size):
                    dist = ((x - cx)**2 + (y - cy)**2)**0.5
                    r[x, y] = (math.sin(dist / 3) + 1) / 2
                    g[x, y] = (math.cos(dist / 3) + 1) / 2
                    b[x, y] = (math.sin(dist / 5) + 1) / 2
        
        elif pattern_type == 2:  # Stripes
            for x in range(input_size):
                for y in range(input_size):
                    r[x, y] = (math.sin(x / 3) + 1) / 2
                    g[x, y] = (math.sin((x + y) / 5) + 1) / 2
                    b[x, y] = (math.sin(y / 3) + 1) / 2
        
        else:  # Spiral
            cx, cy = input_size // 2, input_size // 2
            for x in range(input_size):
                for y in range(input_size):
                    dx, dy = x - cx, y - cy
                    angle = math.atan2(dy, dx) if dx != 0 or dy != 0 else 0
                    dist = ((x - cx)**2 + (y - cy)**2)**0.5
                    r[x, y] = (math.sin(dist / 3 + angle * 3) + 1) / 2
                    g[x, y] = (math.sin(dist / 3 + angle * 2) + 1) / 2
                    b[x, y] = (math.sin(dist / 3 + angle) + 1) / 2
        
        # Combine channels and add to batch
        img = torch.stack([r, g, b])
        inputs.append(img)
    
    # Stack into a batch tensor
    input_tensor = torch.stack(inputs).to(device)
    
    # Create targets (random classes)
    target = torch.randint(0, 10, (batch_size,), device=device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Helper function to create GPU load - using smaller matrices to prevent OOM
    def create_gpu_load():
        """Create additional GPU load with matrix multiplications"""
        matrices = []
        for i in range(3):
            size = 1000 + i * 200  # Smaller matrices
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            matrices.append((a, b))
        
        # Do matrix multiplications to create load
        results = []
        for a, b in matrices:
            results.append(torch.matmul(a, b))
        
        # Force evaluation
        for r in results:
            _ = r.sum().item()
    
    # Function to log network activations for visualization
    def log_network_state(model, batch_idx):
        # Log activations and connections 
        for layer_idx, (name, activations) in enumerate(model.activations.items()):
            if isinstance(activations, torch.Tensor):
                # For convolutional layers, take the mean activation across spatial dimensions
                if len(activations.shape) == 4:  # [batch, channels, height, width]
                    # Take first image in batch, mean across spatial dimensions
                    act_mean = activations[0].mean(dim=(1, 2)).detach().cpu().numpy()
                    # Normalize values to 0-1 range for visualization
                    if act_mean.max() > act_mean.min():
                        act_norm = (act_mean - act_mean.min()) / (act_mean.max() - act_mean.min())
                    else:
                        act_norm = np.zeros_like(act_mean)
                
                # For fully connected layers
                elif len(activations.shape) == 2:  # [batch, features]
                    # Take first image, first few features
                    act_mean = activations[0, :min(10, activations.shape[1])].detach().cpu().numpy()
                    # Normalize
                    if act_mean.max() > act_mean.min():
                        act_norm = (act_mean - act_mean.min()) / (act_mean.max() - act_mean.min())
                    else:
                        act_norm = np.zeros_like(act_mean)
                
                # Limit to first 10 values for visualization
                act_display = act_norm[:10].tolist() if len(act_norm) > 10 else act_norm.tolist()
                monitor.log_layer_state(layer_idx, act_display)
                
                # Create some random connections for visualization
                # In a real network, these would represent actual weights
                if layer_idx > 0 and layer_idx < len(model.activations) - 1:
                    for i in range(min(5, len(act_display))):
                        # Connect to random nodes in previous layer
                        prev_layer = layer_idx - 1
                        for _ in range(3):  # Create 3 random connections
                            prev_node = np.random.randint(0, 5)
                            # Connection is active if activation is high enough
                            active = act_display[i] > 0.5
                            monitor.log_connection_flow(
                                prev_layer, prev_node, 
                                layer_idx, i, 
                                active
                            )
    
    # Function to log predictions with class labels
    def log_predictions(output, target):
        # Get predicted class
        _, predicted = output.max(1)
        
        # For the first image in batch
        pred_class = predicted[0].item()
        true_class = target[0].item()
        
        # Get class probabilities
        probs = output[0].detach().cpu().numpy()
        
        # Create some interesting class names for visualization
        class_names = [
            "Dog", "Cat", "Car", "Bicycle", "Airplane", 
            "Tree", "Flower", "Building", "Mountain", "Beach"
        ]
        
        # Use only first 10 probabilities for visualization
        monitor.log_prediction(
            probs.tolist(), 
            class_names, 
            f"True: {class_names[true_class % 10]}, Predicted: {class_names[pred_class % 10]}"
        )
    
    # Training loop with monitoring
    with trainingmonitor() as monitor:
        print("Starting GPU stress test with visualizations...")
        
        # First, warm up the GPU
        print("Warming up GPU...")
        for i in range(3):
            output = model(input_tensor)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print("Beginning main stress test with visualizations...")
        for epoch in range(10):
            epoch_loss = 0.0
            
            # Log at start of epoch
            monitor.log_epoch(epoch, loss=0.0)
            
            # Run iterations to stress the GPU
            for batch in range(50):
                # Check for user control
                if not monitor.check_control():
                    print("Training stopped by user")
                    return
                
                # Create additional GPU load every few batches
                if batch % 5 == 0:
                    create_gpu_load()
                
                # Forward pass
                output = model(input_tensor)
                
                # Log network state for visualization
                if batch % 2 == 0:
                    log_network_state(model, batch)
                
                # Log predictions
                if batch % 3 == 0:
                    log_predictions(output, target)
                
                # Compute loss
                loss = criterion(output, target)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log batch metrics
                loss_value = loss.item()
                epoch_loss += loss_value
                
                # Calculate fake accuracy for visualization
                accuracy = 100.0 * (1.0 - loss_value / 10.0)  # Just for visualization
                accuracy = max(min(accuracy, 100.0), 0.0)  # Clamp to 0-100 range
                
                # Log to monitor
                monitor.log_batch(batch, loss=loss_value, accuracy=accuracy)
                
                # Print progress
                if batch % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch}, Loss: {loss_value:.4f}, Accuracy: {accuracy:.2f}%")
                
                # Add some computationally intensive operations to stress GPU every 5 batches
                if batch % 5 == 0:
                    # Generate two different load patterns for variety
                    if batch % 10 == 0:
                        # Matrix multiplication - more memory efficient
                        size = 1500  # Smaller size to prevent OOM
                        a = torch.randn(size, size, device=device)
                        b = torch.randn(size, size, device=device)
                        c = torch.matmul(a, b)
                        # Force evaluation
                        _ = c.sum().item()
                        # Clean up to save memory
                        del a, b, c
                    else:
                        # Run multiple smaller operations to maximize GPU utilization
                        for _ in range(5):
                            size = 800
                            a = torch.randn(size, size, device=device)
                            b = torch.randn(size, size, device=device)
                            c = torch.matmul(a, b)
                            d = torch.sin(c)
                            e = torch.matmul(d, d.t())
                            # Force evaluation
                            _ = e.sum().item()
                            # Clean up
                            del a, b, c, d, e
                            
                            # Clear CUDA cache if available to prevent OOM
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                
                # Simulate a bit of waiting to allow UI updates
                time.sleep(0.05)
            
            # Log epoch metrics
            avg_loss = epoch_loss / 50
            avg_accuracy = 100.0 * (1.0 - avg_loss / 10.0)
            avg_accuracy = max(min(avg_accuracy, 100.0), 0.0)
            
            monitor.log_epoch(epoch, loss=avg_loss, accuracy=avg_accuracy)
            print(f"Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.2f}%")
            
            # Check for user control
            if not monitor.check_control():
                print("Training stopped by user")
                return

if __name__ == "__main__":
    gpu_stress_test()
