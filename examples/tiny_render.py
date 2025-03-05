import torch
import torch.nn as nn
import time
from aliyah import trainingmonitor

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Minimal network - just 3 layers with few nodes
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 2)
        self.fc3 = nn.Linear(2, 1)

def main():
    with trainingmonitor() as monitor:
        # Wait a bit for ZMQ connection
        time.sleep(0.5)
        
        # First test - just send some static layer states
        print("Sending initial layer states...")
        
        # Layer 0 (input)
        monitor.log_layer_state(0, [0.5, 0.8])
        time.sleep(0.1)
        
        # Layer 1 (hidden)
        monitor.log_layer_state(1, [0.3, 0.6, 0.9])
        time.sleep(0.1)
        
        # Layer 2 (hidden)
        monitor.log_layer_state(2, [0.4, 0.7])
        time.sleep(0.1)
        
        # Layer 3 (output)
        monitor.log_layer_state(3, [0.5])
        time.sleep(0.1)

        print("Testing individual activations...")
        # Test individual node activations
        for layer in range(4):
            for node in range(3):  # Max 3 nodes per layer
                try:
                    monitor.log_layer_activation(layer, node, 0.8)
                    time.sleep(0.1)
                except:
                    continue

        print("Testing connections...")
        # Test connections between layers
        for layer in range(3):  # Connect adjacent layers
            monitor.log_connection_flow(layer, 0, layer + 1, 0, True)
            time.sleep(0.1)

        # Keep the visualization running for a bit
        for _ in range(50):
            if not monitor.check_control():
                break
            time.sleep(0.1)

if __name__ == "__main__":
    main()
