import time
import torch
import numpy as np
import random
from aliyah import trainingmonitor, monitor

def test_shallow_network():
    """Test visualization with a very small network to expose scaling issues."""
    print("Testing shallow network visualization...")
    
    # Configure layers for a minimal network (just 2 layers with few nodes)
    input_size = 2
    output_size = 1
    
    with trainingmonitor() as mon:
        # First, register the network architecture
        monitor.send_update("model_architecture", {
            "framework": "pytorch",
            "layers": [
                {"name": "input", "layer_type": "Input", "input_size": [input_size], "output_size": [input_size], "parameters": 0},
                {"name": "output", "layer_type": "Linear", "input_size": [input_size], "output_size": [output_size], "parameters": input_size * output_size + output_size}
            ],
            "total_parameters": input_size * output_size + output_size
        })
        
        # Wait for the UI to process
        time.sleep(1)
        
        # Test different activation patterns
        for step in range(30):
            # Alternate between activating different neurons
            activation_values = [abs(np.sin(step * 0.1 + i)) for i in range(input_size)]
            monitor.log_layer_state(0, activation_values)
            
            # Output neuron activation
            output_value = np.mean(activation_values)
            monitor.log_layer_state(1, [output_value])
            
            # Show connection flow
            for i in range(input_size):
                # Only activate connection if source neuron is active
                if activation_values[i] > 0.5:
                    monitor.log_connection_flow(0, i, 1, 0, True)
                else:
                    monitor.log_connection_flow(0, i, 1, 0, False)
            
            # Simulate computation time
            time.sleep(0.2)
            
            # Check for user control commands
            if not monitor.check_control():
                print("Test stopped by user")
                return

def test_medium_network():
    """Test visualization with a medium network."""
    print("Testing medium network visualization...")
    
    # Medium-sized network (3 layers)
    layer_sizes = [5, 7, 3]
    
    with trainingmonitor() as mon:
        # Register network architecture
        layers = []
        total_params = 0
        
        # Input layer
        layers.append({
            "name": "input", 
            "layer_type": "Input", 
            "input_size": [layer_sizes[0]], 
            "output_size": [layer_sizes[0]], 
            "parameters": 0
        })
        
        # Hidden and output layers
        for i in range(1, len(layer_sizes)):
            layer_params = layer_sizes[i-1] * layer_sizes[i] + layer_sizes[i]
            layers.append({
                "name": f"layer_{i}", 
                "layer_type": "Linear", 
                "input_size": [layer_sizes[i-1]], 
                "output_size": [layer_sizes[i]], 
                "parameters": layer_params
            })
            total_params += layer_params
            
            # Add activation layer if not the last layer
            if i < len(layer_sizes) - 1:
                layers.append({
                    "name": f"relu_{i}", 
                    "layer_type": "ReLU", 
                    "input_size": [layer_sizes[i]], 
                    "output_size": [layer_sizes[i]], 
                    "parameters": 0
                })
        
        monitor.send_update("model_architecture", {
            "framework": "pytorch",
            "layers": layers,
            "total_parameters": total_params
        })
        
        # Wait for UI to process
        time.sleep(1)
        
        # Simulate forward passes with varying activations
        for step in range(50):
            # Input layer activations
            input_activations = [abs(np.sin(step * 0.1 + i * 0.5)) for i in range(layer_sizes[0])]
            monitor.log_layer_state(0, input_activations)
            
            # Hidden layer activations (simulated)
            hidden_activations = []
            for i in range(layer_sizes[1]):
                # Create a pattern based on input layer
                hidden_val = sum(input_activations) / layer_sizes[0] * np.sin(step * 0.05 + i * 0.2)
                hidden_activations.append(abs(hidden_val))
                
            monitor.log_layer_state(1, hidden_activations)
            
            # Output layer activations (simulated)
            output_activations = []
            for i in range(layer_sizes[2]):
                # Create a pattern based on hidden layer
                output_val = sum(hidden_activations) / layer_sizes[1] * np.sin(step * 0.03 + i * 0.3)
                output_activations.append(abs(output_val))
                
            monitor.log_layer_state(2, output_activations)
            
            # Show some random active connections
            # Input to hidden
            for _ in range(3):
                from_idx = random.randint(0, layer_sizes[0]-1)
                to_idx = random.randint(0, layer_sizes[1]-1)
                is_active = input_activations[from_idx] > 0.5
                monitor.log_connection_flow(0, from_idx, 1, to_idx, is_active)
                
            # Hidden to output
            for _ in range(2):
                from_idx = random.randint(0, layer_sizes[1]-1)
                to_idx = random.randint(0, layer_sizes[2]-1)
                is_active = hidden_activations[from_idx] > 0.5
                monitor.log_connection_flow(1, from_idx, 2, to_idx, is_active)
            
            # Simulate a training step
            monitor.log_batch(step,  1.0 / (step + 1), min(step * 2, 100))
            
            # Every 10 steps, log epoch
            if step % 10 == 0:
                monitor.log_epoch(step // 10,  0.5 / (step // 10 + 1), min(step * 2, 100))
                # Send a prediction
                monitor.log_prediction(
                    [abs(np.sin(i * 0.5 + step * 0.1)) for i in range(3)],
                    ["Class A", "Class B", "Class C"],
                    f"Prediction at step {step}"
                )
            
            # Simulate computation time
            time.sleep(0.2)
            
            # Check for user control commands
            if not monitor.check_control():
                print("Test stopped by user")
                return

def test_deep_network():
    """Test visualization with a deep network to test scaling."""
    print("Testing deep network visualization...")
    
    # A deeper network with more layers
    layer_sizes = [10, 8, 6, 4, 2]
    
    with trainingmonitor() as mon:
        # Register network architecture
        layers = []
        total_params = 0
        
        # Input layer
        layers.append({
            "name": "input", 
            "layer_type": "Input", 
            "input_size": [layer_sizes[0]], 
            "output_size": [layer_sizes[0]], 
            "parameters": 0
        })
        
        # Hidden and output layers
        for i in range(1, len(layer_sizes)):
            layer_params = layer_sizes[i-1] * layer_sizes[i] + layer_sizes[i]
            layers.append({
                "name": f"layer_{i}", 
                "layer_type": "Linear", 
                "input_size": [layer_sizes[i-1]], 
                "output_size": [layer_sizes[i]], 
                "parameters": layer_params
            })
            total_params += layer_params
            
            # Add activation layer if not the last layer
            if i < len(layer_sizes) - 1:
                layers.append({
                    "name": f"relu_{i}", 
                    "layer_type": "ReLU", 
                    "input_size": [layer_sizes[i]], 
                    "output_size": [layer_sizes[i]], 
                    "parameters": 0
                })
        
        monitor.send_update("model_architecture", {
            "framework": "pytorch",
            "layers": layers,
            "total_parameters": total_params
        })
        
        # Wait for UI to process
        time.sleep(1)
        
        # Simulate forward passes with varying activations
        for step in range(40):
            # Update all layer activations
            for layer_idx, size in enumerate(layer_sizes):
                # Create different activation patterns for each layer
                activations = [abs(np.sin(step * 0.1 + i * 0.3 + layer_idx * 0.2)) for i in range(size)]
                monitor.log_layer_state(layer_idx, activations)
            
            # Add some connection flows between adjacent layers
            for layer_idx in range(len(layer_sizes) - 1):
                # Randomly activate 2 connections per layer
                for _ in range(2):
                    from_idx = random.randint(0, layer_sizes[layer_idx]-1)
                    to_idx = random.randint(0, layer_sizes[layer_idx+1]-1)
                    monitor.log_connection_flow(layer_idx, from_idx, layer_idx+1, to_idx, True)
            
            # Simulate a training step
            monitor.log_batch(step, 1.0 / (step + 1), min(step * 2.5, 100))
            
            # Every 5 steps, log epoch
            if step % 5 == 0:
                monitor.log_epoch(step // 5, 0.5 / (step // 5 + 1), min(step * 2.5, 100))
            
            # Simulate computation time
            time.sleep(0.25)
            
            # Check for user control commands
            if not monitor.check_control():
                print("Test stopped by user")
                return

def test_empty_model():
    """Test handling of empty model to ensure we don't panic on empty vectors."""
    print("Testing empty model handling (should not crash)...")
    
    with trainingmonitor() as mon:
        # Register an empty model architecture
        monitor.send_update("model_architecture", {
            "framework": "pytorch",
            "layers": [],
            "total_parameters": 0
        })
        
        # Wait a bit to see if UI crashes
        time.sleep(3)
        
        # Try to log some metrics anyway
        for i in range(5):
            monitor.log_batch(i, 1.0,  50.0)
            time.sleep(0.5)

def test_error_handling():
    """Test error handling in visualizations."""
    print("Testing error handling...")
    
    with trainingmonitor() as mon:
        # Send partial/invalid model architecture
        monitor.send_update("model_architecture", {
            "framework": "pytorch",
            "layers": [
                {"name": "layer1", "layer_type": "Linear"}  # Missing size information
            ],
            "total_parameters": 100
        })
        
        time.sleep(1)
        
        # Send activation for non-existent layer
        monitor.log_layer_state(99, [0.5, 0.6, 0.7])
        
        time.sleep(1)
        
        # Send connection for non-existent nodes
        monitor.log_connection_flow(0, 0, 1, 0, True)

def main():
    """Run different visualization tests sequentially."""
    print("Running visualization tests...")
    
    # Test empty model first to check for crashes
    test_empty_model()
    
    # Test shallow network
    test_shallow_network()
    
    # Test medium network
    test_medium_network()
    
    # Test deep network
    test_deep_network()
    
    # Test error handling
    test_error_handling()
    
    print("All tests completed!")

if __name__ == "__main__":
    main()
