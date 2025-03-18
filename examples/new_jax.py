import jax
import jax.numpy as jnp
import jax.example_libraries
from jax import grad, jit, random, vmap
import numpy as np
import time
from aliyah import trainingmonitor, monitor
from jax.example_libraries import optimizers

def generate_synthetic_data(rng_key, n_samples=1000, n_features=8, n_classes=3):
    """Generate synthetic classification data."""
    key, subkey = random.split(rng_key)
    
    # Generate random weights
    w = random.normal(subkey, (n_features, n_classes))
    
    # Generate random features
    key, subkey = random.split(key)
    X = random.normal(subkey, (n_samples, n_features))
    
    # Generate outputs with some noise
    key, subkey = random.split(key)
    noise = 0.1 * random.normal(subkey, (n_samples, n_classes))
    logits = jnp.dot(X, w) + noise
    
    # Apply softmax to get probabilities
    probs = jax.nn.softmax(logits, axis=1)
    
    # Get class labels
    y = jnp.argmax(probs, axis=1)
    
    return X, y, ["Class A", "Class B", "Class C"]

def init_network_params(sizes, rng_key):
    """Initialize network parameters with random values."""
    keys = random.split(rng_key, len(sizes))
    return [init_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def init_layer_params(m, n, key):
    """Initialize parameters for a single layer."""
    w_key, b_key = random.split(key)
    return {
        'weights': random.normal(w_key, (m, n)) * 0.1,
        'bias': random.normal(b_key, (n,)) * 0.1
    }

def register_network_architecture(layer_sizes):
    """Register network architecture with Aliyah."""
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
    
    # Hidden layers and output layer
    for i in range(1, len(layer_sizes)):
        in_size = layer_sizes[i-1]
        out_size = layer_sizes[i]
        
        # Dense layer
        params = in_size * out_size + out_size  # weights + bias
        layer_info = {
            "name": f"dense_{i}",
            "layer_type": "Dense",
            "input_size": [in_size],
            "output_size": [out_size],
            "parameters": params
        }
        layers.append(layer_info)
        total_params += params
        
        # ReLU activation (except for output layer)
        if i < len(layer_sizes) - 1:
            layers.append({
                "name": f"relu_{i}",
                "layer_type": "ReLU",
                "input_size": [out_size],
                "output_size": [out_size],
                "parameters": 0
            })
    
    # Send to monitor
    monitor.send_update("model_architecture", {
        "framework": "jax",
        "layers": layers,
        "total_parameters": total_params
    })

def relu(x):
    """ReLU activation function."""
    return jnp.maximum(0, x)

def predict(params, x):
    """Forward pass prediction with intermediate activations."""
    activations = [x]  # Store input as first activation
    
    # Forward pass through layers
    for i, layer_params in enumerate(params):
        x = jnp.dot(x, layer_params['weights']) + layer_params['bias']
        activations.append(x)
        
        # Apply ReLU to all but the last layer
        if i < len(params) - 1:
            x = relu(x)
            activations.append(x)
    
    return x, activations

def loss_fn(params, x_batch, y_batch):
    """Loss function (cross-entropy loss)."""
    logits, _ = predict(params, x_batch)
    one_hot = jax.nn.one_hot(y_batch, 3)
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=1))

def accuracy(params, x_batch, y_batch):
    """Compute accuracy."""
    logits, _ = predict(params, x_batch)
    return jnp.mean(jnp.argmax(logits, axis=1) == y_batch)

def visualize_layer_activations(layer_sizes, epoch):
    """Generate and send simulated layer activations for visualization."""
    # We use synthetic activations since JAX doesn't expose intermediate activations easily
    for i, size in enumerate(layer_sizes):
        # Create some pattern based on layer and epoch
        t = jnp.linspace(0, 1, size)
        factor = 0.5 + 0.3 * jnp.sin(t * (i+1) * jnp.pi + epoch * 0.2)
        
        # Send to monitor
        monitor.log_layer_state(i, factor.tolist())
    
    # Add some connection flows for visualization
    for i in range(len(layer_sizes) - 1):
        # Add a few connections per layer
        for _ in range(min(3, layer_sizes[i])):
            from_node = np.random.randint(0, layer_sizes[i])
            to_node = np.random.randint(0, layer_sizes[i+1])
            monitor.log_connection_flow(i, from_node, i+1, to_node, True)

def train():
    # Initialize random key
    rng_key = random.PRNGKey(0)
    
    # Generate synthetic data
    rng_key, subkey = random.split(rng_key)
    X, y, class_names = generate_synthetic_data(subkey, n_samples=1000)
    
    # Split data into train/test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Define network architecture
    layer_sizes = [8, 16, 8, 3]  # input, hidden1, hidden2, output
    
    # Initialize parameters
    rng_key, subkey = random.split(rng_key)
    params = init_network_params(layer_sizes, subkey)
    
    # Register network architecture
    register_network_architecture(layer_sizes)
    
    # Optimization hyperparameters
    num_epochs = 20
    batch_size = 32
    step_size = 0.01
    num_batches = len(X_train) // batch_size
    
    # Initialize optimizer (simple SGD)
    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(params)
    
    # JIT-compile gradient and update functions
    @jit
    def update(params, x, y, opt_state):
        """Compute gradient and update parameters."""
        grads = grad(loss_fn)(params, x, y)
        return opt_update(0, grads, opt_state)
    
    # Start training with Aliyah monitoring
    with trainingmonitor():
        for epoch in range(num_epochs):
            # Check if we should continue
            if not monitor.check_control():
                print("Training stopped by user")
                break
            
            # Shuffle data
            rng_key, subkey = random.split(rng_key)
            perm = random.permutation(subkey, len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            # Train for one epoch
            epoch_loss = 0.0
            epoch_acc = 0.0
            
            for batch in range(num_batches):
                # Get batch
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Update parameters
                params = get_params(opt_state)
                batch_loss = loss_fn(params, X_batch, y_batch)
                batch_acc = accuracy(params, X_batch, y_batch) * 100
                opt_state = update(params, X_batch, y_batch, opt_state)
                
                # Accumulate metrics
                epoch_loss += batch_loss
                epoch_acc += batch_acc
                
                # Log batch metrics
                if batch % 5 == 0 or batch == num_batches - 1:
                    avg_loss = epoch_loss / (batch + 1)
                    avg_acc = epoch_acc / (batch + 1)
                    
                    monitor.log_batch(batch, float(avg_loss), float(avg_acc))
                    
                    # Visualize layer activations
                    visualize_layer_activations(layer_sizes, epoch)
                    
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch+1}/{num_batches}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%")
            
            # Evaluate on test set
            params = get_params(opt_state)
            test_loss = loss_fn(params, X_test, y_test)
            test_acc = accuracy(params, X_test, y_test) * 100
            
            # Log epoch metrics
            monitor.log_epoch(epoch, float(test_loss), float(test_acc))
            
            # Visualize predictions
            # Get a random test sample
            rng_key, subkey = random.split(rng_key)
            sample_idx = random.randint(subkey, (), 0, len(X_test))
            sample_X = X_test[sample_idx:sample_idx+1]
            sample_y = y_test[sample_idx].item()
            
            # Get predictions
            logits, _ = predict(params, sample_X)
            probs = jax.nn.softmax(logits, axis=1)[0]
            
            # Log prediction
            monitor.log_prediction(
                probs.tolist(),
                class_names,
                f"Prediction (Epoch {epoch+1}, True: {class_names[sample_y]})"
            )
            
            print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Show some final predictions
    params = get_params(opt_state)
    for i in range(3):
        # Get a random test sample
        rng_key, subkey = random.split(rng_key)
        sample_idx = random.randint(subkey, (), 0, len(X_test))
        sample_X = X_test[sample_idx:sample_idx+1]
        sample_y = y_test[sample_idx].item()
        
        # Get prediction
        logits, _ = predict(params, sample_X)
        probs = jax.nn.softmax(logits, axis=1)[0]
        
        # Log prediction
        monitor.log_prediction(
            probs.tolist(),
            class_names,
            f"Final Example {i+1} (True: {class_names[sample_y]})"
        )
        
        # Space out predictions
        time.sleep(0.5)

if __name__ == "__main__":
    train()
