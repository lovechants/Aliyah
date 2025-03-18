import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from jax.example_libraries import optimizers, stax
import numpy as np
import time
from aliyah import trainingmonitor, monitor 

# Set a fixed random seed for reproducibility
key = random.PRNGKey(42)

# Generate a synthetic classification dataset
def generate_data(key, n_samples=1000, n_features=10, n_classes=3):
    key, subkey = random.split(key)
    X = random.normal(subkey, (n_samples, n_features))
    key, subkey = random.split(key)
    w = random.normal(subkey, (n_features, n_classes))
    logits = jnp.dot(X, w)
    y = jnp.argmax(logits, axis=1)
    return X, y, logits

def register_model_architecture(apply_fn, params):
    # Get the layer structure information
    layers = [
        {"name": "input", "layer_type": "Input", "input_size": [10], "output_size": [10]},
        {"name": "dense1", "layer_type": "Dense", "input_size": [10], "output_size": [64]},
        {"name": "relu1", "layer_type": "ReLU", "input_size": [64], "output_size": [64]},
        {"name": "dense2", "layer_type": "Dense", "input_size": [64], "output_size": [32]},
        {"name": "relu2", "layer_type": "ReLU", "input_size": [32], "output_size": [32]},
        {"name": "dense3", "layer_type": "Dense", "input_size": [32], "output_size": [3]},
    ]
    
    # Explicitly send model architecture info
    monitor.send_update("model_architecture", {
        "framework": "jax",
        "layers": layers,
        "total_parameters": sum(
            layer.get("input_size", [0])[0] * layer.get("output_size", [0])[0] 
            for layer in layers if "Dense" in layer["layer_type"]
        )
    })

# Initialize the model
def init_model():
    init_fn, apply_fn = stax.serial(
        stax.Dense(64),
        stax.Relu,
        stax.Dense(32),
        stax.Relu,
        stax.Dense(3),  # 3 output classes
    )
    
    rng = random.PRNGKey(0)
    input_shape = (-1, 10)  # 10 features
    out_shape, params = init_fn(rng, input_shape)
    return init_fn, apply_fn, params

# Loss function (cross entropy)
def loss_fn(params, apply_fn, X, y):
    logits = apply_fn(params, X)
    one_hot_y = jax.nn.one_hot(y, 3)
    # Softmax cross entropy loss
    return -jnp.mean(jnp.sum(one_hot_y * jax.nn.log_softmax(logits), axis=1))

# Compute accuracy
def accuracy(params, apply_fn, X, y):
    logits = apply_fn(params, X)
    predictions = jnp.argmax(logits, axis=1)
    return jnp.mean(predictions == y)

# Visualization function
def visualize_prediction(params, apply_fn, sample_X, sample_y, epoch):
    """Visualize model predictions for a sample"""
    # Get model predictions
    logits = apply_fn(params, sample_X)
    probabilities = jax.nn.softmax(logits, axis=1)
    predictions = jnp.argmax(logits, axis=1)
    
    # Get the first sample for visualization
    sample_idx = 0
    sample_probs = probabilities[sample_idx].tolist()
    sample_pred = int(predictions[sample_idx])
    sample_true = int(sample_y[sample_idx])
    
    # Class labels
    class_labels = ['Class 0', 'Class 1', 'Class 2']
    
    # Send prediction to monitor
    description = f"Prediction (epoch {epoch})"
    monitor.log_prediction(sample_probs, labels=class_labels, description=description)
    
    # Log to console for debugging
    print(f"Sample prediction: {class_labels[sample_pred]} (true: {class_labels[sample_true]})")
    print(f"Probabilities: {[f'{p:.4f}' for p in sample_probs]}")

def train():
    # Generate synthetic data
    X, y, _ = generate_data(key)
    
    # Split into train and validation sets (80/20)
    n_train = int(0.8 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    
    # Initialize the model and optimizer
    init_fn, apply_fn, params = init_model()
    
    # Create a JIT-compiled version of the gradient function with apply_fn fixed
    @jit
    def loss_grad_fn(params, X, y):
        return grad(lambda p: loss_fn(p, apply_fn, X, y))(params)
    
    # Use Adam optimizer
    opt_init, opt_update, get_params = optimizers.adam(step_size=0.01)
    opt_state = opt_init(params)
    
    # Set up training parameters
    n_epochs = 10
    batch_size = 32
    n_batches = int(np.ceil(len(X_train) / batch_size))
    
    # Select a few samples for visualization
    viz_key = random.PRNGKey(123)
    viz_indices = random.randint(viz_key, (5,), 0, len(X_val))
    X_viz = X_val[viz_indices]
    y_viz = y_val[viz_indices]
    
    # Create a jitted version of the training update
    @jit
    def update(i, opt_state, X_batch, y_batch):
        params = get_params(opt_state)
        grads = loss_grad_fn(params, X_batch, y_batch)
        return opt_update(i, grads, opt_state)
    
    # Training loop
    with trainingmonitor():
        for epoch in range(n_epochs):
            # Shuffle training data
            perm_key = random.PRNGKey(epoch)
            perm = random.permutation(perm_key, len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            # Train for one epoch
            start_time = time.time()
            epoch_loss = 0.0
            
            for batch in range(n_batches):
                # Check if we should continue
                if not monitor.check_control():
                    print("Training stopped by user")
                    return
                
                # Prepare batch
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, len(X_train))
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Update parameters
                params = get_params(opt_state)
                batch_loss = loss_fn(params, apply_fn, X_batch, y_batch)
                opt_state = update(batch, opt_state, X_batch, y_batch)
                
                # Accumulate loss
                epoch_loss += batch_loss
                
                # Log batch metrics every few batches
                if batch % 10 == 0 or batch == n_batches - 1:
                    params = get_params(opt_state)
                    batch_acc = accuracy(params, apply_fn, X_batch, y_batch) * 100
                    
                    # Log batch metrics
                    monitor.log_batch(batch, float(batch_loss), float(batch_acc))
                    
                    print(f"Epoch {epoch}, Batch {batch}/{n_batches-1}: Loss = {batch_loss:.4f}, Accuracy = {batch_acc:.2f}%")
            
            # Compute validation metrics
            params = get_params(opt_state)
            val_loss = loss_fn(params, apply_fn, X_val, y_val)
            val_acc = accuracy(params, apply_fn, X_val, y_val) * 100
            
            # Log epoch metrics
            monitor.log_epoch(epoch, float(val_loss), float(val_acc))  
            # Visualize model prediction for a sample
            visualize_prediction(params, apply_fn, X_viz, y_viz, epoch)
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.2f}%, Time: {epoch_time:.2f}s")
    
    # Final visualization with the trained model
    params = get_params(opt_state)
    
    # Get all validation predictions
    val_logits = apply_fn(params, X_val)
    val_probs = jax.nn.softmax(val_logits, axis=1)
    val_preds = jnp.argmax(val_logits, axis=1)
    
    # Log final accuracy
    final_acc = jnp.mean(val_preds == y_val) * 100
    print(f"Final validation accuracy: {final_acc:.2f}%")
    
    # Log multiple predictions
    class_labels = ['Class 0', 'Class 1', 'Class 2']
    for i in range(min(3, len(X_viz))):
        # Get single sample and reshape appropriately
        sample_x = X_viz[i:i+1]
        logits = apply_fn(params, sample_x)
        probabilities = jax.nn.softmax(logits, axis=1)[0]  # Take first sample from batch
        
        monitor.log_prediction(
            probabilities.tolist(),
            labels=class_labels,
            description=f"Sample {i} (true: Class {int(y_viz[i])})"
        )
        # Sleep briefly to allow UI to process
        time.sleep(0.5)

if __name__ == "__main__":
    train()
