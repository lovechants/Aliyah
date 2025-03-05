import jax
import jax.numpy as jnp
from jax import grad, jit, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from aliyah import trainingmonitor, monitor

def load_mnist():
    """Load MNIST data - simplified for example"""
    # This is a placeholder - you'd want to use real data loading
    key = random.PRNGKey(0)
    x_train = random.normal(key, (60000, 784))
    y_train = random.randint(key, (60000,), 0, 10)
    return x_train, y_train

def create_model():
    """Create a simple feedforward neural network"""
    return stax.serial(
        stax.Dense(128),
        stax.Relu,
        stax.Dense(64),
        stax.Relu,
        stax.Dense(10),
    )

def loss(params, batch):
    """Loss function for training"""
    inputs, targets = batch
    logits = predict(params, inputs)
    one_hot = jax.nn.one_hot(targets, 10)
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=1))

@jit
def accuracy(params, batch):
    """Compute accuracy for a batch"""
    inputs, targets = batch
    target_class = targets
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)

def train():
    # Initialize model and optimizer
    rng = random.PRNGKey(0)
    init_fun, predict = create_model()
    input_shape = (-1, 784)
    rng, init_rng = random.split(rng)
    _, init_params = init_fun(init_rng, input_shape)
    
    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)
    opt_state = opt_init(init_params)
    
    # Load data
    x_train, y_train = load_mnist()
    num_epochs = 10
    batch_size = 100
    num_batches = len(x_train) // batch_size
    
    # Training loop
    with trainingmonitor():
        for epoch in range(num_epochs):
            # Shuffle data
            rng, shuffle_rng = random.split(rng)
            perm = random.permutation(shuffle_rng, len(x_train))
            x_train = x_train[perm]
            y_train = y_train[perm]
            
            running_loss = 0.0
            running_accuracy = 0.0
            
            for batch_idx in range(num_batches):
                if not monitor.check_control():
                    print("Training stopped by user")
                    return
                
                # Get batch
                batch_start = batch_idx * batch_size
                batch_end = batch_start + batch_size
                batch = (x_train[batch_start:batch_end], y_train[batch_start:batch_end])
                
                # Compute loss and gradients
                params = get_params(opt_state)
                batch_loss = loss(params, batch)
                grad_fn = jax.grad(loss)
                grads = grad_fn(params, batch)
                
                # Update parameters
                opt_state = opt_update(batch_idx, grads, opt_state)
                
                # Compute accuracy
                batch_accuracy = accuracy(params, batch)
                
                # Update metrics
                running_loss += batch_loss
                running_accuracy += batch_accuracy
                
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = running_loss / 10
                    avg_accuracy = running_accuracy * 100 / 10
                    
                    # Log batch metrics
                    monitor.log_batch(batch_idx + 1, {
                        "loss": float(avg_loss),
                        "accuracy": float(avg_accuracy)
                    })
                    
                    print(f'Training epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.1f}%')
                    
                    running_loss = 0.0
                    running_accuracy = 0.0
            
            # Log epoch metrics
            monitor.log_epoch(epoch, {
                "loss": float(avg_loss),
                "accuracy": float(avg_accuracy)
            })

if __name__ == "__main__":
    train()
