import tensorflow as tf
from tensorflow import keras
import time
from aliyah import trainingmonitor, monitor

class TrainingCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._is_paused = False
        self._should_stop = False

    def on_train_begin(self, logs=None):
        print("Starting training...")

    def on_train_batch_begin(self, batch, logs=None):
        # Check for control signals before each batch
        while self._is_paused and not self._should_stop:
            time.sleep(0.1)  # Sleep briefly to prevent CPU spinning
            if not monitor.check_control():
                self._should_stop = True
                self.model.stop_training = True
                return
        
        # Check for stop signal
        if self._should_stop or not monitor.check_control():
            self.model.stop_training = True
            print("\nTraining stopped by user")
            return

    def on_train_batch_end(self, batch, logs=None):
        if batch % 100 == 0:  # Adjust frequency as needed
            # Convert metrics to float to ensure they're serializable
            metrics = {
                "loss": float(logs.get("loss", 0.0)),
                "accuracy": float(logs.get("accuracy", 0.0)) * 100
            }
            monitor.log_batch(batch, metrics)
            
            # Check control status
            control_status = monitor.check_control()
            if not control_status:
                self._should_stop = True
                self.model.stop_training = True
            elif hasattr(monitor, 'paused'):
                self._is_paused = monitor.paused

    def on_epoch_end(self, epoch, logs=None):
        metrics = {
            "loss": float(logs.get("val_loss", 0.0)),
            "accuracy": float(logs.get("val_accuracy", 0.0)) * 100
        }
        monitor.log_epoch(epoch, metrics)
        print(f'\nEpoch {epoch+1}, Val Loss: {metrics["loss"]:.4f}, Val Accuracy: {metrics["accuracy"]:.1f}%')

def create_model():
    return keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])


def train():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Create and compile model
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        # Disable eager execution for better control
        run_eagerly=False
    )
    
    # Create callback
    aliyah_callback = TrainingCallback()
    
    # Configure training to enable interruption
    try:
        with trainingmonitor():
            model.fit(
                x_train, y_train,
                batch_size=100,
                epochs=10,
                validation_data=(x_test, y_test),
                callbacks=[aliyah_callback],
                verbose=1,  # Keep progress bar for visual feedback
                workers=1,  # Use single worker for better control
                use_multiprocessing=False  # Disable multiprocessing for better control
            )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise


if __name__ == "__main__":
    train()
