import time
import random
from aliyah import monitor
from aliyah import trainingmonitor

def simulate_training():
    """Simulate a training process with metrics"""
    epochs = 10
    batches_per_epoch = 100
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        for batch in range(batches_per_epoch):
            # Check for control signals before each batch
            if not monitor.check_control():
                print("Training stopped by user")
                return
                
            # Simulate batch training
            loss = 1.0 / (1.0 + epoch + batch/batches_per_epoch)
            accuracy = min(100.0, (epoch * 10) + (batch / batches_per_epoch))
            
            # Log batch metrics
            monitor.log_batch(batch, {
                "loss": loss,
                "accuracy": accuracy
            })
            
            # Accumulate for epoch metrics
            epoch_loss += loss
            epoch_acc += accuracy
            
            # Simulate computation time and check for pause
            remaining_time = 0.1  # 100ms per batch
            while remaining_time > 0:
                if not monitor.check_control():
                    print("Training stopped by user")
                    return
                sleep_time = min(0.1, remaining_time)  # Check control every 100ms max
                time.sleep(sleep_time)
                remaining_time -= sleep_time
        
        # Log epoch metrics
        avg_loss = epoch_loss / batches_per_epoch
        avg_acc = epoch_acc / batches_per_epoch
        monitor.log_epoch(epoch, {
            "loss": avg_loss,
            "accuracy": avg_acc
        })
        
        # Print for TUI parsing
        print(f"Training epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.1f}%")

def main():
    """Main test function"""
    try:
        print("Starting training simulation...")
        with trainingmonitor():
            simulate_training()
            print("Training completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
