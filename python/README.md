# Aliyah - ML Training Monitor

Aliyah is a terminal-based machine learning training monitor that lets you visualize and interact with your model training in real-time.

## Installation

```bash
# Install both components
pip install aliyah
cargo install aliyah

# Or from source
git clone https://github.com/lovechants/Aliyah.git
cd Aliyah
cargo build --release
pip install -e python/
```

## Features

- Real-time visualization of model training
- Interactive controls (pause/resume/stop)
- Network architecture visualization
- Training metrics plotting
- System resource monitoring
- Support for PyTorch, JAX, and other frameworks

## Usage

```python
from aliyah import monitor, trainingmonitor

# Inside your training code
with trainingmonitor() as monitor:
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Your training code
            loss = ...
            accuracy = ...
            
            # Log metrics
            monitor.log_batch(batch_idx, loss, accuracy, extra_metrics=extra_metrics)
            
            # Check if user paused/stopped
            if not monitor.check_control():
                break
        
        # Log epoch metrics
        monitor.log_epoch(epoch, val_loss, val_accuracy)
```

## Keyboard Controls

- `q/ESC`: Quit
- `p/SPACE`: Pause/Resume training
- `s`: Stop training
- `e`: Toggle error log
- `↑/↓`: Scroll logs
- `c`: Clear error log
- `h`: Show help
- `tab/n`: Show node information
- `click`: Switch training and node panel
- `o`: Output panel

## Framework Support

- ✅ PyTorch
- 🚧 JAX
- 🚧 TensorFlow/Keras
- 🚧 Scikit-Learn

## License

MIT
