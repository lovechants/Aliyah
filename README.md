# Aliyah
Aliyah is a functional training suite for machine learning in the terminal that allows you to interact with and visualize your model as it trains!
You can find a wrapper/hook library in `\python`.

```bash
aliyah <script.py>
```

## Version 0.1.0 is now available!
[PIP / PyPi](https://pypi.org/project/aliyah/0.1.0/)

[Cargo / Crates.io](https://crates.io/crates/aliyah)

## Installation
#### Install with Pacakge Managers
```bash
cargo install aliyah # Rust
pip install aliyah # Python
```

#### Install with curl
```bash
curl -sSL https://raw.githubusercontent.com/lovechants/Aliyah/main/install.sh | bash
```

#### Install with Python
```bash
python -m pip install aliyah
python -c "$(curl -sSL https://raw.githubusercontent.com/lovechants/Aliyah/main/install.py)"
```

#### Build from source
```bash
git clone https://github.com/lovechants/Aliyah
cd Aliyah
cargo build --release
pip install -e python/
```
If you're building from source, note that the `data` folder is only needed for the example scripts.

## Quick Start
After installation, you can easily add monitoring to your code:
```python
# Inside your training code
with trainingmonitor() as monitor:
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Your training code
            loss = ...
            accuracy = ...

            # Log metrics
            monitor.log_batch(batch_idx, loss, accuracy)

            # Check if user paused/stopped
            if not monitor.check_control():
                break

        # Log epoch metrics
        monitor.log_epoch(epoch, val_loss, val_accuracy)
```

## Keyboard Controls
- Q/ESC: Quit
- P/SPACE: Pause or resume training
- S: Stop training
- E: Toggle error log
- ↑/↓: Scroll error log or training log
- C: Clear error log
- H: Show this help page
- TAB/N: Show node information
- CLICK: Switch training and node panel
- O: Output panel

Visualize and track more data depending on what should be sent and monitored.
The examples in `examples` show in-depth metric tracking and visualization control.
Full documentation for each function will eventually be provided.
For now, please view the source code in the `/python` directory.

## Current Features

#### Real Time visualization!
See a real time visualization of your model architecture, including activations and connections between layers.
Gain insights into each node and edge as the model learns.

#### Training Metrics!
Monitor key metrics appropriate for your model in real time with an interactive log and a real time chart.
Send custom metrics between both using `**KWARG` in the associated functions.

#### System Resource Monitoring!
Keep track of CPU/GPU/memory usage while the model trains.

#### Interactive Control!
Pause, resume, or stop training without killing the process.
This allows you to have real time control throughout the entire training process.

#### Real Time Prediction!
Check on your model's current performance with the prediction panel.

## Planned Features
- User configuration
- Layer-specific visualization
- Advanced GPU monitoring
- Custom metric tracking
- Interactive parameter adjustment
- Extended framework support
- Advanced network visualization
- Custom algorithm support

## Framework Support
| Framework        | Supported? |
|------------------|------------|
| PyTorch          | ✅         |
| JAX              | 🚧         |
| TensorFlow/Keras | 🚧         |
| Scikit-Learn     | 🚧         |
| TinyGrad         | 🚧         |

Contributions to framework hooks and behaviors are appreciated, along with any suggestions.

## Examples

Check out the examples to make the most of Aliyah's features:
- `simpleNet.py`: A simple nerual network that demonstrates training on MNIST data, active visualizations, and metric logging.
- `example_vae.py`: An example autoencoder adapted from PyTorch, showing custom metrics, active visualizations, and a custom prediction panel.
- `error_test.py`: A simple test to show how the error log and errors are recorded in the TUI.

There are a variety of other scripts present for testing purposes.
Feel free to look at them if you are using another framework besides PyTorch.

## Example Images
Visualization hooks differ from framework to framework.
As more frameworks are supported, more images will be added here.
![MINST Rendering](example.png)
![VAE Example 1](example1.png)
![SimpleNet Output](example2.png)
![SimpleNet Example 1](example3.png)
![VAE Example 2](example4.png)
![VAE Output](example5.png)

## Current Todo List
- [ ] Context manager implementation
- [ ] Basic GPU stats
- [ ] Add LSP context to hook functions
- [ ] Fix logging crash
- [ ] Fix prediction screen text
- [ ] Redo drawing logic
- [ ] Add more node information
- [ ] Make visualization better for networks
- [ ] Make classic machine learning visualizations (not just networks)
- [ ] Update examples (Make more robust examples with the new features and remove the original test examples)
    - [ ] Metal CNN | Synthetic generation of CIFAR-10
    - [ ] Deep network
    - [ ] Shallow network
    - [ ] Transformer
    - [ ] Algorithmic Pipeline -> PSO -> PNN (no boltzmann) 
      [adapted from this paper](https://ieeexplore.ieee.org/document/6525976),
      [custom implmentation](https://codeberg.org/8cu/intrusion_detection)
        - Either show each algorithm / model indepent of each other move to the next
        - Or show all of them at the same time running async?
- [ ] Fix output window match statements to be more robust (On pause or stopped script states)
- [ ] Make framework hooks for visualizations
    - [ ] JAX
    - [ ] Keras
    - [ ] TF
    - [ ] TinyGrad
    - [ ] SciKit Learn
    - [ ] Custom
    - [ ] Default
- [ ] User Config (Themeing)
- [ ] Save model weights, save plots and simple metrics as nice visual jpeg
- [ ] Add other GPU monitoring
    - [ ] Test NVIDIA
    - [ ] Test AMD

## Contributing to Aliyah
Thank you for considering contributing to Aliyah! Here's how you can help:
1. Fork the repository.
2. Clone your fork: `git clone https://github.com/your-username/your-fork`
3. Create a branch: `git checkout -b your-feature-branch`
4. Make your changes.
5. Test your changes.
6. Commit your changes: `git commit -m "Add feature-that-you-added"`
7. Push to the branch: `git push origin your-feature-branch`
8. Submit a pull request.

There are no strict coding conventions in place, but please thoroughly test all code!

#### Development Environment
Set up your development environment:
```bash
# Clone your fork
git clone https://github.com/your-username/your-fork
cd Aliyah

# Build the Rust binary
cargo build

# Install the Python package in development mode
pip install -e python/
```

#### Development Requirements
1. Rust
2. Python3
3. C & C++ compiler (gcc, g++)
