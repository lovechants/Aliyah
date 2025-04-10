import torch
import torch.nn as nn
from aliyah import trainingmonitor, monitor

def test_metal():
    print(f"PyTorch: {torch.__version__}")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)
    model = nn.Sequential(
            nn.Linear(10,50),
            nn.ReLU(),
            nn.Linear(50,20),
            nn.ReLU(),
            nn.Linear(20,1)
    ).to(device)

    x = torch.randn(1000,10).to(device)
    y = torch.randn(1000,1).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    with trainingmonitor() as monitor: 
        for epoch in range(5):
            for batch in range(100):
                output = model(x)
                loss = criterion(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                monitor.log_batch(batch, loss=loss.item())

                if not monitor.check_control():
                    break 

            monitor.log_epoch(epoch, loss=loss.item())
            if not monitor.check_control():
                break

if __name__ == "__main__":
    test_metal()

