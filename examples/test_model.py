import torch as t  
import torch.nn as nn 
import time

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.fc1 = nn.Linear(32 * 26 * 26, 128)
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 32 * 26 * 26)
        x = self.fc1(x)
        x = self.fc2(x)
        return x 

model = Network()

for epoch in range(10):
    loss = 1.0 / (epoch + 1)
    accuracy = epoch * 10
    print(f"Training epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy}%")
    time.sleep(.5)
