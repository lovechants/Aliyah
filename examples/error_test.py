import jax
import torch 
import torch.nn as nn

class Badnetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.fc1 = nn.Linear(32,128) # Bad dimensions 
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.fc1(x) # Should be a runtime error here
        x = self.fc2(x)
        return x 

model = Badnetwork()
x = torch.randn(1, 1, 28, 28)
output = model(x) # Error 

print(" End of error testing script ")
