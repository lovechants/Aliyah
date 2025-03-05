from aliyah import monitor 
import time 

def train():
    for i in range(5):
        print(f"Training epoch: {i}")
        time.sleep(.5)

with monitor():
    train()


