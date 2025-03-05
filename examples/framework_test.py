import jax 
import sys 
import time 

print(f"Script Args: {sys.argv[1:]}")

for epoch in range(10):
    loss = 1.0 / (epoch + 1)
    accuracy = epoch * 10
    print(f"Training epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy}%")
