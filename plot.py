import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('logs.pkl', 'rb') as f:
    logs = pickle.load(f)
# print(logs)

# Define the keys for subplots
keys = list(range(10))

# Create a subplot with 2 rows and 2 columns
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot loss and accuracy for keys 0 to 9
for i in range(10):
    key = keys[i]
    axs[0, 0].plot(logs[key]['loss'], label=f'client {key}')
    axs[0, 1].plot(logs[key]['accuracy'], label=f'client {key}')

# Plot loss and accuracy for key 'test'
axs[1, 0].plot(logs['test']['loss'], label='test')
axs[1, 1].plot(logs['test']['accuracy'], label='test')

# Set titles and labels
axs[0, 0].set_title("Clients' Loss")
axs[0, 1].set_title("Clients' Accuracy")
axs[1, 0].set_title('Test data Loss')
axs[1, 1].set_title('Test data Accuracy')

# Add legends
axs[0, 0].legend()
axs[0, 1].legend()
axs[1, 0].legend()
axs[1, 1].legend()

plt.tight_layout()
plt.savefig('logs.png')
plt.show()