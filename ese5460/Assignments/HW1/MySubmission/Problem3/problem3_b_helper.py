import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load the saved training data
with open('self_NN_training_loss.pkl', 'rb') as f:
    train_loss = pickle.load(f)

with open('self_NN_training_error.pkl', 'rb') as f:
    train_error = pickle.load(f)

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot training loss
ax1.plot(train_loss)
ax1.set_xlabel('Weight Updates')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss vs Weight Updates')

# Plot training error
ax2.plot(train_error)
ax2.set_xlabel('Weight Updates')
ax2.set_ylabel('Training Error')
ax2.set_title('Training Error vs Weight Updates')

plt.tight_layout()
plt.savefig('training_loss_error.png')
plt.show()