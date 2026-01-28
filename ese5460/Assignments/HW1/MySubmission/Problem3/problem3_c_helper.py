import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load all data
with open('self_NN_training_loss.pkl', 'rb') as f:
    train_loss = pickle.load(f)
with open('self_NN_training_error.pkl', 'rb') as f:
    train_error = pickle.load(f)
with open('self_NN_validation_loss.pkl', 'rb') as f:
    val_loss = pickle.load(f)
with open('self_NN_validation_error.pkl', 'rb') as f:
    val_error = pickle.load(f)

# X-axis for validation (every 1000 updates)
val_x = np.arange(0, len(val_loss) * 1000, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot loss comparison
ax1.plot(train_loss, label='Training Loss', alpha=0.7)
ax1.plot(val_x, val_loss, 'o-', label='Validation Loss')
ax1.set_xlabel('Weight Updates')
ax1.set_ylabel('Loss')
ax1.set_title('Training vs Validation Loss')
ax1.legend()

# Plot error comparison
ax2.plot(train_error, label='Training Error', alpha=0.7)
ax2.plot(val_x, val_error, 'o-', label='Validation Error')
ax2.set_xlabel('Weight Updates')
ax2.set_ylabel('Error')
ax2.set_title('Training vs Validation Error')
ax2.legend()

plt.tight_layout()
plt.savefig('train_vs_val_comparison.png')
plt.show()