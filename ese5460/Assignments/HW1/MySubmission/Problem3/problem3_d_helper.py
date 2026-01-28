import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load Self-NN data
with open('self_NN_training_loss.pkl', 'rb') as f:
    self_train_loss = pickle.load(f)
with open('self_NN_training_error.pkl', 'rb') as f:
    self_train_error = pickle.load(f)
with open('self_NN_validation_loss.pkl', 'rb') as f:
    self_val_loss = pickle.load(f)
with open('self_NN_validation_error.pkl', 'rb') as f:
    self_val_error = pickle.load(f)

# Load PyTorch NN data
with open('pytorch_NN_training_loss.pkl', 'rb') as f:
    pt_train_loss = pickle.load(f)
with open('pytorch_NN_training_error.pkl', 'rb') as f:
    pt_train_error = pickle.load(f)
with open('pytorch_NN_validation_loss.pkl', 'rb') as f:
    pt_val_loss = pickle.load(f)
with open('pytorch_NN_validation_error.pkl', 'rb') as f:
    pt_val_error = pickle.load(f)

# X-axis for validation (every 1000 updates)
val_x = np.arange(0, len(self_val_loss) * 1000, 1000)

# Create figure with 4 subplots (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training Loss Comparison
axes[0, 0].plot(self_train_loss, label='Self-NN', alpha=0.7)
axes[0, 0].plot(pt_train_loss, label='PyTorch NN', alpha=0.7)
axes[0, 0].set_xlabel('Weight Updates')
axes[0, 0].set_ylabel('Training Loss')
axes[0, 0].set_title('Training Loss: Self-NN vs PyTorch NN')
axes[0, 0].legend()

# Plot 2: Training Error Comparison
axes[0, 1].plot(self_train_error, label='Self-NN', alpha=0.7)
axes[0, 1].plot(pt_train_error, label='PyTorch NN', alpha=0.7)
axes[0, 1].set_xlabel('Weight Updates')
axes[0, 1].set_ylabel('Training Error')
axes[0, 1].set_title('Training Error: Self-NN vs PyTorch NN')
axes[0, 1].legend()

# Plot 3: Validation Loss Comparison
axes[1, 0].plot(val_x, self_val_loss, 'o-', label='Self-NN')
axes[1, 0].plot(val_x, pt_val_loss, 's-', label='PyTorch NN')
axes[1, 0].set_xlabel('Weight Updates')
axes[1, 0].set_ylabel('Validation Loss')
axes[1, 0].set_title('Validation Loss: Self-NN vs PyTorch NN (every 1000 updates)')
axes[1, 0].legend()

# Plot 4: Validation Error Comparison
axes[1, 1].plot(val_x, self_val_error, 'o-', label='Self-NN')
axes[1, 1].plot(val_x, pt_val_error, 's-', label='PyTorch NN')
axes[1, 1].set_xlabel('Weight Updates')
axes[1, 1].set_ylabel('Validation Error')
axes[1, 1].set_title('Validation Error: Self-NN vs PyTorch NN (every 1000 updates)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('self_vs_pytorch_comparison.png')
plt.show()

# Print final metrics for comparison
print("=" * 60)
print("Final Performance Comparison")
print("=" * 60)
print(f"Self-NN Final Training Loss: {self_train_loss[-1]:.4f}")
print(f"PyTorch NN Final Training Loss: {pt_train_loss[-1]:.4f}")
print(f"Self-NN Final Training Error: {self_train_error[-1]:.4f}")
print(f"PyTorch NN Final Training Error: {pt_train_error[-1]:.4f}")
print(f"Self-NN Final Validation Loss: {self_val_loss[-1]:.4f}")
print(f"PyTorch NN Final Validation Loss: {pt_val_loss[-1]:.4f}")
print(f"Self-NN Final Validation Error: {self_val_error[-1]:.4f}")
print(f"PyTorch NN Final Validation Error: {pt_val_error[-1]:.4f}")
