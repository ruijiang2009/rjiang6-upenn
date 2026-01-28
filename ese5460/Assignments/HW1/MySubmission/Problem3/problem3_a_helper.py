import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as T
import importlib

# Import module with numeric name using importlib
main_module = importlib.import_module("62502470_hw1_problem3")
subsample = main_module.subsample

def get_data():
    mnist_train = MNIST(root='./data', train=True, download=True, transform=T.ToTensor())
    return mnist_train

# Load and prepare data
mnist_train = get_data()
trainX = mnist_train.data.numpy().reshape(-1, 784)
trainY = mnist_train.targets.numpy()

# Subsample
trainX, trainY = subsample(trainX, trainY, num_samples=10000, num_classes=10)

# Plot 4 random images
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
random_indices = np.random.choice(len(trainX), 4, replace=False)

for i, idx in enumerate(random_indices):
    image = trainX[idx].reshape(28, 28)
    label = trainY[idx]
    
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('mnist_samples.png')
plt.show()