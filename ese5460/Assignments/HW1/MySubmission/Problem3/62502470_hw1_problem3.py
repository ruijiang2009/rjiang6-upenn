### Run your code with python >= 3.9 ###

import numpy as np

import h5py
import pickle
import dill
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision as thv
from torchvision.datasets import MNIST
import torchvision.transforms as T

import matplotlib.pyplot as plt
import pickle, random
from copy import deepcopy

dill.settings["recurse"] = True

def setup(s):
    """
    DO NOT CHANGE
    """
    torch.manual_seed(s)
    random.seed(s)
    np.random.seed(s)


def get_data():
    """
    Complete for subproblem 3a

    Download training and validation MNIST data using MNIST()

    Return: raw MNIST trianing data, raw MNIST validation data
    """
    ###TODO###
    mnist_train = MNIST(root='./data', train=True, download=True, transform=T.ToTensor())
    mnist_val = MNIST(root='./data', train=False, download=True, transform=T.ToTensor())
    return mnist_train, mnist_val


def subsample(X, Y, num_samples, num_classes):
    """
    Complete for subproblem 3a

    Subsample dataset into size num_samples
    Have equal data for each class
    Normalize x

    Inputs:
    -------
    X: Data Matrix
    Y: Target Values/Vector
    num_samples: The number of subsampled data points (Output)
    num_classes: Total number of classes in the dataset (How many targets/labels do we have in MNIST?)

    Returns:
    --------
    subsampled_x, subsampled_y (self-explanatory)
    """
    ###TODO###
    samples_per_class = num_samples // num_classes
    subsampled_x = []
    subsampled_y = []

    for digit in range(num_classes):
        # Find all indices for this digit
        indices = np.where(Y == digit)[0]
        # Randomly select samples_per_class indices
        selected_indices = np.random.choice(indices, samples_per_class, replace=False)
        subsampled_x.append(X[selected_indices])
        subsampled_y.append(Y[selected_indices])

    subsampled_x = np.vstack(subsampled_x)
    subsampled_y = np.hstack(subsampled_y)

    # Normalize x to [0, 1] range
    subsampled_x = subsampled_x / 255.0

    return subsampled_x, subsampled_y

class MNISTDataset(Dataset):
    """
        Complete for subproblem 3a

        Define a dataset class to initialize Dataloader objects
    """
    ###TODO###
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y


class linear_t:
    """
    Complete for subproblem 3b

    Define the linear layer class for the neural network
    """

    def __init__(self):
        """
        initialize to appropriate sizes, fill with Gaussian entires
        """
        # w: (10, 784) for MNIST (10 classes, 784 input features)
        # b: (10, 1) bias for each output class
        self.w = np.random.randn(10, 784)
        self.b = np.random.randn(10, 1)
        # Initialize gradients
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.hm = None

    def forward(self, hm):
        """
        hm: activations of a previous layer
        h: return activations of linear layer
        """
        # h = w @ hm.T + b, then transpose back
        # hm shape: (batch_size, 784)
        h = (self.w @ hm.T + self.b).T  # (batch_size, 10)
        self.hm = hm
        return h

    def backward(self, dh):
        """
        dh: dl/dh created by the next layer
        """
        # dh shape: (batch_size, 10)
        # dhm = dh @ w
        dhm = dh @ self.w  # (batch_size, 784)
        # dw = dh.T @ hm
        self.dw = dh.T @ self.hm  # (10, 784)
        # db = sum of dh across batch
        self.db = np.sum(dh.T, axis=1, keepdims=True)  # (10, 1)
        return dhm

    # notice that there is no need to cache dh^l return dh^l
    def zero_grad(self):
        """
        useful to delete the stored backprop gradients of the
        previous mini-batch before you start a new mini-batch
        """
        self.dw = 0*self.dw
        self.db = 0*self.db

    def backward_check_dw(self,n =10):
        """
        Complete for subproblem 3e


        """
        # dictionary to store calculated variables for autograder
        ag_dict = {
            'k': [],
            'hm': [],
            'dw': [],
            'dw_e': [],
            'i': [],
            'j': [],
            'e': []
        }
        ###TODO###
        for _ in range(n):
            k = np.random.randint(0, 10)
            dh = np.zeros((1, 10))
            dh[:, k] = 1
            # true forward
            hm = np.random.randn(1, 784)
            # TODO: true dw - compute gradient analytically
            self.forward(hm)
            self.backward(dh)
            dw = self.dw.copy()

            ag_dict['k'].append(k)
            ag_dict['hm'].append(hm)
            ag_dict['dw'].append(dw)

            for _ in range(100):
                e = np.zeros((10, 784));
                i, j = np.random.randint(0, 10), np.random.randint(0, 784)
                e[i, j] = np.random.randn()
                # TODO: estimated dw
                # Finite difference approximation
                self.w += e
                h_plus = self.forward(hm)
                self.w -= 2 * e
                h_minus = self.forward(hm)
                self.w += e
                dw_e = (h_plus[0, k] - h_minus[0, k]) / (2 * e[i, j])
                assert (np.linalg.norm(dw_e - dw[i, j]) < 1e-6)

                ag_dict['i'].append(i)
                ag_dict['j'].append(j)
                ag_dict['e'].append(e)
                ag_dict['dw_e'].append(dw_e)

        return ag_dict

    def backward_check_db(self, n=10):
        ag_dict = {
            'k':[],
            'hm':[],
            'db_e':[],
            'db':[]
        }

        for _ in range(n):
            k = np.random.randint(0, 10)
            dh = np.zeros((1, 10))
            dh[:, k] = 1
            # true forward/backward
            hm = np.random.randn(1, 784)
            # TODO: true db
            self.forward(hm)
            self.backward(dh)
            db = self.db.copy()

            # TODO: estimated db
            db_e = np.zeros((10,1))
            db_e[k] = 1
            assert(np.linalg.norm(db_e-db) < 1e-6)

            ag_dict['k'].append(k)
            ag_dict['hm'].append(hm)
            ag_dict['db_e'].append(db_e)
            ag_dict['db'].append(db)

        return ag_dict

    def backward_check_dhm(self, n=10):
        ag_dict = {
            'k':[],
            'hm':[],
            'dhm':[],
            'dhm_e':[],
            'e':[],
            'i':[]
        }
        for _ in range(n):
            k = np.random.randint(0, 10)
            dh = np.zeros((1, 10))
            dh[:, k] = 1
            # true forward/backward
            hm = np.random.randn(1, 784)
            # TODO: true dhm
            self.forward(hm)
            dhm = self.backward(dh).copy()

            ag_dict['k'].append(k)
            ag_dict['hm'].append(hm)
            ag_dict['dhm'].append(dhm)

            # dhm is similar to dw
            for _ in range(100):
                e = np.zeros((1, 784))
                i =np.random.randint(0,784)
                e[0,i]=np.random.randn()
                # TODO: estimated dhm
                # Finite difference approximation
                h_plus = self.forward(hm + e)
                h_minus = self.forward(hm - e)
                dhm_e = (h_plus[0, k] - h_minus[0, k]) / (2 * e[0, i])
                assert(np.linalg.norm(dhm_e-dhm[0,i]) < 1e-6)

                ag_dict['e'].append(e)
                ag_dict['i'].append(i)
                ag_dict['dhm_e'].append(dhm_e)

        return ag_dict

    def save(self, filename):
        # DO NOT EDIT
        # To save weight and bias values of the trained model
        with h5py.File(filename, 'w') as f:
            grp = f.create_group('instance')
            grp.attrs['w'] = self.w
            grp.attrs['b'] = self.b

class relu_t:
    """
    Complete for subproblem 3c
    """
    def __init__(self):
        self.hm = None

    def forward(self,hm):
        self.hm = hm
        # ReLU: max(0, x)
        return np.maximum(0, hm)

    def backward(self,dh):
        # Gradient of ReLU: 1 if hm > 0, else 0
        dhm = dh * (self.hm > 0)
        return dhm

    def zero_grad(self):
        self.hm = None


class softmax_cross_entropy_t:
    """
    Complete for subproblem 3d
    """

    def __init__(self):
        self.y, self.prob = None, None

    def forward(self, h, y):
        # h: (batch_size, 10) logits
        # y: (batch_size,) labels
        self.y = y

        # Compute softmax probabilities (numerically stable)
        h_max = np.max(h, axis=1, keepdims=True)
        exp_h = np.exp(h - h_max)
        self.prob = exp_h / np.sum(exp_h, axis=1, keepdims=True)

        # Compute cross-entropy loss
        batch_size = h.shape[0]
        log_prob = -np.log(self.prob[np.arange(batch_size), y] + 1e-10)

        # compute the average loss over mini-batch
        ce = np.mean(log_prob)

        # Compute error (misclassification rate)
        predictions = np.argmax(self.prob, axis=1)
        err = np.mean(predictions != y)

        return ce, err

    def backward(self):
        """
        As we saw in the notes, the backprop input to the
        loss layer is 1, so this function does not take any
        arguments
        """
        # Gradient of cross-entropy with softmax
        batch_size = len(self.y)
        dh = self.prob.copy()
        dh[np.arange(batch_size), self.y] -= 1
        dh /= batch_size  # Average over batch
        return dh

    def zero_grad(self):
        self.y, self.prob = None, None

def test_backward():
    """
    Helper function to call all backward checks
    """
    l1=linear_t()
    l1.backward_check_dhm(n=10)
    l1.backward_check_db(n=10)
    l1.backward_check_dw(n=10)


def train_self_nn(train_dataloader, val_dataloader, lr=0.1):
    """
    Complete for subproblem 3f & g

    Train the neural network with the layers you made from scratch
    Save the training loss and error for every weight update for subproblem f

    Additionally, for every 1,000 weight updates, get the validation loss and error
    by calling the helper function validate_self_nn()
    Save the validation loss and error values for subproblem g
    """
    # Do not change the setup() call below
    setup(20)

    train_error_list = []
    train_loss_list = []
    val_error_list = []
    val_loss_list = []

    ###TODO###
    # initialize all the layers
    l1, l2, l3 = linear_t(), relu_t(), softmax_cross_entropy_t()
    net = [l1, l2, l3]

    # Create iterators for dataloaders
    train_iter = iter(train_dataloader)
    val_iter = iter(val_dataloader)

    # train for at least 10,000 iterations
    for t in range(10000):
        # 1. sample a mini-batch of size bb = 32
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            x, y = next(train_iter)

        # Convert to numpy and flatten x
        x = x.numpy().reshape(x.shape[0], -1)  # (batch_size, 784)
        y = y.numpy()

        # 2. zero gradient buffer
        for l in net:
            l.zero_grad()

        # 3. forward pass
        h1 = l1.forward(x)
        h2 = l2.forward(h1)
        ell, error = l3.forward(h2, y)

        # 4. backward pass
        dh2 = l3.backward()
        dh1 = l2.backward(dh2)
        dx = l1.backward(dh1)

        # 5. gather backprop gradients
        dw, db = l1.dw, l1.db

        # 6. print some quantities for logging and debugging
        if t % 1000 == 0:
            print(t, ell, error)

        train_loss_list.append(ell)
        train_error_list.append(error)

        # 7. one step of SGD
        l1.w = l1.w - lr * dw
        l1.b = l1.b - lr * db

        # 8. Validate every 1000 iterations
        if t % 1000 == 0:
            avg_val_loss, avg_val_error = validate_self_nn(l1, l2, l3, val_dataloader)
            val_loss_list.append(avg_val_loss)
            val_error_list.append(avg_val_error)
            print(f"Validation at iteration {t}: Loss={avg_val_loss:.4f}, Error={avg_val_error:.4f}")

    # Save training loss and error values for subproblem f
    with open('self_NN_training_loss.pkl', 'wb') as f:
        pickle.dump(train_loss_list, f)

    with open('self_NN_training_error.pkl', 'wb') as f:
        pickle.dump(train_error_list, f)

    # Save validation loss and error values for suproblem g
    with open('self_NN_validation_loss.pkl', 'wb') as f:
        pickle.dump(val_loss_list, f)

    with open('self_NN_validation_error.pkl', 'wb') as f:
        pickle.dump(val_error_list, f)

    # Save the trained model
    l1.save('linear.h5')

    return train_loss_list, train_error_list, val_loss_list, val_error_list


def validate_self_nn(l1, l2, l3, val_dataloader):
    """
    Complete for subproblem 3g

    Helper function to get validation loss and error of the model
    To be called once every 1,000 weight updates in train_self_nn()

    Iterate over all mini - batches from the validation dataset
    note that this should not be done randomly , we want to check
    every image only once

    Params:
        l1: linear_t()
        l2: relu_t()
        l3: softmax_cross_entropy_t()
        val_dataloader: validation dataloader

    Return: the average validation loss and error per batch over all batches in validation dataset as floats
    """

    ## TODO ##
    val_loss, val_error = 0, 0

    for x_val, y_val in val_dataloader:
        # Convert to numpy and flatten x
        x_val = x_val.numpy().reshape(x_val.shape[0], -1)
        y_val = y_val.numpy()

        # compute forward pass and error
        h1 = l1.forward(x_val)
        h2 = l2.forward(h1)
        ell, error = l3.forward(h2, y_val)
        val_loss += ell
        val_error += error

    avg_val_loss = val_loss / len(val_dataloader)
    avg_val_error = val_error / len(val_dataloader)

    return avg_val_loss, avg_val_error


class Net(nn.Module):
    """
    Complete for subproblem 3h

    Build a neural network using PyTorch with the same layers as
    in parts (b)-(g)
    """
    def __init__(self):
        super(Net, self).__init__()
        # Linear layer: 784 input features -> 10 output classes
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        # Flatten input if needed
        x = x.view(x.size(0), -1)  # (batch_size, 784)
        # Linear layer
        x = self.fc1(x)
        # ReLU activation
        x = F.relu(x)
        return x


def train_pytorch_nn(train_dataloader, val_dataloader, lr=0.1):
    """
    Complete for subproblem 3h

    Train a neural network using PyTorch for at least 10,000 weight updates

    For every 1,000 iterations, call validate_pytorch_nn() to get the
    validation loss and error

    Save the training and validation loss and error
    """
    setup(20)

    train_error_list = []
    train_loss_list = []
    val_error_list = []
    val_loss_list = []

    ###TODO###
    # Initialize network, criterion, and optimizer
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    # Create iterator for dataloader
    train_iter = iter(train_dataloader)

    # Train for at least 10,000 iterations
    for t in range(10000):
        # Sample a mini-batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            x, y = next(train_iter)

        # Convert to float32
        x = x.float()
        y = y.long()

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(x)
        loss = criterion(outputs, y)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Compute error
        _, predicted = torch.max(outputs.data, 1)
        error = (predicted != y).float().mean().item()

        # Log training metrics
        train_loss_list.append(loss.item())
        train_error_list.append(error)

        if t % 1000 == 0:
            print(f"Iteration {t}: Loss={loss.item():.4f}, Error={error:.4f}")

        # Validate every 1000 iterations
        if t % 1000 == 0:
            avg_val_loss, avg_val_error = validate_pytorch_nn(net, criterion, val_dataloader)
            val_loss_list.append(avg_val_loss)
            val_error_list.append(avg_val_error)
            print(f"Validation at iteration {t}: Loss={avg_val_loss:.4f}, Error={avg_val_error:.4f}")

    # Save training and validation loss and error as pickle files
    with open('pytorch_NN_training_loss.pkl', 'wb') as f:
        pickle.dump(train_loss_list, f)

    with open('pytorch_NN_training_error.pkl', 'wb') as f:
        pickle.dump(train_error_list, f)

    with open('pytorch_NN_validation_loss.pkl', 'wb') as f:
        pickle.dump(val_loss_list, f)

    with open('pytorch_NN_validation_error.pkl', 'wb') as f:
        pickle.dump(val_error_list, f)

    ## --- Model name here is "net". MAKE SURE TO RENAME THE `net.state_dict()` TO `"your_model_name".state_dict()` IF YOU EVER CHANGE THE MODEL NAME/LABEL!
    torch.save(net.state_dict(), 'pytorch_nn_weights.pth')


    return train_loss_list, train_error_list, val_loss_list, val_error_list

def validate_pytorch_nn(nn, criterion, val_dataloader):
    """
    Complete for subproblem 3h

    Helper function to get validation loss and error of the model
    To be called once every 1,000 weight updates in train_pytorch_nn()

    Iterate over all mini - batches from the validation dataset
    note that this should not be done randomly , we want to check
    every image only once

    Params:
        nn: pytorch neural network model initialized in train_pytorch_nn()
        criterion: criterion initialized in train_pytorch_nn()
        val_dataloader: validation dataloader

    Return: the average validation loss and error per batch over all batches in validation dataset as floats
    """

    ## TODO ##

    val_loss, val_error = 0, 0

    nn.eval()  # Set to evaluation mode

    for x_val, y_val in val_dataloader:
        # Convert to float32
        x_val = x_val.float()
        y_val = y_val.long()

        # compute forward pass and error
        with torch.no_grad():
            outputs = nn(x_val)
            loss = criterion(outputs, y_val)
            val_loss += loss.item()
            # average error for this batch
            _, predicted = torch.max(outputs.data, 1)
            error = (predicted != y_val).float().mean().item()
            val_error += error

    avg_val_loss = val_loss / len(val_dataloader)
    avg_val_error = val_error / len(val_dataloader)

    nn.train()  # Set back to training mode

    return avg_val_loss, avg_val_error


if __name__ == '__main__':
    ### TODO: Use the get_data() function to download the MNIST train and val data
    mnist_train, mnist_val = get_data()

    ### TODO: Call the subsample function to obtain the subsampled dataset and labels
    # Extract data from MNIST dataset
    trainX = mnist_train.data.numpy().reshape(-1, 784)
    trainY = mnist_train.targets.numpy()
    trainX, trainY = subsample(trainX, trainY, num_samples=10000, num_classes=10)

    ### TODO: Generate the train dataset
    train_dataset = MNISTDataset(trainX, trainY)

    ### TODO: Develop the train dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    ### TODO: call the subsample function on the validation dataset also
    valX = mnist_val.data.numpy().reshape(-1, 784)
    valY = mnist_val.targets.numpy()
    valX, valY = subsample(valX, valY, num_samples=2000, num_classes=10)

    ### TODO: Generate the val dataset
    val_dataset = MNISTDataset(valX, valY)

    ### TODO: Develop the val dataloader
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    ### TODO: Call the train_self_nn function to get the train loss and train error and val loss and val error and plot them
    print("\n" + "="*80)
    print("Training Self-Implemented Neural Network")
    print("="*80)
    train_loss_list, train_error_list, val_loss_list, val_error_list = train_self_nn(train_dataloader, val_dataloader, lr=0.1)

    ### TODO: Call the train_pytorch_nn to get the train loss and train error and val loss and val error for the pytorch implementation and plot them
    print("\n" + "="*80)
    print("Training PyTorch Neural Network")
    print("="*80)
    train_loss_list_pt, train_error_list_pt, val_loss_list_pt, val_error_list_pt = train_pytorch_nn(train_dataloader, val_dataloader, lr=0.1)

    ### TODO: call the test_backward() helper method for checking all backward methods
    print("\n" + "="*80)
    print("Testing Backward Methods")
    print("="*80)
    test_backward()
    print("All backward checks passed!")                                        