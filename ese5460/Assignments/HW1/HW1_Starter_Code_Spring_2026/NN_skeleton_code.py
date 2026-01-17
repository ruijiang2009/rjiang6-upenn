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
    ...


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
    ...

class MNISTDataset(Dataset):
    """
        Complete for subproblem 3a

        Define a dataset class to initialize Dataloader objects
    """
    ###TODO###
    def __init__(self, x_data, y_data):
        self.x_data = ...
        self.y_data = ...

    def __len__(self):
        return ...

    def __getitem__(self, idx):
        x = ...
        y = ...
        return ...


class linear_t:
    """
    Complete for subproblem 3b

    Define the linear layer class for the neural network
    """

    def __init__(self):
        """
        initialize to appropriate sizes, fill with Gaussian entires
        """
        self.w, self.b = ...

    def forward(self, hm):
        """
        hm: activations of a previous layer
        h: return activations of linear layer
        """
        h = ...
        self.hm = ...
        return h

    def backward(self, dh):
        """
        dh: dl/dh created by the next layer
        """
        dhm = ...
        self.dw = ...
        self.db = ...
        return dhm

    # notice that there is no need to cache dh^l return dh^l
    def zero_grad(self):
        """
        useful to delete the stored backprop gradients of the
        previous mini-batch before you start a new mini-batch
        """
        self.dw = 0*self.dw
        self.db = ...

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
            # TODO: true dw
            dw = ...

            ag_dict['k'].append(k)
            ag_dict['hm'].append(hm)
            ag_dict['dw'].append(dw)

            for _ in range(100):
                e = np.zeros((10, 784));
                i, j = np.random.randint(0, 10), np.random.randint(0, 784)
                e[i, j] = np.random.randn()
                # TODO: estimated dw
                dw_e = ...
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
            db = ...

            # TODO: estimated db
            db_e = np.zeros((10,1))
            db_e[k]=...
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
            dhm = ...

            ag_dict['k'].append(k)
            ag_dict['hm'].append(hm)
            ag_dict['dhm'].append(dhm)

            # dhm is similar to dw
            for _ in range(100):
                e = np.zeros((1, 784))
                i =np.random.randint(0,784)
                e[0,i]=np.random.randn()
                # TODO: estimated dhm
                dhm_e = ...
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
        ...

    def forward(self,hm):
        ...

    def backward(self,dh):
        ...

    def zero_grad(self):
        ...


class softmax_cross_entropy_t:
    """
    Complete for subproblem 3d
    """

    def __init__(self):
        self.y, self.prob=...

    def forward(self, h, y):

        ...

        # compute the average loss over mini-batch
        ce = np.mean(...)
        err = np.mean(...)
        return ce, err

    def backward(self):
        """
        As we saw in the notes, the backprop input to the
        loss layer is 1, so this function does not take any
        arguments
        """
        return ...

    def zero_grad(self):
        self.y, self.prob = ...

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

    # train for at least 10,000 iterations
    for t in range(...):
        # 1. sample a mini-batch of size bb = 32
        x, y = ...
        x_val, y_val = ...

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

        # 6. print some quantities for logging
        # and debugging
        print(t, ell, error)

        train_loss_list.append(ell)
        train_error_list.append(error)

        # 7. one step of SGD
        l1.w = l1.w - lr * dw
        l1.b = l1.b - lr * db

        h1 = l1.forward(x_val)
        h2 = l2.forward(h1)
        val_loss, val_error = l3.forward(h2, y_val)

        val_loss_list.append(val_loss)
        val_error_list.append(val_error)

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
        # compute forward pass and error
        ...
        ell, error = ....
        val_loss += ell
        val_error += error

    avg_val_loss = val_loss / len(val_dataloader)
    ...

    return avg_val_loss, avg_val_error


class Net(nn.Module):
    """
    Complete for subproblem 3h

    Build a neural network using PyTorch with the same layers as
    in parts (b)-(g)
    """
    def __init__(self):
        super(Net, self).__init__()
        ...

    def forward(self, x):
        ...


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
    ...

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

    for x_val, y_val in val_dataloader:
        # compute forward pass and error
        with torch.no_grad():
            ...
            val_loss += ...
            # average error for this batch
            val_error += ...
    avg_val_loss = val_loss / len(val_dataloader)
    ...

    return avg_val_loss, avg_val_error


if __name__ == '__main__':
    ### TODO: Use the get_data() function to download the MNIST train and val data
    mnist_train, mnist_val = ...
    ### TODO: Call the subsample function to obtain the subsampled dataset and labels
    trainX, trainY = ...
    ### TODO: Generate the train dataset
    train_dataset = ...
    ### TODO: Develop the train dataloader 
    train_dataloader = ...
    ### TODO: call the subsample function on the validation dataset also 
    valX, valY = ...
    ### TODO: Generate the val dataset 
    val_dataset = ...
    ### TODO: Develop the val dataloader 
    val_dataloader = ...

    ### TODO: Call the train_self_nn function to get the train loss and train error and val loss and val error and plot them
    train_loss_list, train_error_list, val_loss_list, val_error_list = ...

    ### TODO: Call the train_pytorch_nn to get the train loss and train error and val loss and val error for the pytorch implementation and plot them
    train_loss_list, train_error_list, val_loss_list, val_error_list = ...

    ### TODO: call the test_backward() helper method for checking all backward methods