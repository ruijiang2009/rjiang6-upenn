#################### DO NOT EDIT THE GIVEN IMPORTS ##########################
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

import random
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
import cv2
import pickle
import numbers

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.filters import gabor_kernel, gabor
from tqdm import tqdm
import time
#################### AND DO NOT ADD NEW IMPORTS ##########################


def get_data():
    """
    Complete for subproblem 1a

    Get MNIST data from fetch_openml
    Get data and target converted to numpy
    Call plt.imshow on any random reshaped image to (28,28)
    to see what the digit looks like

    Return:
        x: MNIST image data in numpy
        y: MNIST image target in numpy
    """
    ###TODO: Download mnist 784 dataset, do not specify any parameters except for the dataset name
    #         to be written within the single quotes
    ds = fetch_openml('...')
    ###TODO: Get the MNIST image data and target, convert to numpy and cast the data into either float or integer! 
    x, y =
    ### TODO: Plot one of the images with plt.imshow
    ...
    return x, y


def resize(x, size=(14, 14)):
    """
    Complete for subproblem 1b

    The input arguments are:
    x: numpy array of MNIST data, where each image is of size (28 x 28)
    size(): new data size

    Resize data by calling cv2.resize() for each image from (28 x 28) to (14x14),
    flattened to 196.
    Call plt.imshow to check one of the resized data

    Return: the numpy array of resized, downsampled MNIST data of shape (num_of_images, 196)
    """
    ###TODO###
    ...
    return d_sampled


def subsample(x, y, num=10000):
    """
    Complete for subproblem 1c

    Create a dataset of 1000 samples for each digit (10,000 samples in total)

    The input arguments are:
    x: MNIST data
    y: MNIST target
    num: Total number of samples required after subsampling

    Return: x_small, y_small subsampled MNIST dataset containing 1,000 images of each digit

    """
    ### TODO
    ...
    return x_small, y_small


def data_preprocessing(x,y):
    """
    Complete for subproblem 1d

    x: numpy array of MNIST data
    y: numpy array of MNIST targets

    Sub-sample data using the provided subsample method, 
    to have equal number of data for each digit
    Resize data to (14,14)
    Get train and validation split with random state 42, test size 0.2 (since we want 20% as validation set), shuffled, and stratified

    Return: x_train, x_val, y_train, y_val
    """
    ###TODO###
    x, y = subsample(...)
    x = resize(...)
    x_train, x_val, y_train, y_val = train_test_split(..., ..., test_size=..., random_state=..., shuffle=..., stratify=...)
    return x_train, x_val, y_train, y_val


def get_number_of_support_samples(clf):
    """

    Complete for 1e

    The input argument is the SVC classifier

    Check the SVC documentation for how to get the total
    number of support vectors of the classifier
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    Return: Total number of support samples given the classifier
    """
    ###TODO###
    ...


def train_test_SVM(x_train, y_train, x_val, y_val):
    """

    Complete for 1f

    Train test an SVM model given training and validation data
    Print the classifier score and validation error, confusion matrix, and support sample ratio
    Call get_number_of_support_samples() to calculate the ratio of support samples
    (total number of support samples out of total data)

    Return: Validation Accuracy, Confusion Matrix, Support Vector Ratio(percentage) 
    """
    ###TODO###
    # initialize the classifier and get the classifier score
    clf = ...
    clf.fit(...)
    y_pred = clf.predict(...)
    val_acc = clf.score(...)
    cm = ...
    supp_vector_percent = ... 

    return val_acc, cm, supp_vector_percent


def grid_search_SVM(x_train, y_train, x_val, y_val, params):
    """

    Complete for 1g

    Perform grid search to find the best classifier
    Get the following values:
    - best accuracy for training data
    - best estimators of each parameter
    - accuracy of hyperparameters tried
    Look at the documentation for GridSearchCV to get the values above
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    Get the best classifier found by grid search to get predicted y_val

    Print the confusion matrix given the true y_val and predicted y_val

    Return: best classifier found by grid search
    """
    ###TODO###
    clf = GridSearchCV(...)
    clf.fit(...)
    ...

    return clf

def apply_gfilter(images, frequency_list, theta_list, bandwidth_list):
    """
    Complete for 1h
    IMPORTANT: This part is computationally intensive. It will take some time to run.!

    Helper function to be called in gabor_filter() to generate new datasets of convolved images

    For each image in the given list of images, obtain the real coefficients
    returned by gabor() function from each combination of frequency, theta, and bandwith parameters
    Remember to reshape each image to (14,14) to pass to gabor()

    params:
        images: list of mnist images to apply gabor filter to
        frequency_list: list of values for frequency parameter for gabor() function
        theta_list: list of values for theta parameter for gabor() function
        bandwidth_list: list of values for bandwidth parameter in gabor() function
    return:
        gabor_real_coeffs: np.array of size (num of images, num of filters * 14 * 14)
    """

    gabor_real_coeffs = ...
    for img in tqdm(range(images.shape[0])):
        ### TODO ###
        ....

    n_filters = len(frequency_list) * len(theta_list) * len(bandwidth_list)
    assert gabor_real_coeffs.shape[0] == len(images)
    assert gabor_real_coeffs.shape[1] == n_filters * images.shape[1]

    return gabor_real_coeffs

def gabor_filter(x_train, y_train, x_val, y_val):
    """
    Perform gabor filter experiment
    """

    # 1. Prepare a balanced 1,000 training and validation images
    #     of 100 images per class
    ### TODO ###

    # 2. Check gabor kernel and gabor outputs as example
    ### TODO ###


    # 3. Diversify the values for frequency, theta, and bandwidth parameters
    #    Plot 8 pairs of real and imaginary coefficients from gabor_kernel()
    #    to check diversity of this filter-bank
    ### TODO ###

    # 4. Obtain new training and validation datasets of real coefficients
    #    by calling apply_gfilter() function
    ### TODO ###

    # 5. Standardize and then PCA features with StandardScaler() and PCA() functions
    ### TODO ###

    # 6. Find best svm.SVC classifier
    #    Hint: look into cache_size parameter of svm.SVC() to speed up training
    ### TODO ###

    return classifier



if __name__ == '__main__':
    ### TODO: get downloaded MNIST data and target
    x, y = ...

    ### TODO: preprocess the data
    x_train, x_val, y_train, y_val = ...

    train_test_SVM(x_train, y_train, x_val, y_val)

    ### TODO: define SVM hyperparameters to perform grid search with
    params = {
        'C':[...],
              ...,
              ...
    }
    ### TODO: Save the best svm classifier returned by grid_search_SVM to pickle file
    svm = grid_search_SVM(...)
    with open('grid_search_svm.pkl', 'wb') as f:
        pickle.dump(svm, f)

    ### TODO: gabor_filter() and apply_gfilter()
    ...