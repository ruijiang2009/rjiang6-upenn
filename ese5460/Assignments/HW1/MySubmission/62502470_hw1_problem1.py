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
    ds = fetch_openml('mnist_784')
    ###TODO: Get the MNIST image data and target, convert to numpy and cast the data into either float or integer! 
    x, y = ds.data.to_numpy().astype(float), ds.target.to_numpy().astype(int)
    ### TODO: Plot one of the images with plt.imshow
    plt.imshow(x[0].reshape(28, 28), cmap='gray')
    plt.show()
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
    d_sampled = []
    for i in range(x.shape[0]):
        img = x[i].reshape(28, 28)
        resized_img = cv2.resize(img, size)
        d_sampled.append(resized_img.flatten())
    d_sampled = np.array(d_sampled)

    # Plot one resized image to check
    plt.imshow(d_sampled[0].reshape(size), cmap='gray')
    plt.show()

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
    # samples_per_class = num // 10  # 1000 samples per digit
    samples_per_class = 1000
    x_small = []
    y_small = []

    for digit in range(10):
        indices = np.where(y == digit)[0] # Get all indices for this digit
        # Randomly select samples_per_class indices
        selected_indices = np.random.choice(indices, samples_per_class, replace=False)
        x_small.append(x[selected_indices])
        y_small.append(y[selected_indices])

    x_small = np.vstack(x_small)
    y_small = np.hstack(y_small)

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
    x, y = subsample(x, y)
    x = resize(x)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
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
    return len(clf.support_vectors_) 


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
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_val)
    val_acc = clf.score(x_val, y_val)
    cm = confusion_matrix(y_val, y_pred)

    # Calculate support vector ratio
    num_support_vectors = get_number_of_support_samples(clf)
    supp_vector_percent = (num_support_vectors / len(x_train)) * 100

    # Print results
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Error: {1 - val_acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Support Vector Ratio: {supp_vector_percent:.2f}%")

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
    clf = GridSearchCV(svm.SVC(), params, cv=5, verbose=1)
    clf.fit(x_train, y_train)
    
    # Get best values
    print(f"Best accuracy for training data: {clf.best_score_:.4f}")
    print(f"Best estimators: {clf.best_params_}")
    print(f"All cross-validation results: {clf.cv_results_['mean_test_score']}")
    
    # Get predicted y_val using best classifier
    y_pred = clf.best_estimator_.predict(x_val)
    
    # Print confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print(f"Confusion Matrix:\n{cm}")

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

    gabor_real_coeffs = []
    for img_idx in tqdm(range(images.shape[0])):
        ### TODO ###
        img = images[img_idx].reshape(14, 14)
        img_coeffs = []
        
        # Apply each filter combination
        for freq in frequency_list:
            for theta in theta_list:
                for bandwidth in bandwidth_list:
                    # Apply gabor filter and get real coefficients
                    real, _ = gabor(img, frequency=freq, theta=theta, bandwidth=bandwidth)
                    img_coeffs.append(real.flatten())
        
        # Concatenate all filter responses for this image
        gabor_real_coeffs.append(np.concatenate(img_coeffs))
    
    gabor_real_coeffs = np.array(gabor_real_coeffs)

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
    x_train_small, y_train_small = subsample(x_train, y_train, num=1000)
    x_val_small, y_val_small = subsample(x_val, y_val, num=1000)

    # 2. Check gabor kernel and gabor outputs as example
    ### TODO ###
    kernel = gabor_kernel(frequency=0.1, theta=0, bandwidth=1.0)
    sample_img = x_train_small[0].reshape(14, 14)
    real, imag = gabor(sample_img, frequency=0.1, theta=0, bandwidth=1.0)
    
    # Plot example
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(sample_img, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(real, cmap='gray')
    plt.title('Gabor Real')
    plt.subplot(1, 3, 3)
    plt.imshow(imag, cmap='gray')
    plt.title('Gabor Imaginary')
    plt.show()

    # 3. Diversify the values for frequency, theta, and bandwidth parameters
    #    Plot 8 pairs of real and imaginary coefficients from gabor_kernel()
    #    to check diversity of this filter-bank
    ### TODO ###
    frequency_list = [0.1, 0.2]
    theta_list = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    bandwidth_list = [1.0, 2.0]
    
    plt.figure(figsize=(16, 8))
    for i, (freq, theta) in enumerate([(f, t) for f in frequency_list for t in theta_list[:4]]):
        kernel = gabor_kernel(frequency=freq, theta=theta, bandwidth=1.0)
        plt.subplot(2, 8, 2*i + 1)
        plt.imshow(np.real(kernel), cmap='gray')
        plt.title(f'f={freq}, θ={theta:.2f}\nReal')
        plt.axis('off')
        plt.subplot(2, 8, 2*i + 2)
        plt.imshow(np.imag(kernel), cmap='gray')
        plt.title(f'Imaginary')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 4. Obtain new training and validation datasets of real coefficients
    #    by calling apply_gfilter() function
    ### TODO ###
    print("Applying Gabor filters to training data...")
    x_train_gabor = apply_gfilter(x_train_small, frequency_list, theta_list, bandwidth_list)
    print("Applying Gabor filters to validation data...")
    x_val_gabor = apply_gfilter(x_val_small, frequency_list, theta_list, bandwidth_list)

    # 5. Standardize and then PCA features with StandardScaler() and PCA() functions
    ### TODO ###
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_gabor)
    x_val_scaled = scaler.transform(x_val_gabor)
    
    pca = PCA(n_components=50)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_val_pca = pca.transform(x_val_scaled)
    
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")

    # 6. Find best svm.SVC classifier
    #    Hint: look into cache_size parameter of svm.SVC() to speed up training
    ### TODO ###
    classifier = svm.SVC(kernel='rbf', C=1.0, gamma='scale', cache_size=1000)
    classifier.fit(x_train_pca, y_train_small)
    
    val_acc = classifier.score(x_val_pca, y_val_small)
    print(f"Gabor Filter SVM Validation Accuracy: {val_acc:.4f}")

    return classifier



if __name__ == '__main__':
    ### TODO: get downloaded MNIST data and target
    x, y = get_data()

    ### TODO: preprocess the data
    x_train, x_val, y_train, y_val = data_preprocessing(x, y)

    train_test_SVM(x_train, y_train, x_val, y_val)

    ### TODO: define SVM hyperparameters to perform grid search with
    params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    ### TODO: Save the best svm classifier returned by grid_search_SVM to pickle file
    best_svm = grid_search_SVM(x_train, y_train, x_val, y_val, params)
    with open('grid_search_svm.pkl', 'wb') as f:
        pickle.dump(best_svm, f)

    ### TODO: gabor_filter() and apply_gfilter()
    gabor_classifier = gabor_filter(x_train, y_train, x_val, y_val)