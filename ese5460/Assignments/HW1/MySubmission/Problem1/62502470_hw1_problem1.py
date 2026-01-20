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
    ds = fetch_openml('mnist_784', as_frame=False)
    ###TODO: Get the MNIST image data and target, convert to numpy and cast the data into either float or integer! 
    x = ds.data.astype(float)
    y = ds.target.astype(int)
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
    samples_per_class = 1000  # 1000 samples per digit as per requirements
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

# added by me
def subsample_per_class(x, y, samples_per_class=100):
    x_small = []
    y_small = []

    for digit in range(10):
        indices = np.where(y == digit)[0]
        selected_indices = np.random.choice(
            indices, samples_per_class, replace=False
        )
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
    clf = GridSearchCV(svm.SVC(), params, cv=5)
    clf.fit(x_train, y_train)

    # Get best values
    best_score = clf.best_score_
    best_params = clf.best_params_
    cv_results = clf.cv_results_['mean_test_score']

    # Print grid search results
    print(f"Best CV Score: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")
    print("All CV Results:")
    for i, score in enumerate(cv_results):
        print(f"  {clf.cv_results_['params'][i]}: {score:.4f}")

    # Get predicted y_val using best classifier
    y_pred = clf.best_estimator_.predict(x_val)

    # Print confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix:")
    print(cm)

    return clf.best_estimator_

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


def show_gabor_example(x_train):
    """
    Demonstrate gabor_kernel and gabor filter on example images.
    Shows the kernel (real/imaginary), original image, and filtered image coefficients.
    """
    # Set parameters
    freq, theta, bandwidth = 0.1, np.pi/4, 1

    # Create gabor kernel
    gk = gabor_kernel(frequency=freq, theta=theta, bandwidth=bandwidth)
    plt.figure(1); plt.clf(); plt.imshow(gk.real)
    plt.figure(2); plt.clf(); plt.imshow(gk.imag)
    
    # Get original image
    image = x_train[0].reshape((14, 14))

    # Convolve the input image with the kernel and get co-efficients
    # We will use only the real part and throw away the imaginary part
    coeff_real, _ = gabor(image, frequency=freq, theta=theta, bandwidth=bandwidth)
    plt. figure(3); plt.clf (); plt.imshow(coeff_real)
    plt. figure(4); plt.clf (); plt.imshow(image)
    plt.show()


def show_gabor_exampl_v1(x_train, y_train, idx=0):
    """
    Demonstrate gabor_kernel and gabor filter on example images.
    Shows the kernel (real/imaginary), original image, and filtered image coefficients.
    Each image is show in 4 windows
    """
    freq, theta, bandwidth = 0.1, np.pi/4, 1

    # Create gabor kernel
    gk = gabor_kernel(frequency=freq, theta=theta, bandwidth=bandwidth)

    plt.figure(1); plt.clf()
    plt.imshow(gk.real, cmap="gray")
    plt.title("Gabor kernel (real)")
    plt.axis("off")

    plt.figure(2); plt.clf()
    plt.imshow(gk.imag, cmap="gray")
    plt.title("Gabor kernel (imag)")
    plt.axis("off")

    # Get original image
    image = x_train[idx].reshape((14, 14))
    label = y_train[idx]

    # Apply Gabor filter
    coeff_real, _ = gabor(image, frequency=freq, theta=theta, bandwidth=bandwidth)

    plt.figure(3); plt.clf()
    plt.imshow(coeff_real, cmap="gray")
    plt.title(f"Filtered image (label = {label})")
    plt.axis("off")

    plt.figure(4); plt.clf()
    plt.imshow(image, cmap="gray")
    plt.title(f"Original image (label = {label})")
    plt.axis("off")

    plt.show()


def show_gabor_example_v2(x_train, y_train, idx=0):
    """
    Docstring for show_gabor_example_v2
    Show all 4 images in one plot.
    """
    freq, theta, bandwidth = 0.1, np.pi/4, 1

    # Create gabor kernel
    gk = gabor_kernel(frequency=freq, theta=theta, bandwidth=bandwidth)

    # Get original image and label
    image = x_train[idx].reshape((14, 14))
    label = y_train[idx]

    # Apply Gabor filter
    coeff_real, _ = gabor(image, frequency=freq, theta=theta, bandwidth=bandwidth)

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    axs[0, 0].imshow(image, cmap="gray")
    axs[0, 0].set_title(f"Original image (label = {label})")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(coeff_real, cmap="gray")
    axs[0, 1].set_title(f"Filtered image (label = {label})")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(gk.real, cmap="gray")
    axs[1, 0].set_title("Gabor kernel (real)")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(gk.imag, cmap="gray")
    axs[1, 1].set_title("Gabor kernel (imaginary)")
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

# !!! need to use resized 14x14 images
def show_gabor_8_filter_images():
    theta_list = [np.pi/4, np.pi/2, 3 * np.pi / 4, np.pi]
    frequency_list = [0.05, 0.25]
    bandwidth_list = [0.1, 1]
    # Plot 8 pairs of gabor_kernel() real and imaginary coefficients
    print("Plotting 8 pairs of Gabor kernel real and imaginary coefficients...")
    plt.figure(figsize=(16, 4))
    plt.suptitle('Gabor Kernels: Real and Imaginary Components (8 pairs)')
    filter_params = [(f, t) for f in frequency_list for t in theta_list]  # 8 combinations
    for i, (freq, theta) in enumerate(filter_params):
        kernel = gabor_kernel(frequency=freq, theta=theta, bandwidth=1.0)
        # Top row: Real components
        plt.subplot(2, 8, i + 1)
        plt.imshow(np.real(kernel), cmap='gray')
        plt.title(f'f={freq}\nθ={theta:.2f}\nreal', fontsize=8)
        plt.axis('off')
        # Bottom row: Imaginary components
        plt.subplot(2, 8, i + 9)
        plt.imshow(np.imag(kernel), cmap='gray')
        plt.title(f'f={freq}\nθ={theta:.2f}\nimaginary', fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gabor_kernels_8pairs.png', dpi=150)
    plt.show()

# !!! need to use resized 14x14 images
# change the output to 4x4 to make the picture larger
def show_gabor_8_filter_imagesv2():
    theta_list = [np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    frequency_list = [0.05, 0.25]

    filter_params = [(f, t) for f in frequency_list for t in theta_list]  # 8 filters

    plt.figure(figsize=(12, 12))
    plt.suptitle('Gabor Kernels: Real and Imaginary Components (8 filters)', fontsize=16)

    for i, (freq, theta) in enumerate(filter_params):
        kernel = gabor_kernel(frequency=freq, theta=theta, bandwidth=1.0)

        row = i // 2
        col = (i % 2) * 2

        # Real part
        plt.subplot(4, 4, row * 4 + col + 1)
        plt.imshow(kernel.real, cmap='gray')
        plt.title(f'f={freq}, θ={theta:.2f}\nreal', fontsize=9)
        plt.axis('off')

        # Imaginary part
        plt.subplot(4, 4, row * 4 + col + 2)
        plt.imshow(kernel.imag, cmap='gray')
        plt.title(f'f={freq}, θ={theta:.2f}\nimag', fontsize=9)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('gabor_kernels_8filters_4x4.png', dpi=200)
    plt.show()


def gabor_filter(x_train, y_train, x_val, y_val):
    """
    Perform gabor filter experiment
    """

    # 1. Prepare a balanced 1,000 training and validation images
    #     of 100 images per class
    ### TODO ###
    # Combine train and val, then subsample to get balanced 1000 total
    x_combined = np.vstack([x_train, x_val])
    y_combined = np.hstack([y_train, y_val])

    # the old way of doing it.
    # x_gabor_data, y_gabor_data = subsample(x_combined, y_combined, num=1000)

    x_gabor_data, y_gabor_data = subsample_per_class(x_combined, y_combined, samples_per_class=100)

    # Split into train/val (500 each, 50 per class each)
    x_train_small, x_val_small, y_train_small, y_val_small = train_test_split(
        x_gabor_data, y_gabor_data, test_size=0.5, random_state=42, stratify=y_gabor_data
    )

    # 2. Check gabor kernel and gabor outputs as example
    ### TODO ###
    show_gabor_example_v2(x_train, y_train)

    # Sample image for later plots
    sample_img = x_train_small[0].reshape(14, 14)

    # 3. Diversify the values for frequency, theta, and bandwidth parameters
    #    Plot 8 pairs of real and imaginary coefficients from gabor_kernel()
    #    to check diversity of this filter-bank
    ### TODO ###
    theta_list = [np.pi/4, np.pi/2, 3 * np.pi / 4, np.pi]
    frequency_list = [0.05, 0.25]
    bandwidth_list = [0.1, 1]
    show_gabor_8_filter_imagesv2()

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
    params = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto']
    }
    grid_clf = GridSearchCV(svm.SVC(cache_size=8000), params, cv=5)
    grid_clf.fit(x_train_pca, y_train_small)

    print(f"Best parameters: {grid_clf.best_params_}")
    print(f"Best CV score: {grid_clf.best_score_:.4f}")

    classifier = grid_clf.best_estimator_

    val_acc = classifier.score(x_val_pca, y_val_small)
    val_error = 1 - val_acc

    # Print results
    print("\n" + "="*50)
    print("GABOR FILTER SVM RESULTS")
    print("="*50)
    print(f"Filter parameters:")
    print(f"  - Frequencies: {frequency_list}")
    print(f"  - Thetas: {[f'{t:.4f}' for t in theta_list]}")
    print(f"  - Bandwidths: {bandwidth_list}")
    print(f"  - Total filters: {len(frequency_list) * len(theta_list) * len(bandwidth_list)}")
    print(f"\nPCA components: 50")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
    print(f"\nBest SVM parameters: {grid_clf.best_params_}")
    print(f"\nValidation Accuracy: {val_acc:.4f}")
    print(f"Validation Error: {val_error:.4f}")
    print("="*50)

    return classifier



if __name__ == '__main__':
    ### TODO: get downloaded MNIST data and target
    x, y = get_data()

    ### TODO: preprocess the data
    x_train, x_val, y_train, y_val = data_preprocessing(x, y)

    train_test_SVM(x_train, y_train, x_val, y_val)

    # ### Show gabor filter example (for part 1h)
    # # show_gabor_example(x_train)
    # show_gabor_example_v2(x_train, y_train)
    # show_gabor_8_filter_imagesv2()

    ### TODO: define SVM hyperparameters to perform grid search with
    params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    # ### TODO: Save the best svm classifier returned by grid_search_SVM to pickle file
    best_svm = grid_search_SVM(x_train, y_train, x_val, y_val, params)
    with open('grid_search_svm.pkl', 'wb') as f:
        pickle.dump(best_svm, f)

    # show 8 gabor examples
    # show_gabor_8_filter_images()

    ### TODO: gabor_filter() and apply_gfilter()
    gabor_classifier = gabor_filter(x_train, y_train, x_val, y_val)
    with open('gabor_classifier.pkl', 'wb') as f:
        pickle.dump(gabor_classifier, f)