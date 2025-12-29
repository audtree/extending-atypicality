import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes

def split_and_scale_data(X, y, test_size, calib_size, random_seed):
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    
    # Further split train into fit and calibration
    X_fit, X_calib, y_fit, y_calib = train_test_split(X_train, y_train, test_size=calib_size, random_state=random_seed)
    
    # Scale features
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_fit)
    X_calib = scaler.transform(X_calib)
    X_test = scaler.transform(X_test)

    return X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler

def generate_and_split_gaussian_data(random_seed, test_size=0.2, calib_size=0.2, noise_std=0.5, n_samples=5000):
    """
    Corresponding atypicality score: Log Joint MVN
    """
    np.random.seed(random_seed)
    
    # Define mean and randomly generate a symmetric positive semi-definite covariance matrix
    mean = np.zeros(5)  # Mean vector
    random_matrix = np.random.rand(5, 5)  # Generate random values
    symmetric_matrix = (random_matrix + random_matrix.T) / 2  # Make it symmetric
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-6)  # Set a small positive lower bound for eigenvalues
    cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Sample from the multivariate Gaussian distribution
    data = np.random.multivariate_normal(mean, cov, size=n_samples)
    X, y = data[:, :-1], data[:, -1]  # X: first 4 columns, y: last column
    
    # Add Gaussian noise to y
    y += np.random.normal(0, noise_std, size=y.shape)

    X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler = split_and_scale_data(X, y, test_size, calib_size, random_seed)
    
    return X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler

def generate_and_split_lognormal_data(random_seed, test_size=0.2, calib_size=0.2, noise_std=0.9, n_samples=5000):
    """
    Corresponding atypicality score: lognormal_score.
    Generates only positive X values.
    """
    np.random.seed(random_seed)

    # Define mean and covariance for the latent normal distribution
    mean = np.zeros(5)  # Mean vector (all zeros for simplicity)
    random_matrix = np.random.rand(5, 5)  # Generate random values
    cov = (random_matrix + random_matrix.T) / 2  # Make it symmetric
    np.fill_diagonal(cov, 1.0)  # Ensure diagonal values are 1.0 for variance

    # Generate latent normal features
    X_normal = np.random.multivariate_normal(mean, cov, size=n_samples)

    # Ensure no zero values in the log-normal transformation
    X_normal_clipped = np.clip(X_normal, a_min=-10, a_max=None)  # Clip to avoid very large negative values

    # Transform to Log-Normal (exp function ensures strictly positive values)
    X = np.exp(X_normal_clipped)
    assert np.all(X > 0), "Error: X contains non-positive values"

    # Generate target variable y as a weighted sum of informative features + noise
    weights = np.array([2.0, 1.5, 0.5, 0.0, 0.1])  # Only first two are informative
    y = X @ weights + np.random.normal(0, noise_std, size=X.shape[0])  # Add Gaussian noise

    # print("Proportion of values that are flipped:", (y < 0).mean())

    # y = np.abs(y) # Correct any negative y values
    neg = y < 0
    while np.any(neg):
        y[neg] = (X[neg] @ weights
                + np.random.normal(0, noise_std, size=neg.sum()))
        neg = y < 0

    # Split into train, test, calib
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    X_fit, X_calib, y_fit, y_calib = train_test_split(X_train, y_train, test_size=calib_size, random_state=random_seed)

    # temp = 0

    # return X_fit, X_calib, X_test, y_fit, y_calib, y_test, temp

    # Scale features (makes standard deviation 1)
    scaler = StandardScaler(with_mean=False)
    X_fit = scaler.fit_transform(X_fit)
    X_calib = scaler.transform(X_calib)
    X_test = scaler.transform(X_test)

    assert np.all(X_train > 0), "Error: X_train contains non-positive values"
    assert np.all(X_test > 0), "Error: X_test contains non-positive values"
    assert np.all(X_fit > 0), "Error: X_test contains non-positive values"

    return X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler

def generate_and_split_gmm_data(random_seed, test_size=0.2, calib_size=0.2, n_components=3, n_features=4, n_samples=5000):
    """
    Corresponding atypicality score: gmm_score. 
    """
    np.random.seed(random_seed)

    n_samples = n_samples
    means = np.random.uniform(-5, 5, size=(n_components, n_features))
    covariances = np.array([np.random.rand(n_features, n_features) for _ in range(n_components)])
    mixing_proportions = np.random.dirichlet(np.ones(n_components), size=1).flatten()

    # Define and fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.weights_ = mixing_proportions
    
    # Generate X from the GMM
    X, labels = gmm.sample(n_samples)

    # Add gaussian noise
    beta = np.random.uniform(-2, 2, size=n_features)  # Random coefficients for the linear combination
    noise = np.random.normal(0, 1, size=n_samples)  # Gaussian noise

    # Linear combination: y = X * beta + noise
    y = np.dot(X, beta) + noise

    X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler = split_and_scale_data(X, y, test_size, calib_size, random_seed)

    return X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler

# Load California Housing Data
def load_and_split_chd_data(random_seed, test_size=0.2, calib_size=0.2, n_samples=5000):
    # Load California Housing dataset
    X, y = fetch_california_housing(return_X_y=True)

    # Take a random subset of the dataset
    X, _, y, _ = train_test_split(X, y, train_size=n_samples, random_state=random_seed)

    X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler = split_and_scale_data(X, y, test_size, calib_size, random_seed)

    return X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler

def load_wine_quality_data_csv(file_path):
    # Load Wine Quality dataset from a CSV file
    df = pd.read_csv(file_path, sep=";")
    
    # Split into features (X) and target (y)
    X = df.drop("quality", axis=1).values
    y = df["quality"].values
    
    return X, y

def load_and_split_wine_data(random_seed, test_size=0.2, calib_size=0.2):
    # Load Wine dataset
    X, y = load_wine_quality_data_csv("~/Downloads/wine+quality/winequality-white.csv")

    # Take a random subset of the dataset
    X, _, y, _ = train_test_split(X, y, random_state=random_seed)

    X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler = split_and_scale_data(X, y, test_size, calib_size, random_seed)

    return X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler

def load_and_split_diabetes_data(random_seed, test_size=0.2, calib_size=0.2):
    # Load Diabetes dataset
    X, y = load_diabetes(return_X_y=True)

    # Take a random subset of the dataset
    X, _, y, _ = train_test_split(X, y, random_state=random_seed)

    X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler = split_and_scale_data(X, y, test_size, calib_size, random_seed)

    return X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler