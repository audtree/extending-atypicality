import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import scipy.stats as stats
from scipy.special import logsumexp

from sklearn.neighbors import NearestNeighbors
from scipy.stats import multivariate_normal, norm, multivariate_t, lognorm
from sklearn.linear_model import LinearRegression

from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf, ShrunkCovariance
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from tqdm import tqdm

def hash_dataset(dataset):
    """Create a unique hash for a dataset to use as a cache key."""
    dataset_tuple = tuple((tuple(x), y) for x, y in dataset)
    return hashlib.md5(str(dataset_tuple).encode()).hexdigest()

# KNN Atypicality Score
def knn_score(input_point, dataset, k=5):
  """
  Calculates a score based on the average difference between the input point and the
  its k nearest neighbors with similar outputs.

  Args:
    input_point: A tuple (x_i, y_i_hat) where x_i is a vector of predictors and y_i_hat is the predicted output.
    dataset: A list of tuples [(x_1, y_1), (x_2, y_2), ...] where x_i is a vector of predictors and y_i is the corresopnding output.
    k: The number of nearest neighbors to consider.

  Returns:
    The score, which is the difference between the input x and the average x
    of its k nearest neighbors with similar outputs.
  """
  x_i, y_i_hat = input_point
  X = [point[0] for point in dataset]
  y = [point[1] for point in dataset]

  # Find the k nearest neighbors based on y values
  neigh = NearestNeighbors(n_neighbors=k, metric='euclidean')
  neigh.fit(np.array(y).reshape(-1, 1))
  distances, indices = neigh.kneighbors(np.array(y_i_hat).reshape(1, -1))

  nearest_neighbors = [X[i] for i in indices[0]] # Does this do what I want it to do 

  # Compute the mean distance from x_q to the k-nearest neighbors in feature space
  score = np.mean([np.linalg.norm(np.array(x_i) - np.array(neighbor)) for neighbor in nearest_neighbors])

  return score

# KDE Atypicality Score
def gaussian_kernel(y_i, y_q_hat, bandwidth=1.0):
    return np.exp(-((y_i - y_q_hat) ** 2) / (2 * bandwidth ** 2))

def kde_score(input_point, dataset, kernel_function):
    """
    Calculate the KDE-based score.

    Parameters:
        input_point (tuple): A tuple (x_i, y_i_hat) where x_q is the feature vector and y_i_hat is the predicted output.
        dataset (list of tuples): A list [(x_1, y_1), (x_2, y_2), ...] where x_i is a feature vector and y_i is the output.
        kernel_function (callable): Function K(yi, y_i_hat) that returns the kernel weight.

    Returns:
        float: KDE-based score.
    """
    x_i, y_i_hat = input_point  # Extract x_q and y_q_hat from input_point

    # Extract feature vectors (X) and target values (y) from the dataset
    X = [point[0] for point in dataset] 
    y = [point[1] for point in dataset] 

    weights = np.array([kernel_function(yi, y_i_hat) for yi in y])
    if np.sum(weights) == 0:
        weights_normalized = np.zeros_like(weights)
    else:
        weights_normalized = weights / np.sum(weights)
    weighted_mean_distance = np.sum(
        [weights_normalized[i] * np.linalg.norm(np.array(x_i) - np.array(X[i])) for i in range(len(X))]
    )

    return weighted_mean_distance

# Log Joint MVN atypicality score
logjointmvn_cache = {}
def get_logjointmvn_params(dataset):
    """Fit a GMM and return its parameters."""
    # Extract feature vectors (X) and target values (y) from the dataset
    X = np.array([point[0] for point in dataset])  
    y = np.array([point[1] for point in dataset]) 

    # Estimate the parameters of the joint MVN
    mu_X = np.mean(X, axis=0)  
    mu_Y = np.mean(y) 
    
    # print("Means (mu_X, mu_Y):", mu_X, mu_Y)

    # Covariance Matrices
    shrunk = ShrunkCovariance()
    shrunk.fit(X) 
    Sigma_XX = shrunk.covariance_
    Sigma_YY = np.array([[np.var(y, ddof=1)]])          # Variance of Y (scalar, since Y is 1D)
    Sigma_XY = np.cov(X.T, y, rowvar=True)[:-1, -1].reshape(-1, 1)  # Covariance between X and Y

    # Compute joint probability P(X = x_i, Y = y_i_hat)
    # Define the joint mean and covariance matrix for the multivariate normal
    mu_joint = np.hstack((mu_X, mu_Y))
    Sigma_joint = np.block([[Sigma_XX, Sigma_XY],
                            [Sigma_XY.T, Sigma_YY]])
    
    # Regularize the covariance matrix (e.g., adding a small value to the diagonal)
    epsilon = 1e-6
    Sigma_joint += epsilon * np.eye(Sigma_joint.shape[0])  # Add regularization to prevent singular matrix

    # Create the multivariate normal distribution
    # Eigenvalue Clipping to ensure that our covariance matrix is positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(Sigma_joint)  # Perform eigen decomposition
    eigvals[eigvals < epsilon] = epsilon    # Clip small eigenvalues to a small positive number
    Sigma_joint = eigvecs @ np.diag(eigvals) @ eigvecs.T    # Reconstruct the covariance matrix

    rv_joint = multivariate_normal(mean=mu_joint, cov=Sigma_joint, allow_singular=True) # TODO: check whether you can allow singular 
    rv_Y = norm(loc=mu_Y, scale=np.sqrt(Sigma_YY))

    return rv_joint, rv_Y

def logjointmvn_score(input_point, dataset):
    """
    Calculate the Negative Log Joint MVN Atypicality Score.

    Parameters:
        input_point (tuple): A tuple (x_i, y_i_hat) where x_i is the feature vector and y_i_hat is the predicted output.
        dataset (list of tuples): A list [(x_1, y_1), (x_2, y_2), ...] where x_i is a feature vector and y_i is the output.

    Returns:
        float: Negative Log Joint MVN atypicality score.
    """
    x_i, y_i_hat = input_point 
    y_i_hat = np.array(y_i_hat).reshape(-1) # Ensure y_i_hat is a 1D array (shape: (1,))

    # Load or compute Log Joint MVN parameters
    dataset_hash = hash_dataset(dataset)
    if dataset_hash not in logjointmvn_cache:
        logjointmvn_cache[dataset_hash] = get_logjointmvn_params(dataset)
    rv_joint, rv_Y = logjointmvn_cache[dataset_hash]

    # Compute log probability instead of probability
    EPSILON = 1e-12  # Small constant to prevent log(0)
    log_joint_density = rv_joint.logpdf(np.hstack((x_i, y_i_hat)))

    # Compute log marginal probability log P(Y = y_i_hat)
    log_marginal_density = rv_Y.logpdf(y_i_hat)

    # Compute log atypicality score
    log_atypicality_score = - (log_joint_density - log_marginal_density)

    # Truncate values if less than 0 (means the point is very typical)
    log_atypicality_score = max(log_atypicality_score.item(), 0.0)

    return log_atypicality_score

# Lognormal atypicality score
lognormal_cache = {}
def get_lognormal_params(dataset):
    """ Estimate weights, latent normal distribution parameters for X, and lognormal distribution parameters for Y."""
    # Extract feature vectors (X) and target values (y) from the dataset
    X = np.array([point[0] for point in dataset])
    y = np.array([point[1] for point in dataset])

    # Estimate sigma and mu_X of the latent normal distribution
    Z = np.log(X)  # Transform to normal space
    mu_Z = Z.mean(axis=0)

    lw = LedoitWolf()
    lw.fit(Z)
    Sigma_Z = lw.covariance_

    # Estimate weights of Y = Xw + error using least squares
    w = np.linalg.solve(X.T @ X, X.T @ y)

    # Estimate residual variance
    sigma_sq = np.mean((y - X @ w) ** 2)

    # Calculate parameters of estimated lognormal distribution of Y
    ylognorm_shape, ylognorm_loc, ylognorm_scale = lognorm.fit(y, floc=0)

    return mu_Z, Sigma_Z, w, sigma_sq, ylognorm_shape, ylognorm_loc, ylognorm_scale

def lognormal_score(input_point, dataset):
    """
    Calculate the LogNormal Atypicality Score, which is P(X|Y) where X is distributed LogNormal and Y is a linear function of X plus Gaussian Noise.

    Parameters:
        input_point (tuple): A tuple (x_i, y_i_hat) where x_i is the feature vector and y_i_hat is the predicted output.
        dataset (list of tuples): A list [(x_1, y_1), (x_2, y_2), ...] where x_i is a feature vector and y_i is the output.

    Returns:
        float: Negative Log LogNormal Atypicality Score.

    Note:
        Important to note that because we're working with continuous r.v.'s, we are not using P(X|Y).
        We're really using f(x), which is not bounded by [0,1] and can take on any positive real. Thus log(f(x|y)) can be greater than 1.
    """
    x_i, y_i_hat = input_point
    y_i_hat = np.array(y_i_hat).reshape(-1) # Ensure y_i_hat is a 1D array (shape: (1,))

    # Load or compute LogNormal parameters
    dataset_hash = hash_dataset(dataset)
    if dataset_hash not in lognormal_cache:
        lognormal_cache[dataset_hash] = get_lognormal_params(dataset)
    mu_Z, Sigma_Z, w, sigma_sq, ylognorm_shape, ylognorm_loc, ylognorm_scale = lognormal_cache[dataset_hash]

    # 1. Calcualte f_Y|X(y|x)
    # Compute conditional density
    log_py_given_x = norm.logpdf(
        y_i_hat,
        loc=x_i @ w,
        scale=np.sqrt(sigma_sq))

    # 2. Calculate f_X(x)
    # Convert x to normal space
    z_i = np.log(x_i)

    # Use MVN transformation
    log_pz = multivariate_normal.logpdf(
        z_i, mean=mu_Z, cov=Sigma_Z)

    # Use scaling with Jacobian
    log_jacobian = -np.sum(np.log(x_i))

    log_px = log_pz + log_jacobian

    # 3. Estimate f_Y(y):
    log_py = lognorm.logpdf(
        y_i_hat,
        ylognorm_shape,
        ylognorm_loc,
        ylognorm_scale)
    # Assemble bayes
    log_score = log_py_given_x + log_px - log_py

    return -log_score.item()

# GMM atypicality score
gmm_cache = {}
def get_gmm_params(dataset):
    """Fit a GMM and return its parameters."""
    # Extract feature vectors (X) and target values (y) from the dataset
    X = np.array([point[0] for point in dataset])  
    y = np.array([point[1] for point in dataset]) 

    # Estimate beta using linear regression
    model = LinearRegression()
    model.fit(X, y)
    beta_hat = model.coef_
    
    # Estimate sigma^2
    residuals = y - X @ beta_hat
    sigma_hat = max(np.std(residuals), 1e-9) # if the model is perfectly overfit, sigma_hat will be 0 (division by 0)
    
    # Fit a GMM to X
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42) 
    gmm.fit(X)

    return beta_hat, sigma_hat, gmm

def gmm_score(input_point, dataset):
    """
    Calculate the GMM Atypicality Score, which is P(X|Y) where X is distributed GMM and Y is a linear function of X plus Gaussian Noise. 

    Parameters:
        input_point (tuple): A tuple (x_i, y_i_hat) where x_i is the feature vector and y_i_hat is the predicted output.
        dataset (list of tuples): A list [(x_1, y_1), (x_2, y_2), ...] where x_i is a feature vector and y_i is the output.

    Returns:
        float: Negative Log GMM Atypicality Score.

    Note:
        Important to note that because we're working with continuous r.v.'s, we are not using P(X|Y). 
        We're really using f(x), which is not bounded by [0,1] and can take on any positive real. Thus log(f(x|y)) can be greater than 1. 
    """
    x_i, y_i_hat = input_point 
    y_i_hat = np.array(y_i_hat).reshape(-1) # Ensure y_i_hat is a 1D array (shape: (1,))

    # Load or compute GMM parameters
    dataset_hash = hash_dataset(dataset)
    if dataset_hash not in gmm_cache:
        gmm_cache[dataset_hash] = get_gmm_params(dataset)
    beta_hat, sigma_hat, gmm = gmm_cache[dataset_hash]

    # Compute P(Y | X) assuming y | X ~ N(X @ beta, sigma^2)
    py_given_x = norm.pdf(y_i_hat, loc=x_i @ beta_hat, scale=sigma_hat)
    
    # Compute P(X) using the fitted GMM
    if np.isnan(x_i).any():
        raise ValueError("NaN detected in x_i before computing GMM score.")
    px = np.exp(gmm.score_samples(x_i.reshape(1, -1)))[0]
    
    # Compute marginal P(Y) via integral P(Y) = ∫ P(Y|X) P(X) dX
    # Approximate using Monte Carlo by averaging over sampled X
    X_samples = gmm.sample(1000)[0]  # Sample from the GMM
    py_samples = norm.pdf(y_i_hat, loc=X_samples @ beta_hat, scale=sigma_hat)
    py = np.mean(py_samples)  # Approximate integral
    
    # Compute P(X | Y) using Bayes' theorem
    if py == 0:
        print("Warning: py is zero, setting px_given_y to a small value.")
        px_given_y = 1e-9  # Prevent division by zero, use a small value
    else:
        px_given_y = (py_given_x * px) / py

    # Safeguard to prevent log of zero or too small values
    if px_given_y <= 1e-15:
        print(f"Warning: px_given_y is too small ({px_given_y}), setting to 1e-15.")
        px_given_y = 1e-9  # Avoid log(0) or log(small number)

    # Compute the score
    gmm_score = -np.log(px_given_y + 1e-9)

    return gmm_score.item()

    # # Compute P(Y | X) assuming y | X ~ N(X @ beta, sigma^2)
    # log_py_given_x = norm.logpdf(
    #     y_i_hat, loc=x_i @ beta_hat, scale=sigma_hat)
    
    # # Compute P(X) using the fitted GMM
    # if np.isnan(x_i).any():
    #     raise ValueError("NaN detected in x_i before computing GMM score.")
    # log_px = gmm.score_samples(x_i.reshape(1, -1))[0]
    
    # # Compute marginal P(Y) via integral P(Y) = ∫ P(Y|X) P(X) dX
    # # Approximate using Monte Carlo by averaging over sampled X (M # of monte carlo samples)
    # M = 1000
    # X_samples = gmm.sample(M)[0]
    # log_py_samples = norm.logpdf(y_i_hat, loc=X_samples @ beta_hat, scale=sigma_hat)
    # log_py = logsumexp(log_py_samples) - np.log(M)
    
    # # Compute P(X | Y) using Bayes' theorem in log-space
    # log_px_given_y = log_py_given_x + log_px - log_py

    # # Compute the score
    # gmm_score = -log_px_given_y
    # return gmm_score.item()

def compute_atypicality_scores(X_test, y_pred, X_fit, y_fit, score_type):
    dataset = list(zip(X_fit, y_fit))
    scores = []

    for x_i, y_i_hat in tqdm(zip(X_test, y_pred)):
        input_point = (x_i, y_i_hat)
        if score_type == 'knn_score':
            scores.append(knn_score(input_point, dataset))
        elif score_type == 'kde_score':
            scores.append(kde_score(input_point, dataset, gaussian_kernel))
        elif score_type == 'logjointmvn_score':
            scores.append(logjointmvn_score(input_point, dataset)) 
        elif score_type == 'lognormal_score':
            scores.append(lognormal_score(input_point, dataset)) 
        elif score_type == 'gmm_score':
            scores.append(gmm_score(input_point, dataset)) 
        else:
            raise ValueError(f"Invalid score type: {score_type}")

    return scores