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

# Generate simulated datasets
def generate_and_split_mvn_data(random_seed, test_size=0.2, calib_size=0.2, noise_std=0.5, n_samples=5000):
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

# Load real-world datasets
def load_and_split_chd_data(random_seed, test_size=0.2, calib_size=0.2, n_samples=None):
    # Load California Housing dataset
    X, y = fetch_california_housing(return_X_y=True)

    # Take a random subset of the dataset
    X, _, y, _ = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler = split_and_scale_data(X, y, test_size, calib_size, random_seed)

    return X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler

def load_and_split_diabetes_data(random_seed, test_size=0.2, calib_size=0.2, n_samples=None):
    # Load Diabetes dataset
    X, y = load_diabetes(return_X_y=True)

    # Take a random subset of the dataset
    X, _, y, _ = train_test_split(X, y, random_state=random_seed)

    X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler = split_and_scale_data(X, y, test_size, calib_size, random_seed)

    return X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler

def load_and_split_hf_data(random_seed, test_size=0.2, calib_size=0.2, n_samples=None):
    # Load Wine dataset
    df_hf = pd.read_csv('../data/heart_failure_clinical_records_dataset.csv')
    X, y = df_hf.drop(columns=['DEATH_EVENT']), df_hf['DEATH_EVENT']

    # Take a random subset of the dataset
    X, _, y, _ = train_test_split(X, y, random_state=random_seed)

    X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler = split_and_scale_data(X, y, test_size, calib_size, random_seed)

    return X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler

from sklearn.impute import SimpleImputer
def load_and_split_support_data(random_seed, test_size=0.2, calib_size=0.2, n_samples=None):
    df_support2 = pd.read_csv('../data/support2.csv')

    # Drop rows with 1 missing value
    missing_counts = df_support2.isnull().sum()
    columns_with_one_missing = missing_counts[missing_counts == 1].index.tolist()
    rows_to_drop_indices = []

    for col in columns_with_one_missing:
        row_index = df_support2[df_support2[col].isnull()].index[0]
        rows_to_drop_indices.append(row_index)
    df_support2.drop(list(set(rows_to_drop_indices)), inplace=True)

    # Drop previous models' recommendations
    leaky_vars_to_drop = ['aps', 'sps', 'surv2m', 'surv6m', 'prg2m', 'prg6m', 'dnr', 'dnrday']
    df_support2.drop(columns=leaky_vars_to_drop, inplace=True)

    # Drop columns with more than 50% missingness
    drop_cols = ['adlp', 'urine', 'glucose']
    df_support2.drop(columns=drop_cols, inplace=True)

    # Add missingness flags
    for col in df_support2.columns:
        df_support2[col + '_missing'] = df_support2[col].isnull().astype(int)

    # Impute the rest of the columns
    num_cols = df_support2.select_dtypes(include='number').columns.tolist()
    cat_cols = [c for c in df_support2.columns if c not in num_cols and not c.endswith('_missing')]

    # Numeric imputation with median
    num_imputer = SimpleImputer(strategy='median')
    df_support2[num_cols] = num_imputer.fit_transform(df_support2[num_cols])

    # Impute categorical features with mode
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df_support2[cat_cols] = cat_imputer.fit_transform(df_support2[cat_cols])

    # Drop rows with 6 or more missing values
    missing_flag_cols = [c for c in df_support2.columns if c.endswith('_missing')]
    df_support2['num_missing'] = df_support2[missing_flag_cols].sum(axis=1)
    df_support2 = df_support2[df_support2['num_missing'] <= 6].copy()

    # Map ordinal columns
    ordinal_cols = ['income', 'sfdm2']
    income_mapping = {
        'under $11k': 1,
        '$11-$25k': 2,
        '$25-$50k': 3,
        '>$50k': 4}

    sfdm2_mapping = {
        'no(M2 and SIP pres)': 1,
        'adl>=4 (>=5 if sur)': 2,
        'SIP>=30': 3,
        'Coma or Intub': 4,
        '<2 mo. follow-up': 5}

    df_support2['income'] = df_support2['income'].map(income_mapping)
    df_support2['sfdm2'] = df_support2['sfdm2'].map(sfdm2_mapping)

    # One-hot encode remaining categorical columns
    onehot_cols = [c for c in df_support2.columns if c not in ordinal_cols and not c.endswith('_missing') and df_support2[c].dtype == object]
    df_support2 = pd.get_dummies(df_support2, columns=onehot_cols, drop_first=False, dtype=int)

    # # Temporary sample of dataset to make it smaller
    # df_support2 = df_support2.sample(n=500, random_state=random_seed)

    # Split data
    X, y = df_support2.drop(columns=['sfdm2']).to_numpy(), df_support2['sfdm2'].to_numpy()
    X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler = split_and_scale_data(X, y, test_size, calib_size, random_seed)

    return X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler