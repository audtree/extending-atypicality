import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import sys
sys.path.append("../src")

from atypicality import compute_atypicality_scores
from data_generation_settings import generate_and_split_mvn_data, generate_and_split_gmm_data
from fit_cp_models import fit_rf_cp_model, predict_cp_intervals
from compute_bounds import apply_lambda_adjustment, coverage_by_quantile, evaluate_lambda_adjusted_interval_coverage

def compute_mean_stds(data):
    """
    Computes mean, lower, and upper bounds for a given set of data.
    """
    data_array = np.array(data)
    
    # Create the dictionary with mean, lower, and upper bounds
    stats = {
        'mean': np.mean(data_array),
        'lower': np.percentile(data_array, 5),  # Lower bound (5th percentile)
        'upper': np.percentile(data_array, 95)  # Upper bound (95th percentile)
    }
    
    # Add the original data list to the dictionary
    stats['data'] = data
    
    return stats

def generate_atypicality_settings_01(atyp_col, steps):
    # Generate lambda values from 0 to 1 with 20 steps
    # lambda_values = np.linspace(0, 1, steps) # TODO: change back
    lambda_values = np.linspace(0, 1, steps)
    
    # Generate the list of tuples
    atypicality_settings = [(atyp_col, lam) for lam in lambda_values]
    return atypicality_settings

def get_bound_names_from_lambdakey(score_name):
    # Extracting the 'atyp_col' and 'lam' from the score name
    parts = score_name.split('_lam')
    atyp_col = '_'.join(parts[:-1])  # Join the part before '_lam' to get the 'atyp_col'
    lam = parts[-1].replace('-', '.')  # Convert '-' back to '.' for lambda

    # Constructing the upper and lower quantile names
    lower_col = f'aar_{atyp_col}_lower_lam{str(lam).replace(".", "-")}'
    upper_col = f'aar_{atyp_col}_upper_lam{str(lam).replace(".", "-")}'

    return lower_col, upper_col

def calc_lambda_tuning_metrics(df, atyp_col, lower_col, upper_col, y_test_col, alpha=0.2, num_quantiles=5):
    """
    Input: df is a list of dataframes of replication splits. 
    """
    
    # Store replication results
    replication_results = {}
    coverage_dfs = []
    coverage_all = []
    mse_mean_all = []
    beta_coefficients_all = []

    for replication_split in df:
        # Compute coverage by quantile
        coverage_df = coverage_by_quantile(replication_split, atyp_col, lower_col, upper_col, num_quantiles)
        coverage_dfs.append(coverage_df)

        # Compute overall coverage
        overall_coverage = ((replication_split[lower_col] <= replication_split[y_test_col]) & (replication_split[y_test_col] <= replication_split[upper_col])).mean()
        coverage_all.append(float(overall_coverage))

        # Compute MSE from the mean coverage
        mse_mean = mean_squared_error([np.mean(coverage_df['Coverage'])] * len(coverage_df), coverage_df['Coverage'])
        mse_mean_all.append(mse_mean)

        # Compute beta coefficient
        X = coverage_df["Quantile"].values.reshape(-1, 1)  # Reshape to 2D array for sklearn
        y = coverage_df["Coverage"].values
        model = LinearRegression()
        model.fit(X, y)
        beta_coefficients_all.append(model.coef_[0])  # `coef_` contains the slope

    # Aggregate all coverage DataFrames
    merged_df = pd.concat(coverage_dfs).groupby('Quantile').agg(['mean', 'std']).reset_index()

    replication_results['coverage'] = compute_mean_stds(coverage_all)
    replication_results['mse_mean'] = compute_mean_stds(mse_mean_all)
    replication_results['beta_coefficients'] = compute_mean_stds(beta_coefficients_all)

    # Convert results into DataFrame
    return pd.DataFrame(replication_results), merged_df

def print_notable_lambdas(all_results):
    # Initialize variables to track notable lambdas
    highest_coverage_lambda = None
    highest_coverage_value = -float('inf')

    closest_to_zero_lambda = None
    closest_to_zero_value = float('inf')

    lowest_mse_mean_lambda = None
    lowest_mse_mean_value = float('inf')

    # Iterate over the dictionary (all_results) for each lambda
    for lambda_key, replication_results in all_results.items():
        # Assuming replication_results is a DataFrame, and has 'coverage', 'mse', and 'beta_coefficients'
        coverage_mean = replication_results['coverage']['mean']
        beta_mean = np.mean(replication_results['beta_coefficients']['mean'])  # Assuming we want the mean of all betas
        mse_mean_mean = replication_results['mse_mean']['mean']

        # Find the lambda with the highest overall coverage
        if coverage_mean > highest_coverage_value:
            highest_coverage_lambda = lambda_key
            highest_coverage_value = coverage_mean

        # Find the lambda with coefficients closest to 0 (lowest absolute value of the mean beta coefficient)
        if abs(beta_mean) < abs(closest_to_zero_value):
            closest_to_zero_lambda = lambda_key
            closest_to_zero_value = beta_mean
        
        # Find the lambda with the lowest mean MSE
        if mse_mean_mean < lowest_mse_mean_value:
            lowest_mse_mean_lambda = lambda_key
            lowest_mse_mean_value = mse_mean_mean

    # Print the notable lambdas
    print(f"Lambda with highest overall coverage: {highest_coverage_lambda} with coverage {highest_coverage_value}")
    print(f"Lambda with coefficients closest to 0: {closest_to_zero_lambda} with coefficient value {closest_to_zero_value}")
    print(f"Lambda with lowest MSE: {lowest_mse_mean_lambda} with MSE {lowest_mse_mean_value}")

def lambda_hyperparameter_tuning(atyp_col='log_joint_mvn_score', make_and_split_data=generate_and_split_gmm_data, 
                                 fit_cp_model=fit_rf_cp_model, n_splits=2, true_atypicality=True, hyperparameter_tuning=True,
                                 best_lambda=0):
    """
    Important: calculates metrics on y_test. When used for hyperparameter tuning, this 
    treats y_test like a second calibration set for choosing lambda. The data (which 
    includes y_test) is generated in calc_lam_aar_intervals with a random seed up to
    the number of splits. 
    """

    # If hyperparameter tuning is true, treat y_test like a test set (random_seed = 0)
    if hyperparameter_tuning:
        random_seed_start = 0

        # Generate atypicality_settings for 20 lambdas between 0 and 1
        atypicality_settings = generate_atypicality_settings_01(atyp_col, steps=5)
    
    if not hyperparameter_tuning:
        # Random seeds from 0 to n_splits-1 have already been used for hyperparameter tuning; need unseen data for evaluation
        random_seed_start = n_splits 

        # Generate the atypicality settings
        lambda_values = set([0.0, best_lambda])
        atypicality_settings = [ (atyp_col, lam) for lam in lambda_values]
    
    # Calculate n_splits replication splits for each lambda and above settings
    coverage_df, lambda_results = evaluate_lambda_adjusted_interval_coverage(atypicality_settings,
                                                    make_and_split_data,
                                                    fit_cp_model,
                                                    n_samples=2000,
                                                    n_splits=n_splits,
                                                    true_atypicality=true_atypicality,
                                                    num_quantiles=5,
                                                    return_df=True,
                                                    silent=True)

    # Calculate metrics for each lambda across replication splits
    lambda_metrics = {}
    merged_dfs = {}
    for lambda_key, lambda_value in lambda_results.items():
        # Each iteration calcualtes metrics for another lambda
        lower_col, upper_col = get_bound_names_from_lambdakey(lambda_key)
        lambda_metrics[lambda_key], merged_df = calc_lambda_tuning_metrics(lambda_value, atyp_col, lower_col, upper_col, 'y_test', alpha=0.2, num_quantiles=5)
        merged_dfs[lambda_key] = merged_df
    
    # Print notable lambdas
    print_notable_lambdas(lambda_metrics)
    return lambda_metrics, lambda_results, merged_dfs

# Function to select for best lambda
def get_best_lambda(data_dict, coverage_metric='coverage', beta_metric='beta_coefficients', highest=True, beta_weight=0.4):
    """
    Returns the lambda value corresponding to the highest overall coverage, considering both coverage and beta-coefficient.
    The best lambda is the one with high coverage and a beta-coefficient close to 0.

    Parameters:
    - data_dict: dict of DataFrames, where keys contain lambda values
    - coverage_metric: str, the column name for overall coverage (default 'coverage')
    - beta_metric: str, the column name for beta coefficient (default 'beta_coefficients')
    - highest: bool, if True, returns the lambda with the highest metric value, else returns the lowest
    - beta_weight: float, a weight that determines the importance of the beta-coefficient in the decision (default 0.5)

    Returns:
    - float, the lambda value corresponding to the best (highest/lowest) metric
    """
    best_lambda = None
    best_score = float('-inf') if highest else float('inf')
    
    for key, df in data_dict.items():
        lambda_val = float(key.split('lam')[1].replace('-', '.'))
        
        # Extract the coverage and beta coefficient values
        coverage_value = df.loc['mean', coverage_metric]
        beta_value = df.loc['mean', beta_metric]
        
        # Calculate the combined score: high coverage and low beta coefficient
        # Absolute value of beta coefficient close to 0 is better (lower is better)
        score = coverage_value - beta_weight * abs(beta_value)
        
        if (highest and score > best_score) or (not highest and score < best_score):
            best_score = score
            best_lambda = lambda_val
    
    return best_lambda