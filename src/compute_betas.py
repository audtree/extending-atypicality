import os
import sys
import numpy as np
import pandas as pd
from contextlib import contextmanager, nullcontext 

from data_generation_settings import generate_and_split_gaussian_data, generate_and_split_lognormal_data, generate_and_split_gmm_data
from fit_cp_models import fit_rf_cp_model, fit_gaussian_cp_model, fit_conformal_cp_model
from compute_bounds import evaluate_lambda_adjusted_interval_coverage

@contextmanager
def suppress_all_output():
    """Context manager to suppress all stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def linear_slope(x, y):
    """
    Compute slope of linear regression of y ~ x
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    return np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)

def compute_beta(true_atypicality, silent=False):
    beta_results = []

    data_generation_settings = [generate_and_split_gaussian_data,
                                generate_and_split_lognormal_data,
                                generate_and_split_gmm_data]

    cp_models = [fit_rf_cp_model, 
                fit_gaussian_cp_model, 
                fit_conformal_cp_model]

    # Wrap the entire loop in the suppress context
    context = suppress_all_output() if silent else nullcontext()
    with context:
        for data_gen in data_generation_settings:
            for cp_model in cp_models:

                # Define relevant atypicality scores for this data
                if data_gen in [generate_and_split_gaussian_data, generate_and_split_gmm_data]:
                    atypicality_settings = [
                        ("knn_score", 0),
                        ("kde_score", 0),
                        ("logjointmvn_score", 0),
                        ("gmm_score", 0)
                    ]
                else:  # Lognormal
                    atypicality_settings = [
                        ("knn_score", 0),
                        ("kde_score", 0),
                        ("logjointmvn_score", 0),
                        ("lognormal_score", 0),
                        ("gmm_score", 0)
                    ]

                # Run the new evaluation function
                df_results = evaluate_lambda_adjusted_interval_coverage(
                    atypicality_settings,
                    make_and_split_data=data_gen,
                    fit_cp_model=cp_model,
                    n_samples=500,
                    n_splits=5,
                    true_atypicality=true_atypicality,
                    num_quantiles=5
                )

                # Compute slopes for each atypicality score and lambda
                for (score, lam), group_df in df_results.groupby(["score", "lambda"]):
                    # Compute slope for each split
                    slopes = group_df.groupby("split").apply(
                        lambda df: linear_slope(df["quantile"].values, df["coverage"].values),
                        include_groups=False
                    ).values

                    mean_slope = slopes.mean()
                    std_slope = slopes.std(ddof=1)  # sample std
                    
                    beta_results.append({
                        "Data Generation Setting": data_gen.__name__,
                        "CP Model": cp_model.__name__,
                        "Atypicality Score": score,
                        "Lambda": lam,
                        "Mean Beta": mean_slope,
                        "Std Beta": std_slope})
                
    return pd.DataFrame(beta_results)