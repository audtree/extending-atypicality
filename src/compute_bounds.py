import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from collections import defaultdict

import sys
sys.path.append("../src")

from atypicality import compute_atypicality_scores
from data_generation_settings import generate_and_split_gaussian_data
from fit_cp_models import fit_rf_cp_model, predict_cp_intervals

def generate_base_cp_intervals_and_atypicality(
        make_and_split_data,
        fit_cp_model,
        score_type,
        n_samples,
        random_seed,
        test_atypicality):
    X_fit, X_calib, X_test, y_fit, y_calib, y_test, _ = \
        make_and_split_data(random_seed=random_seed, n_samples=n_samples)

    lacp = fit_cp_model(X_fit, y_fit, X_calib, y_calib)
    y_pred, y_lower, y_upper = predict_cp_intervals(lacp, X_test)

    # Calibration median
    calib_scores = compute_atypicality_scores(
        X_calib, y_calib, X_fit, y_fit, score_type=score_type)
    med_score = np.median(calib_scores)

    # Test scores
    y_for_score = y_test if test_atypicality else y_pred[:, 0]
    scores = compute_atypicality_scores(
        X_test, y_for_score, X_fit, y_fit, score_type=score_type)

    df = pd.DataFrame({
            "y_test": y_test,
            "y_pred": y_pred[:, 0],
            "y_pred_lower": y_lower,
            "y_pred_upper": y_upper,
            score_type: scores,
            "base_scaling": (scores - med_score) / med_score})

    return df

def apply_lambda_adjustment(df, lam, lower_col, upper_col):
    scaling = 1 + lam * df["base_scaling"]
    out = df.copy()
    out[lower_col] = out["y_pred"] - scaling * (out["y_pred"] - out["y_pred_lower"])
    out[upper_col] = out["y_pred"] + scaling * (out["y_pred_upper"] - out["y_pred"])
    out["lambda"] = lam
    return out

def coverage_by_quantile(df, score_col, lower_col, upper_col, num_quantiles=5):
    df = df.copy()
    df["quantile"] = pd.qcut(df[score_col], num_quantiles, labels=False)

    rows = []
    for q in range(num_quantiles):
        qdf = df[df["quantile"] == q]
        rows.append({
            "Quantile": q,
            "Coverage": ((qdf[lower_col] <= qdf["y_test"]) &
                         (qdf["y_test"] <= qdf[upper_col])).mean()})

    return pd.DataFrame(rows)

def group_lambdas_by_score(settings):
    grouped = defaultdict(list)
    for score, lam in settings:
        grouped[score].append(lam)
    return grouped

def evaluate_lambda_adjusted_interval_coverage(
        atypicality_settings,
        make_and_split_data,
        fit_cp_model,
        n_samples,
        n_splits,
        test_atypicality,
        num_quantiles):
    
    lambdas_by_score = group_lambdas_by_score(atypicality_settings)
    results = []

    for split in range(n_splits):
        for score_type, lambdas in lambdas_by_score.items():

            base_df = generate_base_cp_intervals_and_atypicality(
                make_and_split_data,
                fit_cp_model,
                score_type,
                n_samples,
                split,
                test_atypicality)

            y_test = base_df["y_test"].values
            y_pred = base_df["y_pred"].values
            lower0 = base_df["y_pred_lower"].values
            upper0 = base_df["y_pred_upper"].values
            scores = base_df[score_type].values
            base_scaling = base_df["base_scaling"].values

            quantiles = pd.qcut(scores, num_quantiles, labels=False)

            for lam in lambdas:
                scaling = 1 + lam * base_scaling

                lower = y_pred - scaling * (y_pred - lower0)
                upper = y_pred + scaling * (upper0 - y_pred)

                for q in range(num_quantiles):
                    mask = quantiles == q
                    coverage = np.mean(
                        (lower[mask] <= y_test[mask]) &
                        (y_test[mask] <= upper[mask]))

                    results.append({
                        "score": score_type,
                        "lambda": lam,
                        "quantile": q,
                        "coverage": coverage,
                        "split": split})

    return pd.DataFrame(results)

# # Calculate adjusted bounds
# def compute_adjusted_bounds(df, score_col, med_score, y_pred_lower_col, y_pred_upper_col, lower_col, upper_col, lam=1):
#     """
#     Given a dataframe with feature columns, creates new columns with adjusted bounds 
#     based on the inputted lambda value.
    
#     :param df: Dataframe with features, atypicality scores
#     :param score_col: Column name for atypicality score (calculated for each point)
#     :param med_score: Median atypicality score
#     :param y_pred_lower_col: Existing column name for lower bound, to be adjusted by lambda
#     :param y_pred_upper_col: Existing column name for upper bound, to be adjusted by lambda
#     :param lower_col: Name of the lower bound column, to be created
#     :param upper_col: Name of the upper bound column, to be created
#     :param lam: Lambda value with which to scale the bounds (higher lambda means 
#                 adjust for atypicality)
#     """
#     if lam is not None: 
#         scaling_factor = 1 + lam * (df[score_col] - med_score) / med_score

#     df[lower_col] = df['y_pred'] - (scaling_factor * (df['y_pred'] - df[y_pred_lower_col]))
#     df[upper_col] = df['y_pred'] + (scaling_factor * (df[y_pred_upper_col] - df['y_pred']))

#     # Compute fractions of non-real numbers
#     lower_invalid_frac = np.mean(~np.isfinite(df[lower_col]))  # Fraction of -inf, inf, or NaN
#     upper_invalid_frac = np.mean(~np.isfinite(df[upper_col]))

#     if (lower_invalid_frac + upper_invalid_frac > 0):
#         print(f"Fraction of non-real values in {lower_col}: {lower_invalid_frac:.4f}")
#         print(f"Fraction of non-real values in {upper_col}: {upper_invalid_frac:.4f}")

#         # Check for NaN or inf in your original data
#         print("Check for NaN or inf in your original data:")
#         print(df[[score_col, 'y_pred', y_pred_lower_col, y_pred_upper_col]].describe())
#         print(df[[score_col, 'y_pred', y_pred_lower_col, y_pred_upper_col]].isna().sum())
#         print(np.isinf(df[[score_col, 'y_pred', y_pred_lower_col, y_pred_upper_col]]).sum())

#         # Check if scaling factor is very large
#         print("Check if scaling factor is large:", scaling_factor)

#         # Check extreme values in df[score_col] - med_score
#         print("Check extreme values in df[score_col] - med_score")
#         print(df[score_col].max(), df[score_col].min(), med_score)

# def compute_coverage_by_quantile(df, atypicality_col, lower_col, upper_col, num_quantiles=5):
#     """
#     Given a dataframe of points
    
#     :param df: 
#     :param atypicality_col: Description
#     :param lower_col: Description
#     :param upper_col: Description
#     :param num_quantiles: Description
#     """

#     df['quantile'] = pd.qcut(df[atypicality_col], num_quantiles, labels=False)
#     coverage_results = []

#     for q in range(num_quantiles):
#         quantile_df = df[df['quantile'] == q]
#         coverage_aar = ((quantile_df[lower_col] <= quantile_df['y_test']) & (quantile_df['y_test'] <= quantile_df[upper_col])).mean()
#         coverage_pred = ((quantile_df['y_pred_lower'] <= quantile_df['y_test']) & (quantile_df['y_test'] <= quantile_df['y_pred_upper'])).mean()
#         coverage_results.append((q, coverage_aar, coverage_pred))

#     return pd.DataFrame(coverage_results, columns=['Quantile', 'Coverage_AAR', 'Coverage_Pred'])

# def run_calibration_for_score(atypicality_settings,
#                               make_and_split_data=generate_and_split_gaussian_data, 
#                               fit_cp_model=fit_rf_cp_model, 
#                               n_samples=5000,
#                               n_splits=10, 
#                               test_atypicality=False):
#     all_results = {f"{atyp_col}_lam{str(lam).replace('.', '')}": [] for atyp_col, _, _, lam in atypicality_settings}

#     for i in range(n_splits):
#         print(f"Running split {i+1}/{n_splits}")
#         X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler = make_and_split_data(random_seed=i, n_samples=n_samples)

#         # Train CP model and get predictions
#         lacp = fit_cp_model(X_fit, y_fit, X_calib, y_calib)
#         y_pred, y_pred_lower, y_pred_upper = predict_cp_intervals(lacp, X_test)

#         # Create DataFrame
#         df = pd.DataFrame(X_test, columns=[f'feature_{j}' for j in range(X_test.shape[1])])
#         df['y_test'], df['y_pred'] = y_test, y_pred[:, 0]
#         assert df['y_test'].notna().all(), "There are NaN values in df['y_test']!"

#         df['y_pred_lower'], df['y_pred_upper'] = y_pred_lower, y_pred_upper

#         r2 = r2_score(y_test, y_pred[:, 0])
#         mse = mean_squared_error(y_test, y_pred[:, 0])
#         print(f'R^2 score for split {i+1}: {r2:.4f}')
#         print(f'MSE score for split {i+1}: {mse:.4f}')

#         # Calculate lambda-adjusted coverage
#         for atyp_col, lower_col, upper_col, lam in atypicality_settings:
#             med_score = np.median(compute_atypicality_scores(X_calib, y_calib, X_fit, y_fit, score_type=atyp_col))
            
#             # Compute atypicality score for the current method
#             if test_atypicality:
#                 # if test_atypicality is True, use y_test instead of y_pred
#                 df[atyp_col] = compute_atypicality_scores(X_test, y_test.flatten(), X_fit, y_fit, score_type=atyp_col)
#             else:
#                 # Otherwise, use y_pred
#                 df[atyp_col] = compute_atypicality_scores(X_test, y_pred[:,0].flatten(), X_fit, y_fit, score_type=atyp_col)

#             # Compute adjusted bounds
#             compute_adjusted_bounds(df, atyp_col, med_score, 'y_pred_lower', 'y_pred_upper', lower_col, upper_col, lam=lam)

#             # Compute coverage per quantile
#             coverage_df = compute_coverage_by_quantile(df, atyp_col, lower_col, upper_col)
#             all_results[f"{atyp_col}_lam{str(lam).replace('.', '')}"].append(coverage_df)

#     return all_results

# def compute_predicted_and_adjusted_bounds(atyp_col="log_joint_mvn_score",
#                               make_and_split_data=generate_and_split_gaussian_data, 
#                               fit_cp_model=fit_rf_cp_model, 
#                               n_samples=5000,
#                               n_splits=10, 
#                               lam=0,
#                               test_atypicality=False):
#     '''
#     For n_splits splits, generate a dataset with the given experimental settings. 
#     Fit a CP model to the training set, returning upper and lower prediction interval bounds. 
#     If lambda is not 0, compute atypicality scores for each point. Adjust the interval
#     bounds according to lambda. Return a dataset with atypicality 
#     '''
#     all_results = {f"{atyp_col}_lam{str(lam).replace('.', '')}": [] for atyp_col, _, _, lam in atypicality_settings}

#     for i in range(n_splits):
#         print(f"Running split {i+1}/{n_splits}")
#         X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler = make_and_split_data(random_seed=i, n_samples=n_samples)

#         # Train CP model and get predictions
#         lacp = fit_cp_model(X_fit, y_fit, X_calib, y_calib)
#         y_pred, y_pred_lower, y_pred_upper = predict_cp_intervals(lacp, X_test)

#         # Create DataFrame
#         df = pd.DataFrame(X_test, columns=[f'feature_{j}' for j in range(X_test.shape[1])])
#         df['y_test'], df['y_pred'] = y_test, y_pred[:, 0]
#         assert df['y_test'].notna().all(), "There are NaN values in df['y_test']!"

#         df['y_pred_lower'], df['y_pred_upper'] = y_pred_lower, y_pred_upper

#         r2 = r2_score(y_test, y_pred[:, 0])
#         mse = mean_squared_error(y_test, y_pred[:, 0])
#         print(f'R^2 score for split {i+1}: {r2:.4f}')
#         print(f'MSE score for split {i+1}: {mse:.4f}')

#         # Calculate lambda-adjusted coverage
#         for atyp_col, lower_col, upper_col, lam in atypicality_settings:
#             med_score = np.median(compute_atypicality_scores(X_calib, y_calib, X_fit, y_fit, score_type=atyp_col))
            
#             # Compute atypicality score for the current method
#             if test_atypicality:
#                 # if test_atypicality is True, use y_test instead of y_pred
#                 df[atyp_col] = compute_atypicality_scores(X_test, y_test.flatten(), X_fit, y_fit, score_type=atyp_col)
#             else:
#                 # Otherwise, use y_pred
#                 df[atyp_col] = compute_atypicality_scores(X_test, y_pred[:,0].flatten(), X_fit, y_fit, score_type=atyp_col)

#             # Compute adjusted bounds
#             compute_adjusted_bounds(df, atyp_col, med_score, 'y_pred_lower', 'y_pred_upper', lower_col, upper_col, lam=lam)

#             # Compute coverage per quantile
#             coverage_df = compute_coverage_by_quantile(df, atyp_col, lower_col, upper_col)
#             all_results[f"{atyp_col}_lam{str(lam).replace('.', '')}"].append(coverage_df)

#     return all_results, df