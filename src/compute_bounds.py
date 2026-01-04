import sys
sys.path.append("../src")

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from collections import defaultdict
from atypicality import compute_atypicality_scores
from data_generation_settings import generate_and_split_mvn_data
from fit_cp_models import fit_rf_cp_model, predict_cp_intervals
from contextlib import contextmanager, nullcontext 

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

def generate_base_cp_intervals_and_atypicality(
        make_and_split_data,
        fit_cp_model,
        score_type,
        n_samples,
        random_seed,
        true_atypicality):
    
    X_fit, X_calib, X_test, y_fit, y_calib, y_test, _ = \
        make_and_split_data(random_seed=random_seed, n_samples=n_samples)

    lacp = fit_cp_model(X_fit, y_fit, X_calib, y_calib)
    y_pred, y_lower, y_upper = predict_cp_intervals(lacp, X_test)

    # Calibration median
    calib_scores = compute_atypicality_scores(
        X_calib, y_calib, X_fit, y_fit, score_type=score_type)
    med_score = np.median(calib_scores)

    # Test scores
    y_for_score = y_test if true_atypicality else y_pred[:, 0]
    scores = compute_atypicality_scores(
        X_test, y_for_score, X_fit, y_fit, score_type=score_type)

    print("Test!")
    print("y_test:", np.shape(y_test))
    print("y_pred:", np.shape(y_pred))
    print("y_lower:", np.shape(y_lower))
    print("y_upper:", np.shape(y_upper))
    print("scores:", np.shape(scores))

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
        n_splits=5,
        true_atypicality=True,
        num_quantiles=5,
        return_df=False,
        silent=False,
        random_seed_start=0):
    
    lambdas_by_score = group_lambdas_by_score(atypicality_settings)
    coverage_results = []
    df_results = {f"{score_type}_lam{str(lam).replace('.', '-')}" : [] 
              for score_type, lambdas in group_lambdas_by_score(atypicality_settings).items() 
              for lam in lambdas}

    # Wrap the entire loop in the suppress context
    context = suppress_all_output() if silent else nullcontext()
    with context:
        for split in range(n_splits):
            seed = random_seed_start + split
            for score_type, lambdas in lambdas_by_score.items():

                base_df = generate_base_cp_intervals_and_atypicality(
                    make_and_split_data,
                    fit_cp_model,
                    score_type,
                    n_samples,
                    seed,
                    true_atypicality)

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

                        coverage_results.append({
                            "score": score_type,
                            "lambda": lam,
                            "quantile": q,
                            "coverage": coverage,
                            "split": split})
                        
                        # Store raw dataframe if requested
                        if return_df:
                            base_df[f"aapi_{score_type}_lower_lam{str(lam).replace('.', '-')}"] = lower
                            base_df[f"aapi_{score_type}_upper_lam{str(lam).replace('.', '-')}"] = upper
                            df_results[f"{score_type}_lam{str(lam).replace('.', '-')}"].append(
                                base_df[['y_test', 'y_pred', 'y_pred_lower', 'y_pred_upper', score_type,
                                        f"aapi_{score_type}_lower_lam{str(lam).replace('.', '-')}",
                                        f"aapi_{score_type}_upper_lam{str(lam).replace('.', '-')}"]])
                            
    return pd.DataFrame(coverage_results), df_results