import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dictionaries for plotting with legible titles
data_generation_mapping = {
    "generate_and_split_mvn_data": "MVN Data Generation Setting",
    "generate_and_split_lognormal_data": "Log Normal Data Generation Setting",
    "generate_and_split_gmm_data": "GMM Data Generation Setting"
}

atypicality_score_mapping = {
    'knn_score': 'KNN',
    'kde_score': 'KDE',
    'lognormal_score': 'Log Normal',
    'gmm_score': 'GMM',
    'logjointmvn_score': 'Log Joint MVN'
}

cp_model_mapping = {
    "fit_rf_cp_model": "RFCP",
    "fit_gaussian_cp_model": "NNCP",
    "fit_conformal_cp_model": "CQR"
}

# Desired order for x-axis / plotting
desired_score_order = ["KNN", "KDE", "Log Joint\nMVN", "GMM", "Log\nNormal"]

def plot_betagrouped_by_atypicality(beta_df, true_atypicality, outputfile):
    """
    Plots regression slopes (Mean Beta) with error bars (Std Beta), 
    grouped by atypicality score and CP model, faceted by data generation setting.
    """
    # Map CP models, data generation, and atypicality scores
    beta_df["CP Model Label"] = beta_df["CP Model"].map(cp_model_mapping)
    beta_df["Data Generation Label"] = beta_df["Data Generation Setting"].map(data_generation_mapping)
    beta_df["Atypicality Label"] = beta_df["Atypicality Score"].map(atypicality_score_mapping)

    width_ratios = [4, 5, 4] 
    fig, axes = plt.subplots(1, 3, gridspec_kw={'width_ratios': width_ratios}, figsize=(11, 4), sharey=True)

    # Unique CP models and colors
    cp_models = beta_df["CP Model Label"].unique()
    cp_colors = sns.color_palette("Set1", len(cp_models))
    cp_color_map = {cp: cp_colors[i] for i, cp in enumerate(cp_models)}

    # Unique data generation settings and titles
    data_generation_settings = beta_df["Data Generation Label"].unique()

    for ax, data_gen in zip(axes, data_generation_settings):
        subset = beta_df[beta_df["Data Generation Label"] == data_gen]

        # atypicality_scores = sorted(subset["Atypicality Label"].unique())
        atypicality_scores = [s for s in desired_score_order if s in subset["Atypicality Label"].values]
        x_positions = np.arange(len(atypicality_scores))

        # Small offsets to separate CP models within the same group
        offsets = np.linspace(-0.15, 0.15, len(cp_models))

        for i, atypicality in enumerate(atypicality_scores):
            group_subset = subset[subset["Atypicality Label"] == atypicality]

            for j, cp_model in enumerate(cp_models):
                df_point = group_subset[group_subset["CP Model Label"] == cp_model]
                if df_point.empty:
                    continue

                mean_beta = df_point["Mean Beta"].values[0]
                std_beta = df_point["Std Beta"].values[0]
                color = cp_color_map[cp_model]

                # Plot error bars and points
                ax.errorbar(
                    x_positions[i] + offsets[j], mean_beta, yerr=std_beta,
                    fmt='o', capsize=5, color=color, markersize=8, elinewidth=1.5,
                    alpha=0.5
                )
                ax.plot(
                    x_positions[i] + offsets[j], mean_beta,
                    marker='o', color=color, markersize=8, alpha=1
                )

        # Customize x-axis and plot
        ax.set_xticks(x_positions)
        ax.set_xticklabels(atypicality_scores, rotation=0, fontsize=11)
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax.set_title(data_gen)
        ax.grid(True, linestyle="dotted", alpha=0.6)
        ax.tick_params(axis='both', colors='grey')
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')

    axes[0].set_ylabel("Slope (β)")

    fig.text(0.5, 0.07, 'Atypicality Score', ha='center', va='center')

    # Legend for CP models
    handles = [plt.Line2D([0], [0], marker='o', color=color, markersize=8, linestyle='')
               for color in cp_color_map.values()]
    labels = cp_models  # or a mapping to nicer labels
    fig.legend(handles, labels, title="CP Model",
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))

    if true_atypicality:
        plt.ylim(-0.14, 0.14)
    else:
        plt.ylim(-0.07, 0.07)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig("../plots/" + outputfile)
    plt.show()

def plot_coverage_across_atypicality_quantile(
        df,
        atypicality_score,
        atypicality_score_title,
        ylim_bottom=None,
        ylim_top=None,
        save=True):
    """
    df columns:
        Quantile, Coverage, lambda, score, split
    """
    # Filter for specified score
    df = df[df['score'] == atypicality_score]

    # Aggregate across splits
    agg = (df.groupby(['lambda', 'quantile'])['coverage']
        .agg(['mean', 'std'])
        .reset_index())
    agg['quantile'] = agg['quantile'].astype(int)

    plt.figure(figsize=(8, 5))

    for lam, lam_df in agg.groupby('lambda'):
        plt.errorbar(
            lam_df['quantile'],
            lam_df['mean'],
            yerr=lam_df['std'],
            fmt='o-',
            capsize=4,
            label=f'λ = {lam}')

    plt.xlabel(f'{atypicality_score_title} Atypicality Quantile')
    plt.ylabel('Coverage')
    plt.title(f'Coverage vs. {atypicality_score_title} Atypicality Quantile')
    plt.legend()
    plt.grid(True)

    if ylim_bottom is not None or ylim_top is not None:
        plt.ylim(ylim_bottom, ylim_top)

    if save:
        score_string = atypicality_score_title.lower().replace(" ", "")
        lambdas = "_".join(str(l).replace(".", "p") for l in sorted(df["lambda"].unique()))
        filename = f"coverage_vs_{score_string}_lam_{lambdas}.png"
        plt.savefig("../plots/" + filename, dpi=300, bbox_inches="tight")

    plt.show()

def plot_coverage_vs_atypicality(df, atypicality_score_title='KNN Score', atypicality_col='knn_score', lower_col='aar_knn_lower', upper_col='aar_knn_upper', 
                                 num_quantiles=5, ylim_bottom=None, ylim_top=None):
    """
    :param df: Dataframe of 
    :param atypicality_score_title: Description
    :param atypicality_col: Description
    :param lower_col: Description
    :param upper_col: Description
    :param num_quantiles: Description
    :param ylim_bottom: Description
    :param ylim_top: Description
    """
    
    # Define quantile bins based on atypicality score
    df['quantile'] = pd.qcut(df[atypicality_col], num_quantiles, labels=False)

    coverage_results = []

    for q in range(num_quantiles):
        quantile_df = df[df['quantile'] == q]
        
        # Compute coverage for both sets of bounds
        coverage_aar = ((quantile_df[lower_col] <= quantile_df['y_test']) & (quantile_df['y_test'] <= quantile_df[upper_col])).mean()
        coverage_pred = ((quantile_df['y_pred_lower'] <= quantile_df['y_test']) & (quantile_df['y_test'] <= quantile_df['y_pred_upper'])).mean()

        coverage_results.append((q, coverage_aar, coverage_pred))

    # Convert results to DataFrame for easy plotting
    coverage_df = pd.DataFrame(coverage_results, columns=['Quantile', 'Coverage_AAR', 'Coverage_Pred'])

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(coverage_df['Quantile'], coverage_df['Coverage_AAR'], marker='o', linestyle='-', label='AAR Bounds')
    plt.plot(coverage_df['Quantile'], coverage_df['Coverage_Pred'], marker='s', linestyle='--', label='Predicted Bounds')

    plt.xlabel(f'{atypicality_score_title} Atypicality Quantile')
    plt.ylabel('Coverage')
    plt.title(f'Coverage vs. {atypicality_score_title} Atypicality Quantile')
    plt.legend()
    plt.grid(True)

    # Set y-axis limits if specified
    if ylim_bottom is not None or ylim_top is not None:
        plt.ylim(ylim_bottom, ylim_top)

    plt.show()

# Where in the originally generated intervals do our points lie? 
def plot_ytest_distribution(df, atypicality_col='log_joint_mvn_score', num_quantiles=5, lower_col='y_pred_lower', upper_col='y_pred_upper', aapi=True):
    # Define quantile bins
    df['quantile'] = pd.qcut(df[atypicality_col], num_quantiles, labels=False)

    # Compute relative position of y_test in its predicted range
    df['relative_position'] = (df['y_test'] - df[lower_col]) / (df[upper_col] - df[lower_col])

    # Plot histogram for each quantile
    fig, axes = plt.subplots(1, num_quantiles, figsize=(15, 4), sharey=True)

    for q in range(num_quantiles):
        ax = axes[q]
        quantile_df = df[df['quantile'] == q]
        
        ax.hist(quantile_df['relative_position'], bins=20, range=(0, 1), alpha=0.7, color='#fc9a98', edgecolor='#F9514E')
        ax.set_title(f'Quantile {q}')

        # Set tick and border lines to grey
        ax.tick_params(axis='both', colors='grey')
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')
    
    axes[0].set_ylabel('Frequency')
    if aapi == True:
        plt.suptitle(f'Distribution of the Relative Position of Test Output $y_i$ Within Predictve Intervals across Quantiles — AAPI Bounds')
    else:
        plt.suptitle(f'Distribution of the Relative Position of Test Output $y_i$ Within Predictve Intervals across Quantiles — Predicted Bounds')
    
    fig.text(0.5, -0.02, 'Relative Position within the Interval', ha='center', va='center')
    

    plt.tight_layout()
    plt.show()

# Plotting a random sample of points and intervals before and after AAPI
def plot_sampled_predictions(df, atypicality_col='log_joint_mvn_score', lower_col='aapi_log_joint_mvn_lower', upper_col='aapi_log_joint_mvn_upper', sample_per_quantile=6, random_seed=1):
    """
    Input: df has columns of y_pred, y_test, y_pred_upper, and y_pred_lower. 

    The function samples a number of points per quantile and graphs them relative to their AAPI intervals. 
    """

    np.random.seed(random_seed)
    
    # Sample 6 rows from each quantile
    sampled_df = df.groupby('quantile', group_keys=False).apply(lambda x: x.sample(n=sample_per_quantile, random_state=random_seed))

    # Sort by quantile for clear visualization
    sampled_df = sampled_df.sort_values(by='quantile')

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 4))  # Wide and short figure

    # X-axis positions
    x_positions = np.arange(len(sampled_df))

    # Plot points
    ax.scatter(x_positions, sampled_df['y_pred'], color='#C154D1', label='predicted y', marker='o')
    ax.scatter(x_positions, sampled_df['y_test'], color='#EC0C09', label='test y', marker='x')

    # Plot prediction intervals (whiskered bars)
    ax.vlines(x_positions, sampled_df['y_pred_lower'], sampled_df['y_pred_upper'], color='#C154D1', linewidth=2, label='Pred. Interval')
    ax.vlines([x + 0.25 for x in x_positions], sampled_df[lower_col], sampled_df[upper_col], color='#4A1153', linewidth=2, label='AAPI Interval')

    # Add horizontal caps
    cap_width = 0.1  # Adjust this to control the width of the caps

    for x, y_low, y_high in zip(x_positions, sampled_df['y_pred_lower'], sampled_df['y_pred_upper']):
        ax.hlines(y=y_low, xmin=x - cap_width, xmax=x + cap_width, color='#C154D1', linewidth=2)
        ax.hlines(y=y_high, xmin=x - cap_width, xmax=x + cap_width, color='#C154D1', linewidth=2)

    for x, y_low, y_high in zip([x + 0.25 for x in x_positions], sampled_df[lower_col], sampled_df[upper_col]):
        ax.hlines(y=y_low, xmin=x - cap_width, xmax=x + cap_width, color='#4A1153', linewidth=2)
        ax.hlines(y=y_high, xmin=x - cap_width, xmax=x + cap_width, color='#4A1153', linewidth=2)

    # # Label points with atypicality score
    # for i, (x, atyp_score, quant) in enumerate(zip(x_positions, sampled_df[atypicality_col], sampled_df['quantile'])):
    #     ax.text(x, sampled_df['y_pred'].iloc[i], f"{atyp_score:.2f}  ", ha='right', fontsize=10, rotation=0)

    # Add atypicality scores below x-axis ticks
    for x, atyp_score in zip(x_positions, sampled_df[atypicality_col]):
        ax.text(x, ax.get_ylim()[0] - 0.12 * (ax.get_ylim()[1] - ax.get_ylim()[0]),  # Position slightly below the x-axis
                f"{atyp_score:.2f}", ha='center', fontsize=10, rotation=0, color='grey')


    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sampled_df['quantile'].astype(str), rotation=0, fontsize=10)  # Show quantiles on x-axis
    ax.set_ylabel('Prediction Values')
    ax.set_title(r'Sampled Predictions with Intervals — Log Joint MVN Score, CQR, GMM Data, AAPI ($\lambda=0.5$)')
    ax.legend(loc='upper right')
    plt.grid(axis='y', linestyle='dotted', alpha=0.6)
    
    ax.tick_params(axis='both', colors='grey')
    for spine in ax.spines.values():
        spine.set_edgecolor('grey')

    ax.set_xlabel('Atypicality Quantile and Atypicality Score', labelpad=20)
    plt.tight_layout()
    plt.show()

def plot_coveragepred_betas(all_results):
    x_labels = []
    slopes_mean = []
    slopes_std = []

    for key, dfs in all_results.items():
        slopes = []
        
        for df in dfs:
            df = df.dropna(axis=0, how='any')

            X = df["Quantile"].values.reshape(-1, 1)  # Reshape to 2D array for sklearn
            y = df["Coverage_Pred"].values
            
            # Create and fit the model
            model = LinearRegression()
            model.fit(X, y)
            
            # Extract the slope (coefficient)
            slopes.append(model.coef_[0])  # `coef_` contains the slope
            
        # Compute mean and standard deviation of slopes
        slopes_mean.append(np.mean(slopes))
        slopes_std.append(np.std(slopes))
        
        # Format x-axis label
        atypicality, lam = key.split("_lam")
        x_labels.append(f"{atypicality}\nlam={lam}")

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.errorbar(x_labels, slopes_mean, yerr=slopes_std, fmt="o", capsize=5, color="b", markersize=8)

    plt.axhline(0, c='r', linestyle="--", alpha=0.5)
    plt.ylim(-.15, .15)
    plt.xlabel("Atypicality Score and Lambda")
    plt.ylabel("Slope (β₁) of Coverage_Pred vs. Quantile")
    plt.title("Regression Slopes with Error Bars (Linear Data, Linear CP)")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()

def print_coverage_lambdas(coverage_dfs):
    """
    To plot coverage by lambdas.
    
    :param coverage_dfs: Description
    """
    
    # Plot all lambdas on the same graph
    plt.figure(figsize=(8, 6))
    for lambda_key, df in coverage_dfs.items():
        plt.plot(df['Quantile'], df['Coverage_AAR'], label=lambda_key)

    plt.xlabel("Quantile")
    plt.ylabel("Coverage_AAR")
    plt.title("Coverage_AAR Across Quantiles for Different Lambdas")
    plt.legend(title="Lambda")
    plt.grid(True)
    plt.show()

def plot_lambda_metrics(lambda_metrics):
    """
    Plots lambda vs. mean coverage, lambda vs. mean coefficients, and lambda vs. mean MSE
    with error bars for each metric from lambda_metrics.

    Args:
    - lambda_metrics (dict): Dictionary containing metrics for each lambda.
    """
    lambdas = []
    coverage_means = []
    coverage_lowers = []
    coverage_uppers = []

    coeff_means = []
    coeff_lowers = []
    coeff_uppers = []

    mse_means = []
    mse_lowers = []
    mse_uppers = []

    mse_mean_means = []
    mse_mean_lowers = []
    mse_mean_uppers = []

    beta_values = []
    coverage_values = []
    lambda_values = []

    # Extract the metrics for each lambda from lambda_metrics
    for lambda_key, metrics in lambda_metrics.items():
        lambda_value = float(lambda_key.split('lam')[1].replace('-', '.'))  # Parsing the lambda value
        lambdas.append(lambda_value)
        
        coverage_means.append(metrics['coverage']['mean'])
        coverage_lowers.append(metrics['coverage']['lower'])
        coverage_uppers.append(metrics['coverage']['upper'])
        
        coeff_means.append(metrics['beta_coefficients']['mean'])
        coeff_lowers.append(metrics['beta_coefficients']['lower'])
        coeff_uppers.append(metrics['beta_coefficients']['upper'])
        
        mse_means.append(metrics['mse']['mean'])
        mse_lowers.append(metrics['mse']['lower'])
        mse_uppers.append(metrics['mse']['upper'])

        mse_mean_means.append(metrics['mse_mean']['mean'])
        mse_mean_lowers.append(metrics['mse_mean']['lower'])
        mse_mean_uppers.append(metrics['mse_mean']['upper'])

        # Collect beta vs. coverage values for scatter plot
        beta_values.append(metrics['beta_coefficients']['mean'])
        coverage_values.append(metrics['coverage']['mean'])
        lambda_values.append(lambda_value)

    # Plot lambda vs. mean coverage
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.errorbar(lambdas, coverage_means, yerr=[np.array(coverage_means) - np.array(coverage_lowers),
                                                 np.array(coverage_uppers) - np.array(coverage_means)],
                 fmt='o', label='Coverage', capsize=5, color='blue')
    plt.xlabel('Lambda')
    plt.ylabel('Mean Coverage')
    plt.title('Lambda vs. Mean Coverage')
    plt.grid(True)

    # Plot lambda vs. mean coefficients
    plt.subplot(1, 4, 2)
    plt.errorbar(lambdas, coeff_means, yerr=[np.array(coeff_means) - np.array(coeff_lowers),
                                              np.array(coeff_uppers) - np.array(coeff_means)],
                 fmt='o', label='Coefficient', capsize=5, color='green')
    plt.xlabel('Lambda')
    plt.ylabel('Mean Coefficients')
    plt.title('Lambda vs. Mean Coefficients')
    plt.grid(True)

    # Plot lambda vs. mean MSE
    plt.subplot(1, 4, 3)
    plt.errorbar(lambdas, mse_means, yerr=[np.array(mse_means) - np.array(mse_lowers),
                                            np.array(mse_uppers) - np.array(mse_means)],
                 fmt='o', label='MSE', capsize=5, color='red')
    plt.xlabel('Lambda')
    plt.ylabel('Mean MSE')
    plt.title('Lambda vs. Mean MSE')
    plt.grid(True)

    # Plot lambda vs. mean MSE
    plt.subplot(1, 4, 4)
    plt.errorbar(lambdas, mse_mean_means, yerr=[np.array(mse_mean_means) - np.array(mse_mean_lowers),
                                            np.array(mse_mean_uppers) - np.array(mse_mean_means)],
                 fmt='o', label='MSE from Mean', capsize=5, color='orange')
    plt.xlabel('Lambda')
    plt.ylabel('Mean MSE - from Mean Alpha')
    plt.title('Lambda vs. Mean MSE from Mean')
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # Second plot of beta values vs. coverage for each lambda
    # TODO: add solid-line light grey line and dotted vertical line where the apex is.
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(beta_values, coverage_values, c=lambda_values, cmap='viridis', alpha=0.8)
    plt.xlabel('Beta Coefficients')
    plt.ylabel('Coverage')
    plt.title('Coverage vs. Beta for Different Lambda Values')
    cbar = plt.colorbar(scatter, label='Lambda')
    cbar.outline.set_visible(False)  # This removes the outline
    plt.grid(True)

    plt.show()