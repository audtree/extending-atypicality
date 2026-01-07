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
    'lognormal_score': 'Log\nNormal',
    'gmm_score': 'GMM',
    'logjointmvn_score': 'Log Joint\nMVN'
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

    # Unique CP models, colors, and markers
    cp_models = beta_df["CP Model Label"].unique()
    cp_colors = sns.color_palette("Set1", len(cp_models))
    cp_color_map = {cp: cp_colors[i] for i, cp in enumerate(cp_models)}
    markers = ['o', 's', '^', 'D', 'P', 'X']  # extend if needed
    cp_marker_map = {cp: markers[i] for i, cp in enumerate(cp_models)}

    # Unique data generation settings and titles
    data_generation_settings = beta_df["Data Generation Label"].unique()

    for ax, data_gen in zip(axes, data_generation_settings):
        subset = beta_df[beta_df["Data Generation Label"] == data_gen]

        # atypicality_scores = sorted(subset["Atypicality Label"].unique())
        atypicality_scores = [s for s in desired_score_order if s in subset["Atypicality Label"].values]
        x_positions = np.arange(len(atypicality_scores))

        # Small offsets to separate CP models within the same group
        offsets = np.linspace(-0.18, 0.18, len(cp_models))

        # Customize x-axis and plot
        ax.set_xticks(x_positions)
        ax.set_xticklabels(atypicality_scores, rotation=0, fontsize=11)
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax.set_title(data_gen)
        ax.grid(True, linestyle="dotted", alpha=0.6)
        ax.tick_params(axis='both', colors='grey')
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')

        for i, atypicality in enumerate(atypicality_scores):
            group_subset = subset[subset["Atypicality Label"] == atypicality]

            for j, cp_model in enumerate(cp_models):
                df_point = group_subset[group_subset["CP Model Label"] == cp_model]
                if df_point.empty:
                    continue

                mean_beta = df_point["Mean Beta"].values[0]
                std_beta = df_point["Std Beta"].values[0]
                color = cp_color_map[cp_model]
                marker = cp_marker_map[cp_model]

                # Plot error bars and points
                ax.errorbar(
                    x_positions[i] + offsets[j], mean_beta, yerr=std_beta,
                    fmt=marker, capsize=5, markersize=8, color=color, elinewidth=1.5, # elinewidth=1.5, markersize=8,
                    alpha=0.5)
                ax.plot(
                    x_positions[i] + offsets[j], mean_beta,
                    marker=marker, markersize=8, color=color, alpha=1
                )

        
    axes[0].set_ylabel("Slope (β)")

    fig.text(0.5, 0.08, 'Atypicality Score', ha='center', va='center')

    # Legend for CP models
    handles = [plt.Line2D( [0], [0], marker=cp_marker_map[cp], color=cp_color_map[cp], 
                          markersize=8, linestyle='' ) for cp in cp_models]
    labels = cp_models 
    fig.legend(handles, labels, 
               loc="lower right", ncol=3, bbox_to_anchor=(1, 0.02))

    if true_atypicality:
        plt.ylim(-0.17, 0.17)
    else:
        plt.ylim(-0.075, 0.075)
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
    df = df[df['score'] == atypicality_score].copy()
    
    # Aggregate across splits
    agg = (df.groupby(['lambda', 'quantile'])['coverage']
           .agg(['mean', 'std'])
           .reset_index())
    agg['quantile'] = agg['quantile'].astype(int)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(5, 4))

    # Define colors and markers for each lambda
    colors = ['#4A1153', '#C154D1', '#FF8000', '#B84000', '#007ACC']  # extend if more lambdas
    
    for i, (lam, lam_df) in enumerate(agg.groupby('lambda')):
        color = colors[i % len(colors)]

        if (lam == 0):
            fmt_solid = 'o-'
        else:
            fmt_solid = 'o--'

        # Plot mean line
        ax.errorbar(
            lam_df['quantile'],
            lam_df['mean'],
            fmt=fmt_solid,
            label=f'λ = {lam}',
            capsize=5,
            color=color,
            alpha=1
        )
        # Add translucent error bars
        ax.errorbar(
            lam_df['quantile'],
            lam_df['mean'],
            yerr=lam_df['std'],
            fmt='o-',
            capsize=5,
            color=color,
            alpha=0.3,
            elinewidth=2,
            label=None
        )

    # Labels and title
    ax.set_xlabel(f'Atypicality Quantile\n({atypicality_score_title} Atypicality Score)')
    ax.set_ylabel('Coverage')
    ax.set_title(f'Coverage vs. {atypicality_score_title} Atypicality Quantile')

    # Grid and style
    ax.grid(True, which="both", linestyle='dotted')
    ax.tick_params(axis='both', colors='grey')
    for spine in ax.spines.values():
        spine.set_edgecolor('grey')

    # Set x-ticks to only quantiles
    ax.set_xticks(agg['quantile'].unique())

    ax.legend()
    plt.tight_layout()

    if ylim_bottom is not None or ylim_top is not None:
        plt.ylim(ylim_bottom, ylim_top)

    if save:
        score_string = atypicality_score_title.lower().replace(" ", "")
        lambdas = "_".join(str(l).replace(".", "p") for l in sorted(df["lambda"].unique()))
        filename = f"coverage_vs_{score_string}_lam_{lambdas}.png"
        plt.savefig("../plots/" + filename, dpi=300, bbox_inches="tight")

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

        mse_mean_means.append(metrics['mse_mean']['mean'])
        mse_mean_lowers.append(metrics['mse_mean']['lower'])
        mse_mean_uppers.append(metrics['mse_mean']['upper'])

        # Collect beta vs. coverage values for scatter plot
        beta_values.append(metrics['beta_coefficients']['mean'])
        coverage_values.append(metrics['coverage']['mean'])
        lambda_values.append(lambda_value)

    # Plot lambda vs. mean coverage
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.errorbar(lambdas, coverage_means, yerr=[np.array(coverage_means) - np.array(coverage_lowers),
                                                 np.array(coverage_uppers) - np.array(coverage_means)],
                 fmt='o', label='Coverage', capsize=5, color='blue')
    plt.xlabel('Lambda')
    plt.ylabel('Mean Coverage')
    plt.title('Lambda vs. Mean Coverage')
    plt.grid(True)

    # Plot lambda vs. mean coefficients
    plt.subplot(1, 3, 2)
    plt.errorbar(lambdas, coeff_means, yerr=[np.array(coeff_means) - np.array(coeff_lowers),
                                              np.array(coeff_uppers) - np.array(coeff_means)],
                 fmt='o', label='Coefficient', capsize=5, color='green')
    plt.xlabel('Lambda')
    plt.ylabel('Mean Coefficients')
    plt.title('Lambda vs. Mean Coefficients')
    plt.grid(True)

    # Plot lambda vs. mean MSE
    plt.subplot(1, 3, 3)
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

# Plotting 3-subplot figure for best lambda
# Helper function to filter list of dictionaries
def preprocess_and_filter_data(data, data_generation_setting, cp_model, atypicality_scores):
    filtered_data = [
        entry for entry in data
        if entry['data_generation_setting'] == data_generation_setting and 
        entry['cp_model'] == cp_model and 
        entry['atyp_col'] in atypicality_scores
    ]
    return filtered_data

# Helper function to get paired data
def extract_paired_metrics(filtered_data, atypicality_scores, metric):
    # For each atypicality score, get the best lambda (if it's not zero)
    lambda_0_data = {}
    best_lambda_data = {}

    for entry in filtered_data:
        lambda_value = entry['lambda']
        if lambda_value == 0.0:
            lambda_0_data[entry['atyp_col']] = entry[metric]
        else:
            # Store non-zero lambda values
            if entry['atyp_col'] not in best_lambda_data or entry[metric] > best_lambda_data[entry['atyp_col']][metric]:
                best_lambda_data[entry['atyp_col']] = entry

    # Merge data for lambda = 0 and the best lambda
    pairs = []
    for score in atypicality_scores:
        # Get coverage for lambda = 0
        coverage_lambda_0 = lambda_0_data.get(score, None)
        # Get coverage for best lambda
        best_lambda_entry = best_lambda_data.get(score, None)
        coverage_best_lambda = best_lambda_entry[metric] if best_lambda_entry else coverage_lambda_0

        pairs.append((coverage_lambda_0, coverage_best_lambda))
    return pairs

# Helper function to make paired bar plot 
# Input axes, filtered data, which metric
def plot_paired_bar_chart(filtered_data, atypicality_scores, metric, metric_title, ax):
    """
    Plots a paired bar chart for lambda = 0 vs. best lambda.
    
    Parameters:
    - atypicality_scores: List of atypicality scores (x-axis labels).
    - pairs: List of tuples containing paired coverage values for each score.
    - ax: The axes object to plot the chart on (for use in subplots).
    """
    bar_width = 0.35
    index = np.arange(len(atypicality_scores))
    pairs = extract_paired_metrics(filtered_data, atypicality_scores, metric)

    # Create bars for lambda = 0 and best lambda
    bar1 = ax.bar(index, [pair[0] for pair in pairs], bar_width, label='Lambda = 0')
    bar2 = ax.bar(index + bar_width, [pair[1] for pair in pairs], bar_width, label='Best Lambda')

    # Adding labels and title
    ax.set_xlabel('Atypicality Score')
    ax.set_ylabel(metric_title)
    ax.set_title(f'Predicted vs. AAPI {metric_title} Comparison ')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(atypicality_scores)

    if metric == 'coverage_mean':
        ax.set_ylim(0, 1)
    elif metric == 'beta_coeff_mean':
        ax.set_ylim(-0.15, 0.15)
        ax.axhline(y=0, color='grey', linestyle='dashed')

    ax.legend()

# Helper function for coverage plot
def coverage_atypicality_plot(merged_df, atypicality_score_title, ax,
                              ylim_bottom=None, ylim_top=None):
    """Distinct from `plot_coverage_across_atypicality_quantile` in that this function
    plots already-aggregated coverage, instead of aggregating within the function. 
    `plot_coverage_across_atypicality_quantile` also plots different lambda values; this
    function does not.
    """

    ax.errorbar(
        merged_df['Quantile'],
        merged_df[('Coverage', 'mean')],
        yerr=merged_df[('Coverage', 'std')],
        fmt='o-',
        capsize=4
    )

    ax.set_xlabel(f'{atypicality_score_title} Atypicality Quantile')
    ax.set_ylabel('Coverage')
    ax.set_title(f'Coverage vs. {atypicality_score_title} Quantile')
    ax.grid(True)

    if ylim_bottom is not None or ylim_top is not None:
        ax.set_ylim(ylim_bottom, ylim_top)

def plot_datagen_lambda_comparison(lambda_metric_results, 
                                   coverage_results,
                                    data_generation_setting,
                                    data_generation_setting_title,
                                    cp_model,
                                    atyp_col = 'log_joint_mvn_score',
                                    atypicality_score_title = 'Log Joint MVN Score',
                                    atypicality_scores = ['knn_score', 'kde_score', 'log_joint_mvn_score', 'gmm_score'],
                                    true_atypicality=False):
    
    # Filter for data generation process and CP algorithm
    lambda_metric_results_filtered = preprocess_and_filter_data(lambda_metric_results, data_generation_setting, cp_model, atypicality_scores)
    # coverage_result_filtered = preprocess_and_filter_data(coverage_results, data_generation_setting, cp_model, [atyp_col])[0]['merged_df']
    coverage_result_filtered = preprocess_and_filter_data(coverage_results, data_generation_setting, cp_model, [atyp_col])[0]['merged_df']

    # Example usage for subplots:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))  # 1 row, 2 columns

    # Assume `atypicality_scores` and `pairs` are already defined
    available_scores = sorted({e['atyp_col'] for e in lambda_metric_results_filtered})
    plot_paired_bar_chart(lambda_metric_results_filtered, available_scores, 'coverage_mean', 'Coverage Mean', axes[0])
    plot_paired_bar_chart(lambda_metric_results_filtered, available_scores, 'beta_coeff_mean', 'Beta Coefficient', axes[1])
    coverage_atypicality_plot(coverage_result_filtered, atypicality_score_title, axes[2])

    # Add a bigger title for the entire plot
    if true_atypicality == True:
        atypicality_type_title = 'True Atypicality'
        filepath = f"../plots/lam3fig-testatypicality-{atyp_col}-{str(data_generation_setting).split(' ')[1].split('_')[3]}data-{str(cp_model).split(' ')[1].split('_')[1]}cp.png"
    else:
        atypicality_type_title = 'Predicted Atypicality'
        filepath = f"../lam3fig-predatypicality-{atyp_col}-{str(data_generation_setting).split(' ')[1].split('_')[3]}data-{str(cp_model).split(' ')[1].split('_')[1]}cp.png"
    fig.suptitle(f"Overall Comparison for {data_generation_setting_title} — {atypicality_type_title}, {str(cp_model).split(' ')[1].split('_')[1].upper()} CP", fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust space for the title

    plt.savefig(filepath)
    plt.show()

# Plot real-world dataset attribute analyses
def avg_coverage_by_attr_witherror(dfs, attribute="attr", attribute_name="Ethnicity", ethnicity_mapping=None, desired_order=None):
    # Compute coverage for each dataframe if not already present
    for df in dfs:
        df['covered'] = ((df['y_test'] >= df['y_pred_lower']) & 
                        (df['y_test'] <= df['y_pred_upper'])).astype(int)
    
    # Stack coverage across splits for each group
    coverage_by_group = pd.concat([df.groupby(attribute)['covered'].mean().rename(f'split_{i}') 
        for i, df in enumerate(dfs)], axis=1)
    
    # Compute mean and standard error
    coverage_mean = coverage_by_group.mean(axis=1)
    coverage_se = coverage_by_group.std(axis=1, ddof=1) / np.sqrt(len(dfs))

    # Apply ethnicity mapping if provided
    if ethnicity_mapping is not None:
        coverage_mean.index = coverage_mean.index.map(lambda x: ethnicity_mapping.get(x, x))
        coverage_se.index = coverage_se.index.map(lambda x: ethnicity_mapping.get(x, x))
    
    # Reorder according to desired_order if provided
    if desired_order is not None:
        coverage_mean = coverage_mean.reindex(desired_order)
        coverage_se = coverage_se.reindex(desired_order)

    fig, ax = plt.subplots(figsize=(5,4))

    # Create shades of blue for bars
    n_bars = len(coverage_mean)
    colors = plt.cm.Blues(np.linspace(1, 0.4, n_bars))

    # Bar positions
    x_pos = np.arange(len(coverage_mean))

    # Bar plot with error bars
    ax.bar(
        x_pos,
        coverage_mean.values,
        yerr=coverage_se.values,
        capsize=5,
        color=colors,
        width=0.8,
        alpha=0.82,
        ecolor='#18203B')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(coverage_mean.index, rotation=0, ha='center')
    ax.set_xlabel(attribute_name)
    ax.set_ylabel('Coverage')

    # Add numbers on top of bars
    for i, (mean, se) in enumerate(zip(coverage_mean.values, coverage_se.values)):
        # Set color: first 3 bars white, rest #18203B
        text_color = 'white' if i < 2 else '#18203B'
        
        ax.text(
            i,
            mean - 0.075,
            f"{mean:.2f}",
            ha='center',
            fontsize=10,
            color=text_color
        )

    # Grid and style
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", which="major", linestyle='dotted')
    ax.tick_params(axis='both', colors='grey')
    for spine in ax.spines.values():
        spine.set_edgecolor('grey')

    plt.show()
def heatmap_by_quantile_attribute(dfs, attribute="attr", atyp_col='logjointmvn_score', num_quantiles=5, attribute_name="Ethnicity", ethnicity_mapping=None):
    # Ensure 'covered' column exists for each split
    for df in dfs:
        df['covered'] = ((df['y_test'] >= df['y_pred_lower']) & 
                         (df['y_test'] <= df['y_pred_upper'])).astype(int)
        df['quantile'] = pd.qcut(df[atyp_col], num_quantiles, labels=False)

    # Compute coverage by quantile and attribute for each split
    coverage_splits = []
    for df in dfs:
        coverage_q_attr = df.groupby(['quantile', attribute])['covered'].mean().unstack()
        coverage_splits.append(coverage_q_attr)
    
    # Take mean across splits
    coverage_mean = pd.concat(coverage_splits).groupby(level=0).mean()
    
    # Apply ethnicity mapping if provided
    if ethnicity_mapping is not None:
        coverage_mean.columns = [ethnicity_mapping.get(col, col) for col in coverage_mean.columns]
    
    # Plot heatmap
    plt.figure(figsize=(5,4))
    ax = sns.heatmap(coverage_mean, annot=True, fmt=".2f", cmap="rocket_r", cbar_kws={'label': 'Average Coverage'})
    
    plt.xticks(rotation=0)
    plt.xlabel(attribute_name)
    plt.ylabel('Atypicality Quantile')
    plt.yticks(rotation=0)
    
    # Style
    ax.tick_params(axis='both', colors='grey')
    for spine in ax.spines.values():
        spine.set_edgecolor('grey')
    
    plt.tight_layout()
    plt.show()
def plot_mean_demographic_composition(
    dfs,
    score_col="logjointmvn_score",
    attr_col="attr",
    ethnicity_map=None,
    desired_order=None,
    num_quantiles=5):
    ct_list = []

    for df in dfs:
        df = df.copy()

        # Map ethnicity labels
        if ethnicity_map is not None:
            df[attr_col] = df[attr_col].map(ethnicity_map)

        # Assign quantiles per split
        df["quantile"] = pd.qcut(df[score_col], num_quantiles, labels=False)

        # Normalized crosstab
        ct = pd.crosstab(df["quantile"], df[attr_col], normalize="index")
        ct_list.append(ct)

    # Align columns across splits
    all_cols = sorted(set(col for ct in ct_list for col in ct.columns))
    ct_aligned = [ct.reindex(columns=all_cols, fill_value=0) for ct in ct_list]

    # Mean composition across splits
    mean_ct = pd.concat(ct_aligned).groupby(level=0).mean()

    # Enforce desired column order
    if desired_order is not None:
        mean_ct = mean_ct.reindex(columns=desired_order)

    # Create shades of blue (darkest first)
    n_groups = mean_ct.shape[1]
    colors = plt.cm.Blues(np.linspace(1, 0.4, n_groups))

    # Plot
    ax = mean_ct.plot(
        kind="bar",
        stacked=True,
        figsize=(5, 4),
        color=colors)

    # Styling
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", which="major", linestyle="dotted")
    ax.tick_params(axis="both", colors="grey")
    for spine in ax.spines.values():
        spine.set_edgecolor("grey")

    plt.xlabel("Atypicality Quantile")
    plt.ylabel("Proportion")
    plt.xticks(rotation=0)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1],
        labels[::-1],
        title="Ethnicity",
        bbox_to_anchor=(1.05, 1),
        loc="upper left")
    plt.tight_layout()
    plt.show()