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
def split_and_scale_data_attr(X, y, test_size, calib_size, random_seed, extra_arrays=None):
    """
    Same as split_and_scale_data, but with a sensitive attribute in extra_arrays.
    """
    if extra_arrays is None:
        extra_arrays = []

    # First split: train / test
    split_out = train_test_split(
        X, y, *extra_arrays,
        test_size=test_size,
        random_state=random_seed)

    X_train, X_test = split_out[0], split_out[1]
    y_train, y_test = split_out[2], split_out[3]

    n_extra = len(extra_arrays)
    extra_train = split_out[4 : 4 + n_extra]
    extra_test  = split_out[4 + n_extra :]

    # Second split: fit / calib (ONLY training extras)
    split_out = train_test_split(
        X_train, y_train, *extra_train,
        test_size=calib_size,
        random_state=random_seed)

    X_fit, X_calib = split_out[0], split_out[1]
    y_fit, y_calib = split_out[2], split_out[3]
    extra_fit_calib = split_out[4:]

    # Scale X only
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_fit)
    X_calib = scaler.transform(X_calib)
    X_test = scaler.transform(X_test)

    return X_fit, X_calib, X_test, y_fit, y_calib, y_test, *extra_fit_calib, *extra_test, scaler

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
def load_and_split_chd_data(random_seed, test_size=0.2, calib_size=0.2, n_samples=50):
    """
    The CHD dataset is 20640 total data points, which means n_samples cannot exceed 20640. 
    """
    # Load California Housing Dataset
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    X['IncQuantile'] = pd.qcut(X['MedInc'], 5, labels=False)

    # cols_to_log = [ 'total_rooms', 'total_bedrooms', 'population', 'households']
    # cols_to_log = [c for c in cols_to_log if c in X.columns]

    # # Shift if necessary to avoid log(0) or negatives
    # for col in cols_to_log:
    #     min_val = X[col].min()
    #     if min_val <= 0:
    #         X[col] = np.log1p(X[col] - min_val + 1)  # shift to >0
    #     else:
    #         X[col] = np.log(X[col])

    # Split data
    inc_quant = X['IncQuantile']
    X.drop(columns=['IncQuantile', 'MedInc'], inplace=True)

    # Sample a subset
    sampled_idx = X.sample(n=n_samples, random_state=1).index
    X = X.loc[sampled_idx].values
    y = y.loc[sampled_idx].values
    inc_quant = inc_quant.loc[sampled_idx].values

    X_fit, X_calib, X_test, \
        y_fit, y_calib, y_test, \
            inc_fit, inc_calib, inc_test, scaler = split_and_scale_data_attr(X, y, 
                                                                                test_size=0.2, 
                                                                                calib_size=0.2, 
                                                                                random_seed=random_seed, 
                                                                                extra_arrays=[inc_quant])

    return X_fit, X_calib, X_test, y_fit, y_calib, y_test, inc_fit, inc_calib, inc_test, scaler

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

def load_and_split_support_numeric_data(random_seed, test_size=0.2, calib_size=0.2, n_samples=None):
    df_support2 = pd.read_csv('../data/support2.csv')

    # Drop if race or sfdm2 is na
    df_support2 = df_support2.dropna(subset=['race', 'sfdm2'])

    # Drop previous models' recommendations (suggested by data doc)
    leaky_vars_to_drop = ['aps', 'sps', 'surv2m', 'surv6m', 'prg2m', 'prg6m', 'dnr', 'dnrday']
    df_support2.drop(columns=leaky_vars_to_drop, inplace=True)
    
    # Drop rows with 1 missing value
    missing_counts = df_support2.isnull().sum()
    columns_with_one_missing = missing_counts[missing_counts == 1].index.tolist()
    rows_to_drop_indices = []

    for col in columns_with_one_missing:
        row_index = df_support2[df_support2[col].isnull()].index[0]
        rows_to_drop_indices.append(row_index)
    df_support2.drop(list(set(rows_to_drop_indices)), inplace=True)

    # Drop columns with more than 50% missingness
    drop_cols = ['adlp', 'urine', 'glucose']
    df_support2.drop(columns=drop_cols, inplace=True)

    # Impute the numeric columns
    num_cols = df_support2.select_dtypes(include='number').columns.tolist()
    num_imputer = SimpleImputer(strategy='median')
    df_support2[num_cols] = num_imputer.fit_transform(df_support2[num_cols])

    # Drop categorical columns except for race and sfdm2
    cols_to_drop = ['ca', 'dzgroup', 'dzclass', 'sex', 'income', 'hospdead', 'diabetes', 'dementia', 'race']
    df_support2.drop(columns=cols_to_drop, inplace=True)

    # Map ordinal sfdm2
    sfdm2_mapping = {
        'no(M2 and SIP pres)': 1,
        'adl>=4 (>=5 if sur)': 2,
        'SIP>=30': 3,
        'Coma or Intub': 4,
        '<2 mo. follow-up': 5}
    df_support2['sfdm2'] = df_support2['sfdm2'].map(sfdm2_mapping)

    # One-hot encode race
    # df_support2 = pd.get_dummies(df_support2, columns=['race'], drop_first=False, dtype=int)

    # Add missingness flags
    for col in df_support2.columns:
        df_support2[col + '_missing'] = df_support2[col].isnull().astype(int)

    # Drop rows with 6 or more missing values
    missing_flag_cols = [c for c in df_support2.columns if c.endswith('_missing')]
    df_support2['num_missing'] = df_support2[missing_flag_cols].sum(axis=1)
    df_support2 = df_support2[df_support2['num_missing'] <= 6].copy()

    # Recompute numeric columns AFTER all transformations
    num_cols = df_support2.select_dtypes(include='number').columns.tolist()

    # Drop _missing and helper columns
    num_cols = [c for c in num_cols if not c.endswith('_missing') and c != 'num_missing']
    assert 'sfdm2' in num_cols
    df_support2 = df_support2[num_cols].copy()

    # Log transform columns
    # cols_to_log = [
    # 'slos', 'd.time', 'num.co', 'scoma', 'charges', 'totcst', 'totmcst', 
    # 'avtisst', 'sps', 'aps', 'hday', 'wblc', 'hrt', 'resp', 'temp', 
    # 'alb', 'bili', 'crea', 'bun'
    # ]
    # cols_to_log = [c for c in cols_to_log if c in df_support2.columns]

    # # Shift if necessary to avoid log(0) or negatives
    # for col in cols_to_log:
    #     min_val = df_support2[col].min()
    #     if min_val <= 0:
    #         df_support2[col] = np.log1p(df_support2[col] - min_val + 1)  # shift to >0
    #     else:
    #         df_support2[col] = np.log(df_support2[col])

    # Temporary sample of dataset to make it smaller
    df_support2 = df_support2.sample(n=1000, random_state=random_seed)

    # Split data
    X, y = df_support2.drop(columns=['sfdm2']).to_numpy(), df_support2['sfdm2'].to_numpy()
    X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler = split_and_scale_data(X, y, test_size, calib_size, random_seed)

    return X_fit, X_calib, X_test, y_fit, y_calib, y_test, scaler

def load_and_split_mimic_data(random_seed, test_size=0.2, calib_size=0.2, n_samples=50):
    # Primary Admissions information
    df = pd.read_csv(r"../data/MIMIC-IV-v1.0/admissions.csv")
    df_pat = pd.read_csv(r"../data/MIMIC-IV-v1.0/patients.csv")
    df_diagcode = pd.read_csv(r"../data/MIMIC-IV-v1.0/diagnoses_icd.csv")
    df_icu = pd.read_csv(r"../data/MIMIC-IV-v1.0/icustays.csv")

    print('Dataset has {} unique admission events.'.format(df['hadm_id'].nunique()))
    print('Dataset has {} unique patients.'.format(df['subject_id'].nunique()))

    # Merge patient demographics including gender and race
    df = df.merge(df_pat[['subject_id', 'gender', 'anchor_age']], 
                on='subject_id', how='left')

    # Calculate Length of Stay
    df['ADMITTIME'] = pd.to_datetime(df['admittime'])
    df['DISCHTIME'] = pd.to_datetime(df['dischtime'])
    df['LOS'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.total_seconds()/86400

    # Drop rows with negative or zero LOS
    df = df[df['LOS'] > 0]

    # Filter LOS < 40 days to reduce skewness
    df = df[df['LOS'] < 40]

    print(f"LOS Statistics:\nMean: {df['LOS'].mean():.2f} days")
    print(f"Median: {df['LOS'].median():.2f} days")
    print(f"Min: {df['LOS'].min():.2f} days")
    print(f"Max: {df['LOS'].max():.2f} days")

    # Standardize ethnicity categories
    df['ethnicity'] = df['ethnicity'].replace(regex=r'^ASIAN\D*', value='ASIAN')
    df['ethnicity'] = df['ethnicity'].replace(regex=r'^WHITE\D*', value='WHITE')
    df['ethnicity'] = df['ethnicity'].replace(regex=r'^HISPANIC\D*', value='HISPANIC/LATINO')
    df['ethnicity'] = df['ethnicity'].replace(regex=r'^BLACK\D*', value='BLACK/AFRICAN AMERICAN')
    df['ethnicity'] = df['ethnicity'].replace(['UNABLE TO OBTAIN', 'OTHER', 'UNKNOWN'], 
                        value='OTHER/UNKNOWN')

    # Keep only top 5 categories
    top_ethnicities = df['ethnicity'].value_counts().nlargest(5).index.tolist()
    df.loc[~df['ethnicity'].isin(top_ethnicities), 'ethnicity'] = 'OTHER/UNKNOWN'

    print("\nEthnicity Distribution:")
    print(df['ethnicity'].value_counts())

    # Process other demographics
    # Marital status
    df['marital_status'] = df['marital_status'].fillna('UNKNOWN')

    # Deceased indicator
    df['DECEASED'] = df['deathtime'].notnull().astype(int)

    print(f"\n{df['DECEASED'].sum()} of {df['subject_id'].nunique()} patients died")

    # Process Diagnosis Codes
    # Filter ICD-9 codes and recode
    df_diagcode['recode'] = df_diagcode['icd_code'][df_diagcode['icd_version'] == 9]
    mask = df_diagcode['recode'].str.contains("[a-zA-Z]", na=False)
    df_diagcode['recode'] = df_diagcode['recode'][~mask]
    df_diagcode['recode'] = df_diagcode['recode'].fillna('999')
    df_diagcode['recode'] = df_diagcode['recode'].str.slice(start=0, stop=3, step=1)
    df_diagcode['recode'] = df_diagcode['recode'].astype(int)

    # ICD-9 Main Category ranges
    icd9_ranges = [(1, 140), (140, 240), (240, 280), (280, 290), (290, 320), 
                (320, 390), (390, 460), (460, 520), (520, 580), (580, 630), 
                (630, 680), (680, 710), (710, 740), (740, 760), (760, 780), 
                (780, 800), (800, 1000), (1000, 2000)]

    diag_dict = {0: 'infectious', 1: 'neoplasms', 2: 'endocrine', 3: 'blood',
                4: 'mental', 5: 'nervous', 6: 'circulatory', 7: 'respiratory',
                8: 'digestive', 9: 'genitourinary', 10: 'pregnancy', 11: 'skin', 
                12: 'muscular', 13: 'congenital', 14: 'prenatal', 15: 'misc',
                16: 'injury', 17: 'misc'}

    for num, cat_range in enumerate(icd9_ranges):
        df_diagcode['recode'] = np.where(
            df_diagcode['recode'].between(cat_range[0], cat_range[1]), 
            num, df_diagcode['recode'])

    df_diagcode['cat'] = df_diagcode['recode'].replace(diag_dict)

    # Create admission-diagnosis matrix
    hadm_list = df_diagcode.groupby('hadm_id')['cat'].apply(list).reset_index()
    hadm_item = pd.get_dummies(hadm_list['cat'].apply(pd.Series).stack()).groupby(level=0).sum()
    hadm_item = hadm_item.join(hadm_list['hadm_id'], how="outer")

    # Merge with main dataframe
    df = df.merge(hadm_item, how='inner', on='hadm_id')

    # Process ICU Information
    df_icu['first_careunit'] = df_icu['first_careunit'].replace({
        'Coronary Care Unit (CCU)': 'ICU',
        'Neuro Stepdown': 'NICU',
        'Neuro Intermediate': 'NICU',
        'Cardiac Vascular Intensive Care Unit (CVICU)': "ICU",
        'Neuro Surgical Intensive Care Unit (Neuro SICU)': 'ICU',
        'Medical/Surgical Intensive Care Unit (MICU/SICU)': 'ICU',
        'Medical Intensive Care Unit (MICU)': 'ICU',
        'Surgical Intensive Care Unit (SICU)': 'ICU',
        'Trauma SICU (TSICU)': 'ICU'})

    df_icu['cat'] = df_icu['first_careunit']
    icu_list = df_icu.groupby('hadm_id')['cat'].apply(list).reset_index()

    # Create admission-ICU matrix
    icu_item = pd.get_dummies(icu_list['cat'].apply(pd.Series).stack()).groupby(level=0).sum()
    icu_item[icu_item >= 1] = 1
    icu_item = icu_item.join(icu_list['hadm_id'], how="outer")

    # Merge with main dataframe
    df = df.merge(icu_item, how='outer', on='hadm_id')
    df['ICU'] = df['ICU'].fillna(value=0)
    df['NICU'] = df['NICU'].fillna(value=0)

    # Drop unnecessary columns
    columns_to_drop = ['admission_location', 'subject_id', 'hadm_id', 'ADMITTIME', 
                    'admittime', 'DISCHTIME', 'dischtime', 'discharge_location', 
                    'language', 'DECEASED', 'deathtime', 'edregtime', 'edouttime',
                    'hospital_expire_flag']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], 
            inplace=True, errors='ignore')

    # Drop rows with NaNs in features or target
    df = df.dropna().copy()

    # Save ethnicity
    ethnicity = df['ethnicity']
    df.drop(columns=['ethnicity'], inplace=True)

    # One-hot encode categorical variables
    categorical_cols = ['admission_type', 'insurance', 'marital_status', 
                    'gender']
    df_encoded = pd.get_dummies(df.drop(['LOS'], axis=1), 
                                columns=categorical_cols, 
                                drop_first=True)
    # Save features and target variables
    X = df_encoded
    y = df['LOS']

    # Sample a subset
    sampled_idx = X.sample(n=n_samples, random_state=random_seed).index
    X = X.loc[sampled_idx].values
    y = y.loc[sampled_idx].values
    ethnicity = ethnicity.loc[sampled_idx].values

    X_fit, X_calib, X_test, \
        y_fit, y_calib, y_test, \
            ethfit, eth_calib, eth_test, scaler = split_and_scale_data_attr(X, y, 
                                                                                test_size=test_size, 
                                                                                calib_size=calib_size, 
                                                                                random_seed=random_seed, 
                                                                                extra_arrays=[ethnicity])
    return X_fit, X_calib, X_test, y_fit, y_calib, y_test, ethfit, eth_calib, eth_test, scaler
