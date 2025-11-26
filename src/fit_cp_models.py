import os
import joblib
import numpy as np
from deel.puncc.regression import LocallyAdaptiveCP, CQR
from deel.puncc.api.prediction import MeanVarPredictor, DualPredictor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from atypicality import hash_dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

### RFCP Model
def fit_rf_cp_model(X_fit, y_fit, X_calib, y_calib):
    # Create two models mu (mean) and sigma (dispersion)
    mu_model = RandomForestRegressor(n_estimators=100, random_state=0)
    sigma_model = RandomForestRegressor(n_estimators=100, random_state=0)
    # Wrap models in a mean/variance predictor
    mean_var_predictor = MeanVarPredictor(models=[mu_model, sigma_model])

    lacp = LocallyAdaptiveCP(mean_var_predictor)
    lacp.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)
    
    return lacp

def predict_cp_intervals(lacp, X_test, alpha=0.2):
    y_pred, y_pred_lower, y_pred_upper = lacp.predict(X_test, alpha=alpha)
    return y_pred, y_pred_lower, y_pred_upper

### NNCP Model
class GPExponentAdapter:
    def __init__(self, gp_model):
        self.gp_model = gp_model
    
    def fit(self, X, y):
        self.gp_model.fit(X, y)
    
    def predict(self, X, return_std=False):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        mean, std = self.gp_model.predict(X_tensor, return_std=True)
        return np.exp(mean)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def predict(self, x):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient tracking
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            output = self(x)
            return output.detach().cpu().numpy()

def load_or_train_cp_nn(X_fit, y_fit, weights_path):
    model = MLP(X_fit.shape[1])  # PyTorch model
    
    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X_fit, dtype=torch.float32)
    y_tensor = torch.tensor(y_fit, dtype=torch.float32).view(-1, 1)  # Ensure y is a column vector

    # DataLoader for batching
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path))
            print(f"Loaded cached weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights, retraining instead: {e}")
            # Train and save new model weights
            model.train()
            for epoch in range(100):
                for X_batch, y_batch in dataloader:
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
            torch.save(model.state_dict(), weights_path)
            print(f"Model retrained and weights saved to {weights_path}")
    else:
        print("Training model from scratch...")
        model.train()
        for epoch in range(100):
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), weights_path)
        print(f"Model trained and weights saved to {weights_path}")

    return model

def load_or_train_cp_gp(X_fit, log_residuals_propertrain, gp_weights_path): 
    # Define the GP with ARD kernel
    kernel = C(1.0, (1e-4, 1e1)) * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10)
    print('Defined GP!')

    # Load the GP model if it exists
    if os.path.exists(gp_weights_path):
        try:
            gp_adapter = GPExponentAdapter(gp)
            gp_adapter.gp_model = joblib.load(gp_weights_path)
        except Exception as e:
            print(f"Error loading model: {e}")
        print(f"Loaded cached GP model from {gp_weights_path}")
    else:
        # Fit the GP model if it doesn't exist
        gp_adapter = GPExponentAdapter(gp)
        gp_adapter.fit(X_fit, log_residuals_propertrain)

        print("GP Log Marginal Likelihood:", gp.log_marginal_likelihood_value_)

        # Save the trained GP model
        joblib.dump(gp_adapter.gp_model, gp_weights_path)
        print(f"Trained and saved new GP model to {gp_weights_path}")

    return gp_adapter

def fit_gaussian_cp_model(X_fit, y_fit, X_calib, y_calib):
    # Define the weight storage directory and filename
    weights_dir = "../intermediate/"
    os.makedirs(weights_dir, exist_ok=True)  # Ensure directory exists

    dataset_hash = hash_dataset(list(zip(X_fit, y_fit))) 
    weights_path = os.path.join(weights_dir, f"model_{dataset_hash}.weights.pt")
    model = load_or_train_cp_nn(X_fit, y_fit, weights_path)
    
    # Predict on the train set
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_fit, dtype=torch.float32)
        y_pred = model(X_tensor).numpy()

    residuals_propertrain = np.abs(y_fit - y_pred.flatten())  # Absolute residuals
    log_residuals_propertrain = np.log(residuals_propertrain + 1e-8)  # Stabilized log-residuals

    gp_weights_path = os.path.join(weights_dir, f"gp_model_{dataset_hash}.pkl")
    gp_adapter = load_or_train_cp_gp(X_fit, log_residuals_propertrain, gp_weights_path)

    # Wrap models in MeanVarPredictor
    mean_var_predictor = MeanVarPredictor(models=[model, gp_adapter])

    print("Manually setting is trained to true:")
    mean_var_predictor.is_trained = [True, True]  

    # Initialize and fit Locally Adaptive CP
    lacp = LocallyAdaptiveCP(mean_var_predictor, train=False)
    lacp.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)
    
    return lacp

### CQR (Conformal CP) Model
def fit_conformal_cp_model(X_fit, y_fit, X_calib, y_calib):

    # Lower quantile regressor
    regressor_q_low = GradientBoostingRegressor(
        loss="quantile", alpha=.2/2, n_estimators=250
    )
    # Upper quantile regressor
    regressor_q_hi = GradientBoostingRegressor(
        loss="quantile", alpha=1 - .2/2, n_estimators=250
    )
    # Wrap models in predictor
    predictor = DualPredictor(models=[regressor_q_low, regressor_q_hi])

    # CP method initialization
    crq = CQR(predictor)

    # The call to `fit` trains the model and computes the nonconformity
    # scores on the calibration set
    crq.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)
    print('Fit CQR!')

    return crq

# Linear CP Model
def fit_linear_cp_model(X_fit, y_fit, X_calib, y_calib):
    # Create two models mu (mean) and sigma (dispersion)
    mu_model = LinearRegression().fit(X_fit, y_fit)
    sigma_model = LinearRegression().fit(X_fit, y_fit)
    # Wrap models in a mean/variance predictor
    mean_var_predictor = MeanVarPredictor(models=[mu_model, sigma_model])

    lacp = LocallyAdaptiveCP(mean_var_predictor)
    lacp.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)
    
    return lacp