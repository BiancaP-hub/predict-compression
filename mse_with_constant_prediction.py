from sklearn.metrics import mean_squared_error
import numpy as np

def calculate_baseline_mse(y_train, y_test):
    """
    Calculate the baseline mean squared error by always predicting the mean of y_train.

    Parameters:
    y_train (ndarray): Training data labels.
    y_test (ndarray): Test data labels used for evaluating the baseline MSE.

    Returns:
    float: Baseline mean squared error.
    """
    # Calculate the mean of y_train
    mean_y_train = np.mean(y_train)

    # Create a baseline prediction array with the mean value
    baseline_predictions = np.full(shape=y_test.shape, fill_value=mean_y_train)

    # Calculate the MSE for these baseline predictions
    baseline_mse = mean_squared_error(y_test, baseline_predictions)

    return baseline_mse