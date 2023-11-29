import numpy as np
from sklearn.metrics import mean_squared_error
from config import suppress_stdout

def bootstrap_confidence_interval(model, X_test, y_test, n_iterations=1000, ci=95):
    mse_values = []

    for i in range(n_iterations):
        # Sample with replacement from X_test and y_test
        indices = np.random.choice(len(X_test), len(X_test), replace=True)
        X_sample, y_sample = X_test[indices], y_test[indices]

        # Predict and calculate MSE
        with suppress_stdout():
            y_pred = model.predict(X_sample)

        # If NaN values in y_pred, skip this iteration
        if np.isnan(y_pred).any():
            continue
        mse = mean_squared_error(y_sample, y_pred)
        mse_values.append(mse)

    # Compute percentiles for confidence interval
    lower = (100 - ci) / 2
    upper = 100 - lower
    lower_bound, upper_bound = np.percentile(mse_values, [lower, upper])

    # Calculate the standard deviation of MSE
    mse_std_dev = np.std(mse_values)

    return (lower_bound, upper_bound), mse_std_dev
