import matplotlib.pyplot as plt

def residuals_plot(y_test, y_pred, model_type):
    # Ensure y_pred is flattened if it has an extra dimension
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()

    residuals = y_test - y_pred
    plt.title(f'{model_type} Residuals Plot')
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='-')

    # save plot in plots folder
    plt.savefig(f'plots/{model_type}_residuals.png')
    plt.show()

def loss_plot(history, model_type):
    plt.title(f'{model_type} Loss Plot')
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()

    # save plot in plots folder
    plt.savefig(f'plots/{model_type}_loss.png')

    plt.show()

def actual_vs_predicted_plot(y_test, y_pred, model_type):
    plt.scatter(y_test, y_pred)
    plt.title(f'{model_type} Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line

    # Save plot in plots folder
    plt.savefig(f'plots/{model_type}_actual_vs_predicted.png')
    plt.show()