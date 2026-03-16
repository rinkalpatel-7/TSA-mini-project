import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_failures(data):
    result = seasonal_decompose(data)
    # The 'Residual' plot shows the variance not captured by the model, 
    # which can indicate potential failures in capturing underlying patterns.
    result.resid.plot(title="Model Residuals - Identifying Non-captured Variance")
    plt.savefig("analysis/residuals.png")