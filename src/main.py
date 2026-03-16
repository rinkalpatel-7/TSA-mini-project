import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from generate_logs import log_experiment
from src.models import TimeSeriesAnalyzer
import joblib
from src.generate_logs import log_experiment



def run_experiment():
    # 1. Reproducibility: Set seeds 
    np.random.seed(42)
    
    print("--- Starting TSA Project ---")
    
    # 2. Generate Synthetic Data
    date_range = pd.date_range(start='2017', periods=120, freq='MS')
    raw_values = np.random.randint(-10, 10, size=len(date_range)).cumsum()
    ts_data = pd.Series(raw_values, index=date_range)
    
    # 3. Initialize Analyzer
    analyzer = TimeSeriesAnalyzer(ts_data)
    
    # 4. Perform Statistical Test
    stats = analyzer.perform_adf_test()
    print(f"ADF Statistic: {stats['statistics']:.4f}")
    print(f"P-Value: {stats['p_value']:.4f}")
    print(f"Is stationary? {stats['is_stationary']}")
    
    # 5. Model Fitting (ARIMA)
    print("\nFitting ARIMA(1,1,1) model...")
    results = analyzer.fit_arima(order=(1, 1, 1))
    print(results.summary())

    # Call the logger after fitting the model to log the results
    log_experiment(order=(1, 1, 1), results=results)
    
    # 6. Save a Visual for Error Analysis (Residuals)
    plt.figure(figsize=(10,6))
    ts_data.plot(title="Time Series Data & Trend")
    plt.savefig("analysis/results_plot.png")
    print("\nAnalysis plot saved to analysis/results_plot.png")

    # 7. Save Model Checkpoint for Reproducibility
    checkpoint_path = "analysis/arima_model_v1.pkl"
    joblib.dump(results, checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    run_experiment()