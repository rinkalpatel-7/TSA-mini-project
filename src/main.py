import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.models import TimeSeriesAnalyzer

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
    summary = analyzer.fit_arima(order=(1, 1, 1))
    print(summary)
    
    # 6. Save a Visual for Error Analysis (Section D Requirement)
    plt.figure(figsize=(10,6))
    ts_data.plot(title="Time Series Data & Trend")
    plt.savefig("analysis/results_plot.png")
    print("\nAnalysis plot saved to analysis/results_plot.png")

if __name__ == "__main__":
    run_experiment()