import csv
import os
from datetime import datetime

def log_experiment(order, results, log_path="analysis/log.csv"):
    """
    Automates experiment tracking by appending model results to a CSV file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    run_id = f"EXP-{datetime.now().strftime('%Y%m%d-%H%M')}"
    file_exists = os.path.isfile(log_path)
    
    data = {
        "Run_ID": run_id,
        "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "Model": "ARIMA",
        "Order": str(order),
        "AIC": round(results.aic, 2),
        "BIC": round(results.bic, 2),
        "Log_Likelihood": round(results.llf, 2)
    }

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
    
    print(f" Experiment {run_id} successfully logged to {log_path}")