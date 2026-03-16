import pandas as pd
import numpy as np

def generate_synthetic_data(seed=42):
    """Sets seed for reproducibility and generates synthetic time series data."""
    np.random.seed(seed)
    date_range = pd.date_range(start='2017', periods=120, freq='MS')
    return pd.Series(np.random.randint(-10, 10, len(date_range)), index=date_range).cumsum()

def clean_financial_data(df, column_name):
    """Specific cleaning step: removing currency symbols and converting to float."""
    return df[column_name].str.replace('$', '', regex=False).astype(float)