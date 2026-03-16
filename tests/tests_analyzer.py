import pytest
import pandas as pd
import numpy as np
from src.models import TimeSeriesAnalyzer

def test_adf_output_format():
    # Setup
    data = pd.Series(np.random.randn(100))
    analyzer = TimeSeriesAnalyzer(data)
    
    # Execute 
    results = analyzer.perform_adf_test()
    
    # Verify 
    assert 'p_value' in results
    assert isinstance(results['is_stationary'], bool)

def test_model_fitting():
    data = pd.Series(np.random.randn(50)).cumsum()
    analyzer = TimeSeriesAnalyzer(data)
    summary = analyzer.fit_arima(order=(1,0,0))
    assert summary is not None