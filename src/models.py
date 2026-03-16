import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

class TimeSeriesAnalyzer:
    def __init__(self, data):
        """
        Initializes with a pandas Series.
        """
        self.data = data
        self.model_fit = None

    def perform_adf_test(self):
        """
        Checks for stationarity using the Augmented Dickey-Fuller test.
        Null Hypothesis (H0): Series is non-stationary.
        """
        result = adfuller(self.data)
        return {
            'statistics': result[0],
            'p_value': result[1],
            'is_stationary': result[1] <= 0.05
        }

    def fit_arima(self, order=(1, 1, 1)):
        """
        Fits an ARIMA model. 
        Task Type: Regression/Forecasting 
        """
        model = ARIMA(self.data, order=order)
        self.model_fit = model.fit()
        return self.model_fit