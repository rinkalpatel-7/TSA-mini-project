# Time Series Analysis (TSA) Research Project
**Applicant:** Rinkal Patel  


## Project Overview
This project implements a modular pipeline for Time Series Analysis, focusing on stationarity testing and autoregressive modeling. It uses synthetic data to demonstrate statistical rigor and reproducibility in ML engineering.

## Engineering Features
* **Modular Code:** Core logic is encapsulated in `src/models.py`.
* **Reproducibility:** Global seeds are set in `main.py`; environment managed via `requirements.txt`.
* **Testing:** Unit tests implemented in `tests/` using `pytest`.
* **Analysis:** Seasonal decomposition and residual analysis located in `analysis/`.

## Setup & Execution
1. Create environment: `python -m venv venv`
2. Activate: `.\venv\Scripts\activate` (Windows)
3. Install: `pip install -r requirements.txt`
4. Run: `python main.py`

## Technical Details
* **Task:** Sequence Modeling / Regression
* **Model:** ARIMA(1, 1, 1)
* **Optimization Metric:** Log-Likelihood