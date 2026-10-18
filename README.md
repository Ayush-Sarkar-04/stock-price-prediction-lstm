

\# Stock Price Prediction using LSTM



A Python project for forecasting Tesla (TSLA) stock prices using historical market data and machine learning.



\## Current Features



\- Downloads historical stock data using Yahoo Finance.

\- Visualizes Tesla closing prices.

\- Scales closing prices using MinMaxScaler.

\- Creates 60-day sliding-window sequences.

\- Splits data chronologically into training and testing sets.

\- Evaluates a naive forecasting baseline.

\- Evaluates a Linear Regression baseline.

\- Calculates RMSE and MAPE.

\- Exports baseline results to CSV.



\## Technology Stack



\- Python

\- NumPy and Pandas

\- Matplotlib

\- Scikit-learn

\- TensorFlow

\- yfinance



\## Configuration



\- Ticker: TSLA

\- Start date: 2018-01-01

\- End date: 2024-12-31

\- Sequence length: 60 trading days

\- Test set: 20%



\## Baseline Results



| Model | RMSE | MAPE |

|---|---:|---:|

| Naive Baseline | 9.279 | 2.66% |

| Linear Regression | 9.652 | 2.93% |



These are results from the current local run. Results may vary with data and library versions.



\## Run Locally



Install dependencies:



```bash

pip install -r requirements.txt

```



Run the script:



```bash

python main.py

```



\## Limitations



\- The current scaler is fitted on the full dataset, which introduces data leakage.

\- The LSTM model is the next development stage.

\- Historical predictions do not guarantee future performance.

\- This project is for educational purposes and is not financial advice.



