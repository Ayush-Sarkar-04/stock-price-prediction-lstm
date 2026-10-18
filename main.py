
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ==========================================
# 1. CONFIGURATION
# ==========================================

TICKER = "TSLA"
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

SEQ_LEN = 60
TEST_SIZE = 0.2


# ==========================================
# 2. DOWNLOAD STOCK DATA
# ==========================================

print(f"Downloading {TICKER} stock data...")

data = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
    progress=False
)

if data.empty:
    raise RuntimeError(
        "No stock data downloaded. Check your internet connection."
    )

if isinstance(data.columns, pd.MultiIndex):
    close_prices = data["Close"][TICKER]
else:
    close_prices = data["Close"]

close_prices = close_prices.dropna()

print(f"Records downloaded: {len(close_prices)}")
print(f"Date range: {close_prices.index.min().date()} "
      f"to {close_prices.index.max().date()}")


# ==========================================
# 3. PLOT CLOSING PRICES
# ==========================================

plt.figure(figsize=(12, 6))
plt.plot(close_prices.index, close_prices.values)
plt.title(f"{TICKER} Closing Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.savefig("TSLA_closing_price.png", dpi=150)
plt.close()


# ==========================================
# 4. SCALE THE DATA
# ==========================================

# Faithful to the original notebook:
# the scaler is fitted on the complete dataset.
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(
    close_prices.to_numpy().reshape(-1, 1)
)

print("\nData scaling completed.")


# ==========================================
# 5. CREATE 60-DAY SEQUENCES
# ==========================================

X = []
y = []

for i in range(SEQ_LEN, len(scaled_data)):
    X.append(scaled_data[i - SEQ_LEN:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

print(f"Total sequences: {len(X)}")
print(f"Input shape: {X.shape}")


# ==========================================
# 6. CHRONOLOGICAL TRAIN-TEST SPLIT
# ==========================================

split_index = int(len(X) * (1 - TEST_SIZE))

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ==========================================
# 7. NAIVE BASELINE
# ==========================================

# Predict tomorrow's price as today's closing price.
naive_predictions = X_test[:, -1]

naive_predictions_actual = scaler.inverse_transform(
    naive_predictions.reshape(-1, 1)
).flatten()

y_test_actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()


def calculate_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    nonzero = actual != 0
    mape = np.mean(
        np.abs(
            (actual[nonzero] - predicted[nonzero])
            / actual[nonzero]
        )
    ) * 100

    return rmse, mape


naive_rmse, naive_mape = calculate_metrics(
    y_test_actual,
    naive_predictions_actual
)

print("\n--- Naive Baseline ---")
print(f"RMSE: ${naive_rmse:.3f}")
print(f"MAPE: {naive_mape:.2f}%")


# ==========================================
# 8. LINEAR REGRESSION BASELINE
# ==========================================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_predictions_actual = scaler.inverse_transform(
    linear_predictions.reshape(-1, 1)
).flatten()

linear_rmse, linear_mape = calculate_metrics(
    y_test_actual,
    linear_predictions_actual
)

print("\n--- Linear Regression Baseline ---")
print(f"RMSE: ${linear_rmse:.3f}")
print(f"MAPE: {linear_mape:.2f}%")


# ==========================================
# 9. BASELINE COMPARISON
# ==========================================

results = pd.DataFrame({
    "Model": ["Naive Baseline", "Linear Regression"],
    "RMSE": [naive_rmse, linear_rmse],
    "MAPE (%)": [naive_mape, linear_mape]
})

print("\n--- Baseline Model Comparison ---")
print(results.to_string(index=False))

results.to_csv("baseline_results.csv", index=False)

print("\nBaseline results saved to baseline_results.csv")
print("Closing-price chart saved to TSLA_closing_price.png")
