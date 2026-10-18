
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ==========================================
# 1. CONFIGURATION
# ==========================================

TICKER = "TSLA"
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

SEQ_LEN = 60
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.15
EPOCHS = 30
BATCH_SIZE = 32

np.random.seed(42)
tf.random.set_seed(42)


# ==========================================
# 2. DOWNLOAD DATA
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

close_prices = close_prices.dropna().astype(float)

prices = close_prices.to_numpy()
price_dates = close_prices.index

print(f"Records downloaded: {len(prices)}")
print(
    f"Date range: {price_dates.min().date()} "
    f"to {price_dates.max().date()}"
)


# ==========================================
# 3. CALCULATE RETURNS AND SPLIT DATA
# ==========================================

returns = close_prices.pct_change().dropna()

return_values = returns.to_numpy()
return_dates = returns.index

test_start = int(len(return_values) * (1 - TEST_SIZE))
validation_start = int(
    test_start * (1 - VALIDATION_SIZE)
)

# Historical returns before the test period.
pretest_returns = return_values[:test_start]

# Final validation period within pre-test data.
train_returns = return_values[:validation_start]
validation_returns = return_values[validation_start:test_start]
test_returns = return_values[test_start:]

# Price alignment:
# return_values[i] = prices[i + 1] / prices[i] - 1
test_previous_prices = prices[test_start:-1]
test_actual_prices = prices[test_start + 1:]
test_dates = return_dates[test_start:]

if not (
    len(test_returns)
    == len(test_previous_prices)
    == len(test_actual_prices)
    == len(test_dates)
):
    raise RuntimeError("Test data alignment failed.")

if not np.allclose(
    test_returns,
    test_actual_prices / test_previous_prices - 1,
    rtol=1e-6,
    atol=1e-8
):
    raise RuntimeError("Test returns do not match prices.")

# Scaler is fitted only on training returns.
scaler = StandardScaler()
train_scaled = scaler.fit_transform(
    train_returns.reshape(-1, 1)
).flatten()

# Validation and test returns are transformed with the same scaler.
validation_scaled = scaler.transform(
    validation_returns.reshape(-1, 1)
).flatten()

pretest_scaled = scaler.transform(
    pretest_returns.reshape(-1, 1)
).flatten()

test_scaled = scaler.transform(
    test_returns.reshape(-1, 1)
).flatten()

print("\nData split completed.")
print(f"Training returns: {len(train_returns)}")
print(f"Validation returns: {len(validation_returns)}")
print(f"Test returns: {len(test_returns)}")


# ==========================================
# 4. CREATE SEQUENCES
# ==========================================

# Each sequence contains 60 preceding returns.
# The target is the following return.

def create_sequences(values, sequence_length):
    X, y = [], []

    for i in range(sequence_length, len(values)):
        X.append(values[i - sequence_length:i])
        y.append(values[i])

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32)
    )


# Training sequences use only training data.
X_train, y_train = create_sequences(
    train_scaled,
    SEQ_LEN
)

# Validation sequences can use preceding training returns as context.
validation_input = np.concatenate([
    train_scaled[-SEQ_LEN:],
    validation_scaled
])

X_val, y_val = create_sequences(
    validation_input,
    SEQ_LEN
)

# Test sequences use the 60 preceding returns as context.
test_input = np.concatenate([
    pretest_scaled[-SEQ_LEN:],
    test_scaled
])

X_test, y_test = create_sequences(
    test_input,
    SEQ_LEN
)

X_train = X_train.reshape(-1, SEQ_LEN, 1)
X_val = X_val.reshape(-1, SEQ_LEN, 1)
X_test = X_test.reshape(-1, SEQ_LEN, 1)

if len(X_test) != len(test_returns):
    raise RuntimeError("Test sequence alignment failed.")

print(f"Training sequences: {len(X_train)}")
print(f"Validation sequences: {len(X_val)}")
print(f"Testing sequences: {len(X_test)}")


# ==========================================
# 5. METRICS
# ==========================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mae = mean_absolute_error(actual, predicted)

    # Price MAPE is meaningful for these positive stock prices.
    nonzero = actual != 0

    mape = np.mean(
        np.abs(
            (actual[nonzero] - predicted[nonzero])
            / actual[nonzero]
        )
    ) * 100

    return rmse, mae, mape


def directional_accuracy(actual_returns, predicted_returns):
    actual_direction = np.asarray(actual_returns) > 0
    predicted_direction = np.asarray(predicted_returns) > 0

    return np.mean(
        actual_direction == predicted_direction
    ) * 100


# ==========================================
# 6. BASELINE FORECASTS
# ==========================================

# Baseline A: predict zero return.
naive_returns = np.zeros(len(test_returns))

# Baseline B: predict the historical mean return.
# Mean uses pre-test observations only.
historical_mean = np.mean(pretest_returns)

mean_returns = np.full(
    len(test_returns),
    historical_mean
)

# Convert predicted returns into prices.
naive_prices = test_previous_prices * (1 + naive_returns)
mean_prices = test_previous_prices * (1 + mean_returns)

print("\n--- Baseline Forecasts ---")
print(f"Historical mean daily return: {historical_mean:.8f}")


# ==========================================
# 7. LINEAR REGRESSION ON RETURNS
# ==========================================

# Use the previous 60 returns as features.
# Training targets are the following day's returns.
X_lr_train, y_lr_train = [], []

for i in range(SEQ_LEN, len(train_scaled)):
    X_lr_train.append(train_scaled[i - SEQ_LEN:i])
    y_lr_train.append(train_returns[i])

X_lr_train = np.array(X_lr_train)
y_lr_train = np.array(y_lr_train)

# Test inputs are constructed using the same chronological context.
X_lr_test = []

for i in range(SEQ_LEN, len(test_input)):
    X_lr_test.append(test_input[i - SEQ_LEN:i])

X_lr_test = np.array(X_lr_test)

linear_model = LinearRegression()
linear_model.fit(X_lr_train, y_lr_train)

linear_returns = linear_model.predict(X_lr_test)

linear_prices = (
    test_previous_prices * (1 + linear_returns)
)

print("\nLinear Regression completed.")


# ==========================================
# 8. RETURNS-BASED LSTM
# ==========================================

model = Sequential([
    Input(shape=(SEQ_LEN, 1)),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0003
    ),
    loss="mean_squared_error"
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

print("\nTraining returns-based LSTM...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr],
    shuffle=False,
    verbose=1
)

predicted_returns_scaled = model.predict(
    X_test,
    verbose=0
).flatten()

lstm_returns = scaler.inverse_transform(
    predicted_returns_scaled.reshape(-1, 1)
).flatten()

lstm_prices = (
    test_previous_prices * (1 + lstm_returns)
)


# ==========================================
# 9. EVALUATE ALL MODELS
# ==========================================

models = {
    "Naive Zero-Return": (naive_prices, naive_returns),
    "Historical Mean Return": (mean_prices, mean_returns),
    "Linear Regression": (linear_prices, linear_returns),
    "Returns-Based LSTM": (lstm_prices, lstm_returns)
}

rows = []

for name, (predicted_prices, predicted_returns) in models.items():
    rmse, mae, mape = calculate_metrics(
        test_actual_prices,
        predicted_prices
    )

    direction = directional_accuracy(
        test_returns,
        predicted_returns
    )

    rows.append({
        "Model": name,
        "RMSE ($)": rmse,
        "MAE ($)": mae,
        "MAPE (%)": mape,
        "Directional Accuracy (%)": direction
    })

results = pd.DataFrame(rows)
results = results.sort_values("RMSE ($)")

print("\n========== MODEL COMPARISON ==========")
print(results.to_string(index=False))

results.to_csv(
    "returns_baseline_comparison.csv",
    index=False
)


# ==========================================
# 10. PLOT PRICE PREDICTIONS
# ==========================================

plt.figure(figsize=(14, 7))

plt.plot(
    test_dates,
    test_actual_prices,
    label="Actual Price",
    linewidth=2
)

plt.plot(
    test_dates,
    naive_prices,
    label="Naive Zero-Return",
    linewidth=1
)

plt.plot(
    test_dates,
    linear_prices,
    label="Linear Regression",
    linewidth=1
)

plt.plot(
    test_dates,
    lstm_prices,
    label="Returns-Based LSTM",
    linewidth=1
)

plt.title(f"{TICKER}: Baseline and LSTM Comparison")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "returns_baseline_comparison.png",
    dpi=150
)

plt.close()


# ==========================================
# 11. PLOT TRAINING HISTORY
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Returns-Based LSTM Training History")
plt.xlabel("Epoch")
plt.ylabel("Scaled Return MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "returns_baseline_training_history.png",
    dpi=150
)

plt.close()


# ==========================================
# 12. FINAL SUMMARY
# ==========================================

print("\n========== FINAL SUMMARY ==========")
print(f"Ticker: {TICKER}")
print(f"Test observations: {len(test_returns)}")

best_model = results.iloc[0]

print(f"\nLowest test RMSE: {best_model['Model']}")
print(f"RMSE: ${best_model['RMSE ($)']:.4f}")
print(f"MAE: ${best_model['MAE ($)']:.4f}")
print(f"MAPE: {best_model['MAPE (%)']:.4f}%")
print(
    "Directional accuracy: "
    f"{best_model['Directional Accuracy (%)']:.2f}%"
)

print("\nSaved files:")
print("- returns_baseline_comparison.csv")
print("- returns_baseline_comparison.png")
print("- returns_baseline_training_history.png")

print("\nExperiment completed.")
