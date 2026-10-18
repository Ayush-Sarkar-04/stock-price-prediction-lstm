
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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

close_prices = close_prices.dropna().astype(float)
prices = close_prices.to_numpy()

print(f"Records downloaded: {len(prices)}")
print(
    f"Date range: {close_prices.index.min().date()} "
    f"to {close_prices.index.max().date()}"
)


# ==========================================
# 3. CALCULATE DAILY RETURNS
# ==========================================

# Return at day t = (price[t] / price[t-1]) - 1
returns = close_prices.pct_change().dropna()

return_values = returns.to_numpy().reshape(-1, 1)
return_dates = returns.index

test_start = int(len(return_values) * (1 - TEST_SIZE))

train_val_returns = return_values[:test_start]
test_returns = return_values[test_start:]

# Fit scaler on pre-test data only.
scaler = StandardScaler()
train_val_scaled = scaler.fit_transform(train_val_returns)
test_scaled = scaler.transform(test_returns)

# Chronological validation split.
validation_start = int(
    len(train_val_scaled) * (1 - VALIDATION_SIZE)
)

# Training sequences.
X_train, y_train = [], []

for i in range(SEQ_LEN, validation_start):
    X_train.append(train_val_scaled[i - SEQ_LEN:i, 0])
    y_train.append(train_val_scaled[i, 0])

# Validation sequences.
X_val, y_val = [], []

for i in range(validation_start, len(train_val_scaled)):
    X_val.append(train_val_scaled[i - SEQ_LEN:i, 0])
    y_val.append(train_val_scaled[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

# Test sequences use preceding returns as context.
test_input = np.concatenate(
    [train_val_scaled[-SEQ_LEN:], test_scaled],
    axis=0
)

X_test, y_test = [], []

for i in range(SEQ_LEN, len(test_input)):
    X_test.append(test_input[i - SEQ_LEN:i, 0])
    y_test.append(test_input[i, 0])

X_test = np.array(X_test)
y_test = np.array(y_test)

test_dates = return_dates[test_start:]

# For test return i:
# Previous closing price = prices[i]
# Actual closing price   = prices[i + 1]
test_previous_prices = prices[test_start:-1]
test_actual_prices = prices[test_start + 1:]

if not (
    len(test_previous_prices)
    == len(test_actual_prices)
    == len(test_returns)
    == len(X_test)
    == len(test_dates)
):
    raise RuntimeError("Test data alignment failed.")

# Verify price changes match the test returns.
reconstructed_returns = (
    test_actual_prices / test_previous_prices
) - 1

if not np.allclose(
    reconstructed_returns,
    test_returns.flatten(),
    rtol=1e-6,
    atol=1e-8
):
    raise RuntimeError("Test returns do not match prices.")

print("\nReturn preprocessing completed.")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Testing samples: {len(X_test)}")


# ==========================================
# 4. METRICS
# ==========================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    nonzero = actual != 0

    mape = np.mean(
        np.abs(
            (actual[nonzero] - predicted[nonzero])
            / actual[nonzero]
        )
    ) * 100

    return rmse, mape


# ==========================================
# 5. NAIVE PRICE BASELINE
# ==========================================

# Zero-return prediction:
# Tomorrow's closing price = today's closing price.
naive_predictions = test_previous_prices.copy()

naive_rmse, naive_mape = calculate_metrics(
    test_actual_prices,
    naive_predictions
)

print("\n--- Naive Price Baseline ---")
print(f"RMSE: ${naive_rmse:.6f}")
print(f"MAPE: {naive_mape:.6f}%")


# ==========================================
# 6. BUILD RETURNS-BASED LSTM
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

model.summary()


# ==========================================
# 7. TRAIN MODEL
# ==========================================

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
    X_train.reshape(-1, SEQ_LEN, 1),
    y_train,
    validation_data=(
        X_val.reshape(-1, SEQ_LEN, 1),
        y_val
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    shuffle=False,
    verbose=1
)


# ==========================================
# 8. CONVERT PREDICTED RETURNS TO PRICES
# ==========================================

predicted_returns_scaled = model.predict(
    X_test.reshape(-1, SEQ_LEN, 1),
    verbose=0
)

predicted_returns = scaler.inverse_transform(
    predicted_returns_scaled
).flatten()

# Predicted price = previous price * (1 + predicted return)
predicted_prices = (
    test_previous_prices * (1 + predicted_returns)
)

lstm_rmse, lstm_mape = calculate_metrics(
    test_actual_prices,
    predicted_prices
)

# Directional accuracy.
actual_returns = (
    test_actual_prices / test_previous_prices
) - 1

predicted_directions = predicted_returns > 0
actual_directions = actual_returns > 0

directional_accuracy = np.mean(
    predicted_directions == actual_directions
) * 100

# Simple directional reference: always predict up.
always_up_accuracy = np.mean(
    actual_directions
) * 100

print("\n--- Returns-Based LSTM Results ---")
print(f"RMSE: ${lstm_rmse:.6f}")
print(f"MAPE: {lstm_mape:.6f}%")
print(
    f"Directional Accuracy: "
    f"{directional_accuracy:.2f}%"
)
print(
    f"Always-Up Directional Baseline: "
    f"{always_up_accuracy:.2f}%"
)


# ==========================================
# 9. MODEL COMPARISON
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Naive Price Baseline",
        "Returns-Based LSTM"
    ],
    "RMSE": [
        naive_rmse,
        lstm_rmse
    ],
    "MAPE (%)": [
        naive_mape,
        lstm_mape
    ],
    "Directional Accuracy (%)": [
        np.nan,
        directional_accuracy
    ]
})

print("\n--- Model Comparison ---")
print(results.to_string(index=False))

results.to_csv(
    "returns_model_comparison.csv",
    index=False
)


# ==========================================
# 10. PREDICTION CHART
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
    predicted_prices,
    label="Returns-Based LSTM",
    linewidth=1.5
)

plt.title(
    f"{TICKER}: Actual vs Returns-Based LSTM"
)
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "returns_prediction.png",
    dpi=150
)
plt.close()


# ==========================================
# 11. TRAINING HISTORY
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
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "returns_training_history.png",
    dpi=150
)
plt.close()


# ==========================================
# 12. FINAL SUMMARY
# ==========================================

print("\n========== FINAL SUMMARY ==========")

print(f"Ticker: {TICKER}")
print(f"Test observations: {len(X_test)}")

print("\nNaive Baseline:")
print(f"  RMSE: ${naive_rmse:.4f}")
print(f"  MAPE: {naive_mape:.4f}%")

print("\nReturns-Based LSTM:")
print(f"  RMSE: ${lstm_rmse:.4f}")
print(f"  MAPE: {lstm_mape:.4f}%")
print(
    f"  Directional Accuracy: "
    f"{directional_accuracy:.2f}%"
)

print("\nAlways-Up Directional Baseline:")
print(f"  Accuracy: {always_up_accuracy:.2f}%")

if lstm_rmse < naive_rmse:
    print("\nLSTM outperformed the naive RMSE baseline.")
else:
    print("\nLSTM did not outperform the naive RMSE baseline.")

print("\nReturns experiment completed.")
print("Saved: returns_model_comparison.csv")
print("Saved: returns_prediction.png")
print("Saved: returns_training_history.png")
