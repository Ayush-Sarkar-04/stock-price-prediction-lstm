
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
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
EPOCHS = 20
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

close_prices = close_prices.dropna()

if len(close_prices) <= SEQ_LEN:
    raise RuntimeError("Not enough stock data to create sequences.")

print(f"Records downloaded: {len(close_prices)}")
print(
    f"Date range: {close_prices.index.min().date()} "
    f"to {close_prices.index.max().date()}"
)


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
# 4. CHRONOLOGICAL SPLIT AND SCALING
# ==========================================

prices = close_prices.to_numpy(dtype=float).reshape(-1, 1)

test_start = int(len(prices) * (1 - TEST_SIZE))

train_val_prices = prices[:test_start]
test_prices = prices[test_start:]

# Fit scaler only on the training/validation period.
scaler = MinMaxScaler(feature_range=(0, 1))
train_val_scaled = scaler.fit_transform(train_val_prices)
test_scaled = scaler.transform(test_prices)

# Validation is the final part of the pre-test period.
validation_start = int(
    len(train_val_scaled) * (1 - VALIDATION_SIZE)
)

# Training sequences can use only observations before
# the validation period.
X_train, y_train = [], []

for i in range(SEQ_LEN, validation_start):
    X_train.append(train_val_scaled[i - SEQ_LEN:i, 0])
    y_train.append(train_val_scaled[i, 0])

# Validation sequences use earlier observations as context.
X_val, y_val = [], []

for i in range(validation_start, len(train_val_scaled)):
    X_val.append(train_val_scaled[i - SEQ_LEN:i, 0])
    y_val.append(train_val_scaled[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

# Test sequences use the final 60 pre-test observations as
# context, then proceed through the held-out test period.
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

y_test_actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

test_dates = close_prices.index[test_start:]

print("\nLeakage-aware preprocessing completed.")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Testing samples: {len(X_test)}")


# ==========================================
# 5. METRICS
# ==========================================

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


# ==========================================
# 6. NAIVE BASELINE
# ==========================================

# Predict the next closing price as the most recent
# observed closing price.
naive_predictions_scaled = X_test[:, -1]

naive_predictions_actual = scaler.inverse_transform(
    naive_predictions_scaled.reshape(-1, 1)
).flatten()

naive_rmse, naive_mape = calculate_metrics(
    y_test_actual,
    naive_predictions_actual
)

print("\n--- Naive Baseline ---")
print(f"RMSE: ${naive_rmse:.3f}")
print(f"MAPE: {naive_mape:.2f}%")


# ==========================================
# 7. LINEAR REGRESSION BASELINE
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
# 8. BUILD THE LSTM MODEL
# ==========================================

X_train_lstm = X_train.reshape(-1, SEQ_LEN, 1)
X_val_lstm = X_val.reshape(-1, SEQ_LEN, 1)
X_test_lstm = X_test.reshape(-1, SEQ_LEN, 1)

model = Sequential([
    Input(shape=(SEQ_LEN, 1)),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss="mean_squared_error"
)

model.summary()


# ==========================================
# 9. TRAIN THE MODEL
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2
)

print("\nTraining LSTM model...")

history = model.fit(
    X_train_lstm,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_lstm, y_val),
    callbacks=[early_stopping, reduce_lr],
    shuffle=False,
    verbose=1
)


# ==========================================
# 10. FINAL TEST EVALUATION
# ==========================================

predictions = model.predict(X_test_lstm, verbose=0)

predictions_actual = scaler.inverse_transform(
    predictions
).flatten()

lstm_rmse, lstm_mape = calculate_metrics(
    y_test_actual,
    predictions_actual
)
errors = predictions_actual - y_test_actual

print("\n--- LSTM Prediction Diagnostics ---")
print(f"Mean prediction error: ${np.mean(errors):.2f}")
print(f"Mean absolute error: ${np.mean(np.abs(errors)):.2f}")
print(f"Maximum absolute error: ${np.max(np.abs(errors)):.2f}")
print(f"Prediction bias: {'Overprediction' if np.mean(errors) > 0 else 'Underprediction'}")

print("\n--- LSTM Results ---")
print(f"RMSE: ${lstm_rmse:.3f}")
print(f"MAPE: {lstm_mape:.2f}%")


# ==========================================
# 11. MODEL COMPARISON
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Naive Baseline",
        "Linear Regression",
        "LSTM"
    ],
    "RMSE": [
        naive_rmse,
        linear_rmse,
        lstm_rmse
    ],
    "MAPE (%)": [
        naive_mape,
        linear_mape,
        lstm_mape
    ]
})

print("\n--- Final Model Comparison ---")
print(results.to_string(index=False))

results.to_csv("model_comparison.csv", index=False)


# ==========================================
# 12. ACTUAL VS PREDICTED CHART
# ==========================================

plt.figure(figsize=(14, 7))
plt.plot(
    test_dates,
    y_test_actual,
    label="Actual Price",
    linewidth=2
)
plt.plot(
    test_dates,
    predictions_actual,
    label="LSTM Prediction",
    linewidth=1.5
)
plt.title(f"{TICKER} Actual vs Predicted Closing Prices")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction.png", dpi=150)
plt.close()


# ==========================================
# 13. TRAINING HISTORY CHART
# ==========================================

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)
plt.title("LSTM Training History")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
plt.close()

print("\nLSTM training completed.")
print("Saved: model_comparison.csv")
print("Saved: TSLA_closing_price.png")
print("Saved: prediction.png")
print("Saved: training_history.png")
