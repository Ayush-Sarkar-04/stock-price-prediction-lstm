
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
INITIAL_TRAIN_FRACTION = 0.60
WINDOW_FRACTION = 0.10
EPOCHS = 20
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

returns = close_prices.pct_change().dropna()
return_values = returns.to_numpy()
return_dates = returns.index

if len(return_values) < 500:
    raise RuntimeError("Not enough data for walk-forward evaluation.")

print(f"Price records: {len(prices)}")
print(f"Return observations: {len(return_values)}")
print(
    f"Date range: {price_dates.min().date()} "
    f"to {price_dates.max().date()}"
)


# ==========================================
# 3. METRICS
# ==========================================

def evaluate_prices(actual_prices, predicted_prices):
    actual_prices = np.asarray(actual_prices).flatten()
    predicted_prices = np.asarray(predicted_prices).flatten()

    rmse = np.sqrt(
        mean_squared_error(actual_prices, predicted_prices)
    )
    mae = mean_absolute_error(actual_prices, predicted_prices)
    mape = np.mean(
        np.abs((actual_prices - predicted_prices) / actual_prices)
    ) * 100

    return rmse, mae, mape


def directional_accuracy(actual_returns, predicted_returns):
    actual_direction = np.asarray(actual_returns) > 0
    predicted_direction = np.asarray(predicted_returns) > 0

    return np.mean(
        actual_direction == predicted_direction
    ) * 100


# ==========================================
# 4. DEFINE WALK-FORWARD WINDOWS
# ==========================================

n = len(return_values)

initial_train_end = int(n * INITIAL_TRAIN_FRACTION)
window_size = int(n * WINDOW_FRACTION)

if initial_train_end <= SEQ_LEN or window_size < 20:
    raise RuntimeError("Invalid walk-forward configuration.")

windows = []
start = initial_train_end

while start < n:
    end = min(start + window_size, n)
    windows.append((start, end))
    start = end

print(f"\nWalk-forward windows: {len(windows)}")
print(f"Initial training observations: {initial_train_end}")
print(f"Approximate observations per test window: {window_size}")


# ==========================================
# 5. WALK-FORWARD EVALUATION
# ==========================================

all_results = []
all_predictions = []

for window_number, (test_start, test_end) in enumerate(windows, start=1):

    print(
        f"\n{'=' * 55}\n"
        f"Window {window_number}/{len(windows)}\n"
        f"{'=' * 55}"
    )

    # Expanding training window:
    # all observations before this test window are historical.
    history_returns = return_values[:test_start]

    window_test_returns = return_values[test_start:test_end]

    # Return index i corresponds to prices[i] -> prices[i + 1].
    previous_prices = prices[test_start:test_end]
    actual_prices = prices[test_start + 1:test_end + 1]
    dates = return_dates[test_start:test_end]

    if not (
        len(window_test_returns)
        == len(previous_prices)
        == len(actual_prices)
        == len(dates)
    ):
        raise RuntimeError("Walk-forward alignment failed.")

    # Split historical observations chronologically:
    # final 15% is used for validation.
    validation_start = int(
        len(history_returns) * 0.85
    )

    training_returns = history_returns[:validation_start]
    validation_returns = history_returns[validation_start:]

    if len(training_returns) <= SEQ_LEN:
        raise RuntimeError("Insufficient training history.")

    # Fit scaler only on training returns.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(
        training_returns.reshape(-1, 1)
    ).flatten()

    validation_scaled = scaler.transform(
        validation_returns.reshape(-1, 1)
    ).flatten()

    history_scaled = scaler.transform(
        history_returns.reshape(-1, 1)
    ).flatten()

    test_scaled = scaler.transform(
        window_test_returns.reshape(-1, 1)
    ).flatten()

    # ------------------------------
    # A. NAIVE ZERO-RETURN BASELINE
    # ------------------------------

    naive_returns = np.zeros(len(window_test_returns))
    naive_prices = previous_prices.copy()

    # ------------------------------
    # B. HISTORICAL MEAN RETURN
    # ------------------------------

    mean_return = np.mean(history_returns)

    mean_returns = np.full(
        len(window_test_returns),
        mean_return
    )

    mean_prices = previous_prices * (1 + mean_returns)

    # ------------------------------
    # C. LINEAR REGRESSION
    # ------------------------------

    X_lr, y_lr = [], []

    # Fit features and targets using training observations only.
    for i in range(SEQ_LEN, len(training_returns)):
        X_lr.append(
            train_scaled[i - SEQ_LEN:i]
        )
        y_lr.append(
            training_returns[i]
        )

    X_lr = np.asarray(X_lr)
    y_lr = np.asarray(y_lr)

    # Use raw return targets consistently.
    linear_model = LinearRegression()
    linear_model.fit(X_lr, y_lr)

    # Test sequences use historical returns and earlier test returns.
    lr_context = history_scaled[-SEQ_LEN:].tolist()
    linear_returns = []

    for scaled_return in test_scaled:
        features = np.asarray(
            lr_context[-SEQ_LEN:]
        ).reshape(1, -1)

        predicted_return = linear_model.predict(features)[0]
        linear_returns.append(predicted_return)

        # Append the observed return only after forecasting that date.
        lr_context.append(scaled_return)

    linear_returns = np.asarray(linear_returns)
    linear_prices = previous_prices * (1 + linear_returns)

    # ------------------------------
    # D. RETURNS-BASED LSTM
    # ------------------------------

    X_train, y_train = [], []

    for i in range(SEQ_LEN, len(train_scaled)):
        X_train.append(
            train_scaled[i - SEQ_LEN:i]
        )
        y_train.append(
            train_scaled[i]
        )

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    validation_input = np.concatenate([
        train_scaled[-SEQ_LEN:],
        validation_scaled
    ])

    X_val, y_val = [], []

    for i in range(SEQ_LEN, len(validation_input)):
        X_val.append(
            validation_input[i - SEQ_LEN:i]
        )
        y_val.append(
            validation_input[i]
        )

    X_val = np.asarray(X_val, dtype=np.float32).reshape(
        -1, SEQ_LEN, 1
    )
    y_val = np.asarray(y_val, dtype=np.float32)

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
        patience=3,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    print("Training LSTM...")

    model.fit(
        X_train.reshape(-1, SEQ_LEN, 1),
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        shuffle=False,
        verbose=0
    )

    # Sequential one-step-ahead evaluation.
    # Each forecast uses only returns observable before that forecast.
    lstm_context = history_scaled[-SEQ_LEN:].tolist()
    lstm_returns = []

    for scaled_return in test_scaled:
        sequence = np.asarray(
            lstm_context[-SEQ_LEN:],
            dtype=np.float32
        ).reshape(1, SEQ_LEN, 1)

        predicted_scaled = model.predict(
            sequence,
            verbose=0
        )[0, 0]

        predicted_return = scaler.inverse_transform(
            np.array([[predicted_scaled]])
        )[0, 0]

        lstm_returns.append(predicted_return)

        # Add the observed return after generating the forecast.
        lstm_context.append(scaled_return)

    lstm_returns = np.asarray(lstm_returns)
    lstm_prices = previous_prices * (1 + lstm_returns)

    # ------------------------------
    # E. EVALUATE WINDOW
    # ------------------------------

    model_predictions = {
        "Naive Zero-Return": (naive_prices, naive_returns),
        "Historical Mean Return": (mean_prices, mean_returns),
        "Linear Regression": (linear_prices, linear_returns),
        "Returns-Based LSTM": (lstm_prices, lstm_returns)
    }

    for model_name, (predicted_prices, predicted_returns) in model_predictions.items():

        rmse, mae, mape = evaluate_prices(
            actual_prices,
            predicted_prices
        )

        direction = directional_accuracy(
            window_test_returns,
            predicted_returns
        )

        all_results.append({
            "Window": window_number,
            "Start Date": dates[0].date(),
            "End Date": dates[-1].date(),
            "Model": model_name,
            "RMSE ($)": rmse,
            "MAE ($)": mae,
            "MAPE (%)": mape,
            "Directional Accuracy (%)": direction
        })

        for i in range(len(dates)):
            all_predictions.append({
                "Window": window_number,
                "Date": dates[i],
                "Model": model_name,
                "Actual Price": actual_prices[i],
                "Predicted Price": predicted_prices[i],
                "Actual Return": window_test_returns[i],
                "Predicted Return": predicted_returns[i]
            })

    print(
        f"Evaluated {len(window_test_returns)} observations: "
        f"{dates[0].date()} to {dates[-1].date()}"
    )


# ==========================================
# 6. SAVE WINDOW RESULTS
# ==========================================

window_results = pd.DataFrame(all_results)

window_results.to_csv(
    "walk_forward_window_results.csv",
    index=False
)

predictions_df = pd.DataFrame(all_predictions)

predictions_df.to_csv(
    "walk_forward_predictions.csv",
    index=False
)


# ==========================================
# 7. AGGREGATE RESULTS
# ==========================================

# Each window is evaluated separately.
# Report the mean and standard deviation across windows.
summary = (
    window_results
    .groupby("Model")
    .agg({
        "RMSE ($)": ["mean", "std"],
        "MAE ($)": ["mean", "std"],
        "MAPE (%)": ["mean", "std"],
        "Directional Accuracy (%)": ["mean", "std"]
    })
)

summary.columns = [
    f"{metric} {stat.upper()}"
    for metric, stat in summary.columns
]

summary = summary.reset_index()
summary = summary.sort_values("RMSE ($) MEAN")

print("\n========== WALK-FORWARD SUMMARY ==========")
print(summary.to_string(index=False))

summary.to_csv(
    "walk_forward_summary.csv",
    index=False
)


# ==========================================
# 8. PLOT WINDOW RMSE
# ==========================================

rmse_pivot = window_results.pivot(
    index="Window",
    columns="Model",
    values="RMSE ($)"
)

ax = rmse_pivot.plot(
    kind="bar",
    figsize=(13, 7)
)

ax.set_title("Walk-Forward RMSE by Test Window")
ax.set_xlabel("Test Window")
ax.set_ylabel("RMSE (USD)")
ax.grid(axis="y")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "walk_forward_rmse.png",
    dpi=150
)

plt.close()


# ==========================================
# 9. FINAL REPORT
# ==========================================

print("\n========== FINAL REPORT ==========")
print(f"Ticker: {TICKER}")
print(f"Walk-forward windows: {len(windows)}")
print(f"Total test observations: {sum(end - start for start, end in windows)}")

best_row = summary.iloc[0]

print(f"\nLowest average RMSE: {best_row['Model']}")
print(f"Mean RMSE: ${best_row['RMSE ($) MEAN']:.4f}")
print(f"RMSE standard deviation: ${best_row['RMSE ($) STD']:.4f}")

print("\nSaved files:")
print("- walk_forward_window_results.csv")
print("- walk_forward_predictions.csv")
print("- walk_forward_summary.csv")
print("- walk_forward_rmse.png")

print("\nWalk-forward experiment completed.")
