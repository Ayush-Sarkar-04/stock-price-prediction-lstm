
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
N_WINDOWS = 4
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

n = len(return_values)
initial_train_end = int(n * INITIAL_TRAIN_FRACTION)

if n - initial_train_end < N_WINDOWS:
    raise RuntimeError("Insufficient data for the requested windows.")

# Equal-sized windows; any remainder is assigned to the final window.
remaining = n - initial_train_end
window_size = remaining // N_WINDOWS

windows = []

for window_number in range(N_WINDOWS):
    start = initial_train_end + window_number * window_size

    if window_number == N_WINDOWS - 1:
        end = n
    else:
        end = start + window_size

    windows.append((start, end))

print(f"Price records: {len(prices)}")
print(f"Return observations: {n}")
print(f"Walk-forward windows: {len(windows)}")
print(f"Initial historical observations: {initial_train_end}")


# ==========================================
# 3. METRICS
# ==========================================

def evaluate_prices(actual, predicted):
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(
        np.abs((actual - predicted) / actual)
    ) * 100

    return rmse, mae, mape


def directional_accuracy(actual_returns, predicted_returns):
    actual_direction = np.asarray(actual_returns) > 0
    predicted_direction = np.asarray(predicted_returns) > 0

    return np.mean(
        actual_direction == predicted_direction
    ) * 100


# ==========================================
# 4. WALK-FORWARD EVALUATION
# ==========================================

all_results = []
all_predictions = []

for window_number, (test_start, test_end) in enumerate(
    windows, start=1
):
    print(
        f"\n{'=' * 55}\n"
        f"Window {window_number}/{N_WINDOWS}\n"
        f"{'=' * 55}"
    )

    # Only observations before the test window are used for fitting.
    history_returns = return_values[:test_start]
    test_returns = return_values[test_start:test_end]

    previous_prices = prices[test_start:test_end]
    actual_prices = prices[test_start + 1:test_end + 1]
    dates = return_dates[test_start:test_end]

    if not (
        len(test_returns)
        == len(previous_prices)
        == len(actual_prices)
        == len(dates)
    ):
        raise RuntimeError("Test data alignment failed.")

    # Reserve the latest 15% of historical returns for validation.
    validation_start = int(len(history_returns) * 0.85)

    training_returns = history_returns[:validation_start]
    validation_returns = history_returns[validation_start:]

    if len(training_returns) <= SEQ_LEN:
        raise RuntimeError("Insufficient training observations.")

    # Fit the scaler only on the training partition.
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
        test_returns.reshape(-1, 1)
    ).flatten()

    # ------------------------------
    # A. NAIVE ZERO-RETURN BASELINE
    # ------------------------------

    naive_returns = np.zeros(len(test_returns))
    naive_prices = previous_prices.copy()

    # ------------------------------
    # B. HISTORICAL MEAN RETURN
    # ------------------------------

    historical_mean = np.mean(history_returns)

    mean_returns = np.full(
        len(test_returns),
        historical_mean
    )

    mean_prices = previous_prices * (1 + mean_returns)

    # ------------------------------
    # C. LINEAR REGRESSION
    # ------------------------------

    X_lr, y_lr = [], []

    for i in range(SEQ_LEN, len(training_returns)):
        X_lr.append(train_scaled[i - SEQ_LEN:i])
        y_lr.append(training_returns[i])

    X_lr = np.asarray(X_lr)
    y_lr = np.asarray(y_lr)

    linear_model = LinearRegression()
    linear_model.fit(X_lr, y_lr)

    # One-step forecasts. Each observed test return is added to
    # context only after its prediction has been generated.
    lr_context = history_scaled[-SEQ_LEN:].tolist()
    linear_returns = []

    for scaled_return in test_scaled:
        features = np.asarray(
            lr_context[-SEQ_LEN:]
        ).reshape(1, -1)

        predicted_return = linear_model.predict(features)[0]
        linear_returns.append(predicted_return)
        lr_context.append(scaled_return)

    linear_returns = np.asarray(linear_returns)
    linear_prices = previous_prices * (1 + linear_returns)

    # ------------------------------
    # D. RETURNS-BASED LSTM
    # ------------------------------

    X_train, y_train = [], []

    for i in range(SEQ_LEN, len(train_scaled)):
        X_train.append(train_scaled[i - SEQ_LEN:i])
        y_train.append(train_scaled[i])

    X_train = np.asarray(
        X_train, dtype=np.float32
    ).reshape(-1, SEQ_LEN, 1)

    y_train = np.asarray(y_train, dtype=np.float32)

    # Validation sequences include the preceding training context.
    validation_input = np.concatenate([
        train_scaled[-SEQ_LEN:],
        validation_scaled
    ])

    X_val, y_val = [], []

    for i in range(SEQ_LEN, len(validation_input)):
        X_val.append(validation_input[i - SEQ_LEN:i])
        y_val.append(validation_input[i])

    X_val = np.asarray(
        X_val, dtype=np.float32
    ).reshape(-1, SEQ_LEN, 1)

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

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    print("Training LSTM...")

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=False,
        verbose=0
    )

    # Sequential one-step-ahead forecasts.
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

        # Add the observed return only after the prediction.
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

    for name, (predicted_prices, predicted_returns) in (
        model_predictions.items()
    ):
        rmse, mae, mape = evaluate_prices(
            actual_prices,
            predicted_prices
        )

        direction = directional_accuracy(
            test_returns,
            predicted_returns
        )

        all_results.append({
            "Window": window_number,
            "Start Date": dates[0].date(),
            "End Date": dates[-1].date(),
            "Model": name,
            "RMSE ($)": rmse,
            "MAE ($)": mae,
            "MAPE (%)": mape,
            "Directional Accuracy (%)": direction
        })

        for i in range(len(dates)):
            all_predictions.append({
                "Window": window_number,
                "Date": dates[i],
                "Model": name,
                "Actual Price": actual_prices[i],
                "Predicted Price": predicted_prices[i],
                "Actual Return": test_returns[i],
                "Predicted Return": predicted_returns[i]
            })

    print(
        f"Evaluated {len(test_returns)} observations: "
        f"{dates[0].date()} to {dates[-1].date()}"
    )


# ==========================================
# 5. SAVE RESULTS
# ==========================================

window_results = pd.DataFrame(all_results)
predictions_df = pd.DataFrame(all_predictions)

window_results.to_csv(
    "walk_forward_v2_window_results.csv",
    index=False
)

predictions_df.to_csv(
    "walk_forward_v2_predictions.csv",
    index=False
)


# ==========================================
# 6. POOLED METRICS
# ==========================================

pooled_rows = []

for model_name, group in predictions_df.groupby("Model"):
    actual_prices = group["Actual Price"].to_numpy()
    predicted_prices = group["Predicted Price"].to_numpy()

    actual_returns = group["Actual Return"].to_numpy()
    predicted_returns = group["Predicted Return"].to_numpy()

    rmse, mae, mape = evaluate_prices(
        actual_prices,
        predicted_prices
    )

    direction = directional_accuracy(
        actual_returns,
        predicted_returns
    )

    pooled_rows.append({
        "Model": model_name,
        "Observations": len(group),
        "Pooled RMSE ($)": rmse,
        "Pooled MAE ($)": mae,
        "Pooled MAPE (%)": mape,
        "Pooled Directional Accuracy (%)": direction
    })

pooled_results = pd.DataFrame(pooled_rows).sort_values(
    "Pooled RMSE ($)"
)

pooled_results.to_csv(
    "walk_forward_v2_pooled_summary.csv",
    index=False
)

print("\n========== POOLED RESULTS ==========")
print(pooled_results.to_string(index=False))


# ==========================================
# 7. PER-WINDOW SUMMARY
# ==========================================

print("\n========== PER-WINDOW RMSE ==========")

rmse_pivot = window_results.pivot(
    index="Window",
    columns="Model",
    values="RMSE ($)"
)

print(rmse_pivot.to_string())

rmse_pivot.to_csv(
    "walk_forward_v2_window_rmse.csv"
)


# ==========================================
# 8. PLOT RMSE BY WINDOW
# ==========================================

ax = rmse_pivot.plot(
    kind="bar",
    figsize=(13, 7)
)

ax.set_title("Walk-Forward V2: RMSE by Window")
ax.set_xlabel("Test Window")
ax.set_ylabel("RMSE (USD)")
ax.grid(axis="y")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "walk_forward_v2_rmse.png",
    dpi=150
)

plt.close()


# ==========================================
# 9. FINAL SUMMARY
# ==========================================

print("\n========== FINAL SUMMARY ==========")
print(f"Ticker: {TICKER}")
print(f"Number of windows: {N_WINDOWS}")
print(f"Total test observations: {len(predictions_df) // 4}")

best = pooled_results.iloc[0]

print(f"\nLowest pooled RMSE: {best['Model']}")
print(f"Pooled RMSE: ${best['Pooled RMSE ($)']:.4f}")
print(f"Pooled MAE: ${best['Pooled MAE ($)']:.4f}")
print(f"Pooled MAPE: {best['Pooled MAPE (%)']:.4f}%")
print(
    "Pooled directional accuracy: "
    f"{best['Pooled Directional Accuracy (%)']:.2f}%"
)

print("\nSaved files:")
print("- walk_forward_v2_window_results.csv")
print("- walk_forward_v2_predictions.csv")
print("- walk_forward_v2_pooled_summary.csv")
print("- walk_forward_v2_window_rmse.csv")
print("- walk_forward_v2_rmse.png")

print("\nWalk-forward V2 experiment completed.")
