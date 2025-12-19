import pandas as pd
import numpy as np
from collections import deque
import tensorflow as tf

import joblib
import csv

# === CONFIG ===
WINDOW_SIZE = 10
ENTRY_THRESHOLD = 0.7  # model confidence threshold
STOP_LOSS = 5.0
epsilon = 1e-10

# === Load Historical Data ===
df = pd.read_csv("kalman_output.csv")  # Ensure it's sorted by time

# === Load Model and Scaler ===
model = tf.keras.models.load_model("trade_classifier_model.h5")
scaler = joblib.load("kalman_scaler.pkl")

# === Initialize State ===
kalman_window = deque(maxlen=WINDOW_SIZE)
position = 0
entry_price = None
trades = []

# === Backtest Loop ===
for i, row in df.iterrows():
    ltp = row["Measured_Price"]
    est_price = row["Estimated_Price"]
    velocity = row["Estimated_Velocity"]
    rel_diff = row["rel_diff"]

    tick_features = [est_price, velocity, rel_diff]
    kalman_window.append(tick_features)

    if len(kalman_window) < WINDOW_SIZE:
        continue

    # Model prediction
    input_window = np.array(kalman_window).reshape(1, 10, 3)
    scaled_window = scaler.transform(input_window.reshape(-1, 3)).reshape(1, 10, 3)
    prob = model.predict(scaled_window, verbose=0)[0][0]

    # === Entry Logic ===
    if position == 0:
        if (velocity < 0 and (-0.001 + epsilon) < rel_diff < (0 - epsilon)) and prob >= ENTRY_THRESHOLD:
            entry_price = ltp
            position = 1
            trades.append({
                "action": "BUY",
                "price": ltp,
                "est_price": est_price,
                "prob": prob,
                "tick": i
            })

    # === Exit Logic ===
    elif position == 1:
        # Exit condition: small positive reversal
        if (velocity > 0 and (0 + epsilon) < rel_diff < (0.001 - epsilon)):
            trades.append({
                "action": "SELL",
                "price": ltp,
                "est_price": est_price,
                "reason": "Signal Exit",
                "tick": i
            })
            position = 0
            entry_price = None

        # Stop loss condition
        elif ltp <= entry_price - STOP_LOSS:
            trades.append({
                "action": "SELL",
                "price": ltp,
                "est_price": est_price,
                "reason": "STOP LOSS",
                "tick": i
            })
            position = 0
            entry_price = None

# === Final Exit if Open ===
if position == 1:
    trades.append({
        "action": "SELL",
        "price": ltp,
        "est_price": est_price,
        "reason": "EOD Exit",
        "tick": i
    })

# === Save Trade Log ===
with open("simulated_trades.csv", mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["tick", "action", "price", "est_price", "prob", "reason"])
    writer.writeheader()
    for t in trades:
        writer.writerow(t)

# === PnL Calculation ===
pnl = 0
num_trades = 0
for i in range(0, len(trades)-1, 2):
    buy = trades[i]
    sell = trades[i+1]
    profit = sell["price"] - buy["price"]
    pnl += profit
    num_trades += 1

print("\n=== Backtest Results ===")
print(f"Total Trades: {num_trades}")
print(f"Total PnL: ₹{pnl:.2f}")
print(f"Avg PnL per trade: ₹{(pnl/num_trades) if num_trades else 0:.2f}")
print("Trade log saved to 'simulated_trades.csv'")
