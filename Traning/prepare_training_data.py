import pandas as pd
import numpy as np
from collections import deque

WINDOW_SIZE = 10
FUTURE_WINDOW = 5  # Number of ticks ahead to evaluate trade outcome
PROFIT_THRESHOLD = 0.5  # Minimum profit (in price units) to consider a "profitable trade"

# === Step 1: Load Kalman Output ===
df = pd.read_csv("kalman_output.csv")  # Columns: Measured_Price, Estimated_Price, Estimated_Velocity, rel_diff

# === Step 2: Create Input Sequences and Labels ===
X, y = [], []

tick_window = deque(maxlen=WINDOW_SIZE)

for i in range(len(df) - FUTURE_WINDOW):
    # Extract current tick's Kalman features
    row = df.iloc[i]
    features = [row["Estimated_Price"], row["Estimated_Velocity"], row["rel_diff"]]
    tick_window.append(features)

    if len(tick_window) == WINDOW_SIZE:
        # Predict trade outcome over future 5 ticks
        future_prices = df.iloc[i + 1:i + 1 + FUTURE_WINDOW]["Measured_Price"].values
        entry_price = row["Measured_Price"]
        max_future_price = np.max(future_prices)

        profit = max_future_price - entry_price
        label = 1 if profit >= PROFIT_THRESHOLD else 0

        X.append(list(tick_window))
        y.append(label)

# === Step 3: Convert and Save ===
X = np.array(X)
y = np.array(y)

print(f"Created {len(X)} samples, each of shape {X[0].shape}")
np.savez("trade_classifier_data.npz", X=X, y=y)
