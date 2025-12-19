import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import joblib
import os

import tensorflow as tf
from tensorflow import keras

Sequential = keras.models.Sequential
Conv1D = keras.layers.Conv1D
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
EarlyStopping = keras.callbacks.EarlyStopping


# === Step 1: Load .npz dataset ===
data_file = "trade_classifier_data.npz"
if not os.path.exists(data_file):
    raise FileNotFoundError(f"{data_file} not found.")

data = np.load(data_file)
X = data['X']  # Shape: (num_samples, 10, 3)
y = data['y']  # Shape: (num_samples,)

print(f"Loaded dataset: {X.shape[0]} samples, each with shape {X.shape[1:]}")

# === Step 2: Normalize Features ===
# Flatten for scaler: (num_samples * 10, 3)
X_reshaped = X.reshape(-1, X.shape[-1])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)

# Reshape back to (num_samples, 10, 3)
X = X_scaled.reshape(-1, 10, 3)

# === Step 3: Train/Validation Split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Step 4: Build CNN → LSTM → MLP Model ===
model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(10, 3)),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Output: 1 = profitable trade
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# === Step 5: Train the Model ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop]
)

# === Step 6: Save Model and Scaler ===
model.save("trade_classifier_model.h5")
joblib.dump(scaler, "kalman_scaler.pkl")

print("\n✅ Training complete. Model saved as 'trade_classifier_model.h5'")
print("✅ Scaler saved as 'kalman_scaler.pkl'")
