import asyncio
import json
import ssl
import websockets
import requests
import numpy as np
from datetime import datetime
from google.protobuf.json_format import MessageToDict

import tensorflow as tf

import joblib
from collections import deque

import MarketDataFeedV3_pb2 as pb
from config import INSTRUMENT_KEY, LIVE_ACCESS_TOKEN
from kalman import KalmanFilter
from trader import place_order, record_trade
from logger import log

# === Kalman & Model Setup ===
kf = KalmanFilter()
position = 0
entry_price = None
epsilon = 1e-10
STOP_LOSS = 50.0

# === Load trained model and scaler ===
model = tf.keras.models.load_model("trade_classifier_model.h5")
scaler = joblib.load("kalman_scaler.pkl")
kalman_window = deque(maxlen=10)  # last 10 Kalman outputs

def get_market_data_feed_authorize_v3():
    url = 'https://api.upstox.com/v3/feed/market-data-feed/authorize'
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {LIVE_ACCESS_TOKEN}'
    }
    response = requests.get(url=url, headers=headers)
    return response.json()

def decode_protobuf(buffer):
    feed_response = pb.FeedResponse()
    feed_response.ParseFromString(buffer)
    return MessageToDict(feed_response)

def extract_ltp_from_dict(feed_dict):
    try:
        return float(feed_dict['feeds'][INSTRUMENT_KEY]['ff']['marketFF']['ltpc']['ltp'])
    except Exception as e:
        log(f"Failed to extract LTP: {e}")
        return None

async def fetch_market_data():
    global position, entry_price
    log("Starting Kalman strategy with model filter...")

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    auth = get_market_data_feed_authorize_v3()
    uri = auth["data"]["authorized_redirect_uri"]

    async with websockets.connect(uri, ssl=ssl_context) as ws:
        log("WebSocket connected...")

        await asyncio.sleep(1)

        sub_data = {
            "guid": "kalman-guid",
            "method": "sub",
            "data": {
                "mode": "full",
                "instrumentKeys": [INSTRUMENT_KEY]
            }
        }

        await ws.send(json.dumps(sub_data).encode())
        log(f"Subscription sent for: {INSTRUMENT_KEY}")

        while True:
            try:
                msg = await ws.recv()
                data_dict = decode_protobuf(msg)

                if "feeds" not in data_dict:
                    log(f"Non-feed message received: {data_dict}")
                    continue

                ltp = extract_ltp_from_dict(data_dict)
                if ltp is None:
                    continue

                est_price, velocity = kf.update(ltp)
                rel_diff = round((est_price - ltp) / ltp, 6)

                # Update Kalman signal window
                tick_features = [est_price, velocity, rel_diff]
                kalman_window.append(tick_features)

                log(f"Tick {kf.ticks} | Price ₹{ltp:.2f} | KF ₹{est_price:.6f} | Vel {velocity:.6f} | Δ {rel_diff:.6f}")

                if kf.ticks < 100 or len(kalman_window) < 10:
                    continue  # Wait until we have enough history

                # === MODEL FILTER ===
                input_window = np.array(kalman_window).reshape(1, 10, 3)
                scaled_window = scaler.transform(input_window.reshape(-1, 3)).reshape(1, 10, 3)
                prob = model.predict(scaled_window)[0][0]

                log(f"Model prediction: {prob:.4f}")

                # === Entry Condition ===
                if (velocity < 0 and (-0.001 + epsilon) < rel_diff < (0 - epsilon)) and position == 0:
                    if prob >= 0.63:
                        log(f"ENTRY | BUY SIGNAL | RelDiff: {rel_diff:.6f} | Velocity: {velocity:.6f} | Prob: {prob:.4f}")
                        place_order("BUY")
                        record_trade("BUY", ltp, est_price)
                        entry_price = ltp
                        position = 1
                    else:
                        log(f"REJECTED | Trade blocked by model | Prob: {prob:.4f}")

                # === Exit Condition ===
                elif (velocity > 0 and (0 + epsilon) < rel_diff < (0.001 - epsilon)) and position == 1:
                    log(f"EXIT | SELL SIGNAL | RelDiff: {rel_diff:.6f} | Velocity: {velocity:.6f}")
                    place_order("SELL")
                    record_trade("SELL", ltp, est_price)
                    position = 0
                    entry_price = None

                # === Hard Stop Loss ===
                if position == 1 and entry_price and (ltp <= entry_price - STOP_LOSS):
                    log(f"STOP LOSS HIT! SELLING @ ₹{ltp:.2f} (Entry: ₹{entry_price:.2f})")
                    place_order("SELL")
                    record_trade("STOPLOSS SELL", ltp, est_price)
                    position = 0
                    entry_price = None

                # === End-of-Day Flattening ===
                if datetime.now().strftime("%H:%M") >= "15:29":
                    if position == 1:
                        log("Market Close: Flattening position.")
                        place_order("SELL")
                        record_trade("EOD SELL", ltp, est_price)
                        position = 0
                        entry_price = None
                    kf.__init__()  # Reset Kalman filter for new session
                    kalman_window.clear()
                    log("Kalman filter reset for new session.")

            except Exception as e:
                log(f"Error in loop: {str(e)}")
                await asyncio.sleep(1)

if __name__ == "__main__":
    while True:
        try:
            asyncio.run(fetch_market_data())
        except Exception as e:
            log(f"WebSocket disconnected or crashed: {e}. Retrying in 5s...")
            asyncio.sleep(5)
