import requests
import csv
from logger import log
from config import INSTRUMENT_KEY, TRADE_QTY, SANDBOX_ACCESS_TOKEN

def place_order(transaction_type: str):
    url = "https://api-sandbox.upstox.com/v3/order/place"
    payload = {
        "quantity": TRADE_QTY,
        "product": "D",
        "validity": "DAY",
        "price": 0,
        "tag": "auto_trade",
        "instrument_token": INSTRUMENT_KEY,
        "order_type": "MARKET",
        "transaction_type": transaction_type.upper(),
        "disclosed_quantity": 0,
        "trigger_price": 0,
        "is_amo": False,
        "slice": True
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {SANDBOX_ACCESS_TOKEN}'
    }
    res = requests.post(url, headers=headers, json=payload)
    log(f"{transaction_type.upper()} ORDER RESPONSE: {res.text}")
    return res.json()

def record_trade(action, price, est_price):
    with open('trades.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([action, price, est_price])
