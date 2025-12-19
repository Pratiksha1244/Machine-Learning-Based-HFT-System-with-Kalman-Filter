import logging
from datetime import datetime

logging.basicConfig(
    filename="strategy.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def log(message):
    print(f"[{datetime.now()}] {message}")
    logging.info(message)
