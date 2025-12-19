# Machine-Learning-Based-HFT-System-with-Kalman-Filter

A high-frequency trading (HFT) system that uses a 2-state Kalman filter for price estimation and a machine learning model for trade signal filtering. The system connects to Upstox market data feed via WebSocket and executes automated trades based on Kalman filter predictions.
 
## 🎯 Features
 
- **2-State Kalman Filter**: Estimates price and velocity from noisy market data
- **ML-Based Signal Filtering**: Uses a trained TensorFlow/Keras model to filter trading signals
- **Real-Time Market Data**: Connects to Upstox WebSocket feed for live market data
- **Automated Trading**: Executes buy/sell orders based on Kalman filter signals
- **Risk Management**: Implements stop-loss and end-of-day position flattening
- **Trade Logging**: Records all trades to CSV for analysis
- **Comprehensive Logging**: Detailed logs for debugging and monitoring
 
## 🏗️ Architecture
 
The system consists of several key components:
 
1. **Kalman Filter** (`kalman.py`): Implements a 2-state Kalman filter that tracks price and velocity
2. **Market Data Feed** (`main.py`): Connects to Upstox WebSocket and processes real-time market data
3. **Trading Logic** (`trader.py`): Handles order placement and trade recording
4. **ML Model**: Pre-trained model (`trade_classifier_model.h5`) filters trading signals
5. **Configuration** (`config.py`): Centralized configuration for API tokens and trading parameters
 
## 📋 Prerequisites
 
- Python 3.7+
- Upstox API account with valid access tokens
- TensorFlow 2.x
- Required Python packages (see Installation)
 
## 🚀 Installation
 
1. Clone the repository:
```bash
git clone <repository-url>
cd 2_state_kalman_train
```
 
2. Install required dependencies:
```bash
pip install numpy tensorflow websockets requests protobuf joblib
```
 
3. Generate protobuf files (if needed):
```bash
protoc --python_out=. MarketDataFeedV3.proto
```
 
4. Configure your API tokens in `config.py`:
   - Set `LIVE_ACCESS_TOKEN` for live trading
   - Set `SANDBOX_ACCESS_TOKEN` for sandbox testing
   - Configure `INSTRUMENT_KEY` for your trading instrument
   - Set `TRADE_QTY` for position size
 
5. Ensure you have the trained model files:
   - `trade_classifier_model.h5` - Trained TensorFlow model
   - `kalman_scaler.pkl` - Scaler for feature normalization
 
## ⚙️ Configuration
 
Edit `config.py` to set your trading parameters:
 
```python
LIVE_ACCESS_TOKEN = "your_live_token"
SANDBOX_ACCESS_TOKEN = "your_sandbox_token"
INSTRUMENT_KEY = "BSE_FO|843110"  # Your instrument key
INSTRUMENT_TOKEN = 55785  # Your instrument token
TRADE_QTY = 600  # Position size
```
 
## 📖 Usage
 
Run the main script to start the trading strategy:
 
```bash
python main.py
```
 
The system will:
1. Connect to Upstox WebSocket feed
2. Subscribe to market data for the configured instrument
3. Process incoming ticks through the Kalman filter
4. Generate trading signals based on:
   - Kalman filter price estimation and velocity
   - Relative difference between estimated and actual price
   - ML model probability score (threshold: 0.63)
5. Execute trades when conditions are met
6. Apply stop-loss and end-of-day flattening
 
### Trading Logic
 
**Entry Condition:**
- Velocity < 0 (price decreasing)
- Relative difference between estimated and actual price is negative (price below estimate)
- ML model probability ≥ 0.63
- No existing position
 
**Exit Condition:**
- Velocity > 0 (price increasing)
- Relative difference is positive (price above estimate)
- Existing position
 
**Stop Loss:**
- Hard stop-loss at ₹50 below entry price
 
**End-of-Day:**
- Automatically flattens position at 15:29
 
## 📁 Project Structure
 
```
2_state_kalman_train/
├── main.py                      # Main entry point and market data feed handler
├── kalman.py                    # 2-state Kalman filter implementation
├── trader.py                    # Order placement and trade recording
├── config.py                    # Configuration (API tokens, instrument keys)
├── logger.py                    # Logging utility
├── MarketDataFeedV3.proto       # Protobuf schema for market data
├── MarketDataFeedV3_pb2.py      # Generated protobuf Python bindings
├── trade_classifier_model.h5    # Trained ML model
├── kalman_scaler.pkl            # Feature scaler for ML model
├── trades.csv                   # Trade history log
└── strategy.log                 # Application logs
```
 
## 🔧 Key Components
 
### Kalman Filter
The 2-state Kalman filter tracks:
- **State 1**: Estimated price
- **State 2**: Price velocity (rate of change)
 
It uses process noise (Q) and measurement noise (R) matrices to balance between prediction and observation.
 
### ML Model Filter
A TensorFlow/Keras model that takes a sliding window of 10 Kalman filter outputs (price, velocity, relative difference) and outputs a probability score. Only trades with probability ≥ 0.63 are executed.
 
### Market Data Processing
- Receives real-time market data via WebSocket
- Decodes protobuf messages
- Extracts Last Traded Price (LTP)
- Updates Kalman filter with each tick
 
## 📊 Monitoring
 
- **Logs**: Check `strategy.log` for detailed execution logs
- **Trades**: Review `trades.csv` for trade history
- **Console**: Real-time output shows tick-by-tick processing
 
## ⚠️ Risk Disclaimer
 
**This software is for educational and research purposes only. Trading in financial markets involves substantial risk of loss. The authors and contributors are not responsible for any financial losses incurred from using this software.**
 
- Always test in sandbox mode before live trading
- Use appropriate position sizing
- Monitor the system actively
- Understand the risks involved in algorithmic trading
- Ensure compliance with your broker's terms of service
 
## 🔒 Security Notes
 
- **Never commit API tokens to version control**
- Use environment variables or secure configuration management
- Rotate access tokens regularly
- Review `config.py` and ensure sensitive data is not exposed
 
## 📝 License
 
[Specify your license here]
 
## 🤝 Contributing
 
[Add contribution guidelines if applicable]
 
## 📧 Contact
 
[Add contact information if desired]
