import os
import sys
import pandas as pd
import joblib
from dotenv import load_dotenv
import torch
import numpy as np

# Alpaca Trading modules
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Ensure project root is in sys.path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.trading.lstm_model import LSTMClassifier

class LiveExecutor:
    """
    LiveExecutor interfaces with Alpaca API for Paper Trading, generating signals
    for the current trading day using the saved LSTM model and executing market orders.
    """
    def __init__(self, features_path: str = "data/processed/features.parquet",
                 model_path: str = "models/lstm_model.pth",
                 scaler_path: str = "models/lstm_scaler.joblib"):
        """
        Initializes the executor, loads environment variables, configures the Alpaca client,
        loads the StandardScaler, and loads the trained LSTM model.

        Args:
            features_path (str): Path to processed features parquet file.
            model_path (str): Path to saved PyTorch model weights file.
            scaler_path (str): Path to saved StandardScaler joblib file.
        """
        # Resolve paths
        if not os.path.isabs(features_path):
            self.features_path = os.path.normpath(os.path.join(project_root, features_path))
        else:
            self.features_path = features_path

        if not os.path.isabs(model_path):
            self.model_path = os.path.normpath(os.path.join(project_root, model_path))
        else:
            self.model_path = model_path

        if not os.path.isabs(scaler_path):
            self.scaler_path = os.path.normpath(os.path.join(project_root, scaler_path))
        else:
            self.scaler_path = scaler_path

        # Features list expected by model
        self.features_list = [
            'log_return', 'sma_20', 'sma_50', 'sma_200', 'volatility_20',
            'rsi_14', 'macd', 'macd_signal', 'bollinger_pb'
        ]

        # Load environment variables
        env_path = os.path.join(project_root, ".env")
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
            print(f"[LiveExecutor] Loaded environment variables from {env_path}")
        else:
            load_dotenv()

        # Load scaler
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}. Please run lstm_model first.")
        self.scaler = joblib.load(self.scaler_path)
        print(f"[LiveExecutor] Loaded StandardScaler from {self.scaler_path}")

        # Load LSTM model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model weights not found at {self.model_path}. Please run lstm_model first.")
        
        # Instantiate network
        self.model = LSTMClassifier(
            input_size=len(self.features_list),
            hidden_size=64,
            num_layers=2
        )

        # Load weights using pickle/numpy inverse logic (immune to Windows DLL load issues)
        import pickle
        with open(self.model_path, 'rb') as f:
            numpy_weights = pickle.load(f)
        state_dict = {k: torch.tensor(v) for k, v in numpy_weights.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[LiveExecutor] Loaded and initialized LSTM model from {self.model_path}")

        # Initialize Alpaca Client (Paper Trading)
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        # Safely handle missing/placeholder credentials to prevent execution crash
        self.dry_run = False
        if not api_key or not secret_key or api_key == "tu_api_key_aqui" or secret_key == "tu_secret_key_aqui":
            print("[LiveExecutor] WARNING: Alpaca API credentials are not set or contain placeholder values.")
            print("[LiveExecutor] Running in DRY-RUN mode. No actual orders will be sent to Alpaca.")
            self.dry_run = True
            self.client = None
        else:
            try:
                self.client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
                print("[LiveExecutor] Connected successfully to Alpaca Paper Trading client.")
            except Exception as e:
                print(f"[LiveExecutor] Failed to initialize Alpaca TradingClient: {e}")
                print("[LiveExecutor] Falling back to DRY-RUN mode.")
                self.dry_run = True
                self.client = None

    def get_latest_signals(self) -> dict:
        """
        Loads the processed features, extracts the latest 10 rows (lookback) for each symbol,
        scales them, and uses the LSTM model to predict tomorrow's direction (1=UP, 0=DOWN).

        Returns:
            dict: Dictionary mapping symbol to predicted direction.
        """
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Processed features file not found at: {self.features_path}")
            
        df = pd.read_parquet(self.features_path)
        
        # Sort by timestamp
        df = df.sort_index(level='timestamp')
        
        # Universe list
        universe_symbols = ['SPY', 'QQQ', 'IWM', 'TLT']
        
        # Filter for universe symbols present in the data
        df_filtered = df[df.index.get_level_values('symbol').isin(universe_symbols)]
        
        signals = {}
        
        print("\n--- Generating LSTM Trading Signals ---")
        for symbol in universe_symbols:
            # Filter rows for this symbol
            if symbol not in df_filtered.index.get_level_values('symbol'):
                print(f"[LiveExecutor] Symbol {symbol} not found in features data. Skipping.")
                continue
                
            group = df_filtered.xs(symbol, level='symbol').sort_index()
            
            # Extract last 10 rows
            latest_rows = group.tail(10)
            
            if len(latest_rows) < 10:
                print(f"[LiveExecutor] WARNING: Symbol {symbol} has only {len(latest_rows)} rows. Exactly 10 rows required for LSTM. Skipping.")
                continue
                
            # Print check of latest row for user reference
            latest_row = latest_rows.iloc[-1]
            timestamp = latest_rows.index[-1]
            print(f"Symbol: {symbol:<5} | Latest Timestamp: {str(timestamp):<30} | Close: {latest_row['close']:<8.2f} | RSI: {latest_row['rsi_14']:<5.2f}")
            
            # Extract features matrix
            X_latest = latest_rows[self.features_list]
            
            # Scale features
            X_latest_scaled = self.scaler.transform(X_latest)
            
            # Convert to torch tensor with shape (1, 10, len(features_list))
            X_tensor = torch.tensor(X_latest_scaled, dtype=torch.float32).unsqueeze(0)
            
            # Run prediction
            with torch.no_grad():
                logit = self.model(X_tensor)
                prob = torch.sigmoid(logit).item()
                prediction = 1 if prob > 0.5 else 0
                
            signals[symbol] = prediction
            
        print("\n--- Generated Trading Signals ---")
        for symbol, signal in signals.items():
            direction = "UP (BUY)" if signal == 1 else "DOWN/FLAT (SELL/CLOSE)"
            print(f"Symbol: {symbol:<5} -> Signal: {signal} ({direction})")
            
        return signals

    def execute_trades(self, signals: dict) -> None:
        """
        Executes trading decisions based on predictions and current portfolio positions.

        Args:
            signals (dict): Dictionary mapping symbol to predicted signal.
        """
        # Fetch current positions
        position_symbols = []
        if not self.dry_run:
            try:
                positions = self.client.get_all_positions()
                position_symbols = [p.symbol for p in positions]
                print(f"\n[LiveExecutor] Active portfolio positions: {position_symbols}")
            except Exception as e:
                print(f"[LiveExecutor] Error fetching positions: {e}. Defaulting to empty positions.")
        else:
            # Simulating position layout in Dry Run (e.g. assume we already have QQQ)
            position_symbols = ['QQQ']
            print(f"\n[LiveExecutor] [DRY-RUN] Simulating active portfolio positions: {position_symbols}")
            
        print("\n--- Trade Execution Loop ---")
        for symbol, signal in signals.items():
            has_position = symbol in position_symbols
            
            if signal == 1:
                # UP signal: Buy if we don't have a position
                if not has_position:
                    if not self.dry_run:
                        print(f"[LiveExecutor] Buying 1 share of {symbol} (Market Order)...")
                        order_data = MarketOrderRequest(
                            symbol=symbol,
                            qty=1,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY
                        )
                        try:
                            order = self.client.submit_order(order_data=order_data)
                            print(f"[LiveExecutor] ORDER PLACED: Buy 1 {symbol} (ID: {order.id})")
                        except Exception as e:
                            print(f"[LiveExecutor] Failed to place order for {symbol}: {e}")
                    else:
                        print(f"[LiveExecutor] [DRY-RUN] Would submit BUY market order for 1 share of {symbol}")
                else:
                    print(f"[LiveExecutor] Signal UP for {symbol}, but position already exists. No action taken.")
                    
            elif signal == 0:
                # DOWN signal: Sell/Close if we have a position
                if has_position:
                    if not self.dry_run:
                        print(f"[LiveExecutor] Closing position in {symbol}...")
                        try:
                            self.client.close_position(symbol_or_asset_id=symbol)
                            print(f"[LiveExecutor] POSITION CLOSED: {symbol}")
                        except Exception as e:
                            print(f"[LiveExecutor] Failed to close position for {symbol}: {e}")
                    else:
                        print(f"[LiveExecutor] [DRY-RUN] Would close existing position in {symbol}")
                else:
                    print(f"[LiveExecutor] Signal DOWN/FLAT for {symbol}, and no position exists. No action taken.")

if __name__ == "__main__":
    executor = LiveExecutor()
    signals = executor.get_latest_signals()
    executor.execute_trades(signals)
