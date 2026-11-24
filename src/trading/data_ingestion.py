import os
from datetime import datetime
import yaml
import pandas as pd
from dotenv import load_dotenv

# Alpaca API modules
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

class DataIngestor:
    """
    DataIngestor is responsible for loading the trading universe configuration,
    connecting to market data providers (Alpaca), and downloading,
    validating, and storing historical market data.
    """
    def __init__(self, config_path: str = "../../config/trading_universe.yaml"):
        """
        Initializes the DataIngestor and loads the trading universe configuration.

        Args:
            config_path (str): Path to the trading_universe.yaml configuration file.
                               Defaults to relative path from src/trading/.
        """
        # Load environment variables from .env
        # We start by finding the root of the project where .env is stored
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.normpath(os.path.join(current_dir, "../.."))
        env_path = os.path.join(root_dir, ".env")
        
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
            print(f"[DataIngestor] Loaded environment variables from {env_path}")
        else:
            load_dotenv() # Fallback to standard search
            print("[DataIngestor] Warning: .env file not found at project root. Using system env.")

        # Resolve relative path for YAML config
        if not os.path.isabs(config_path):
            config_path = os.path.normpath(os.path.join(current_dir, config_path))

        self.config_path = config_path
        self.config = self._load_config()

        # Parse key config elements
        self.research_cycle = self.config.get("current_research_cycle", {})
        self.universe = self.config.get("universe", {})
        self.data_quality = self.config.get("data_quality", {})

        # Handy shortcuts
        self.symbols = self.universe.get("symbols", [])
        self.benchmark = self.universe.get("benchmark", "SPY")

        print(f"[DataIngestor] Loaded configuration from {self.config_path}")
        print(f"[DataIngestor] Active Universe: {self.universe.get('universe_id')} with symbols {self.symbols}")

    def _load_config(self) -> dict:
        """
        Loads the YAML configuration file.

        Returns:
            dict: Parsed YAML content.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError as exc:
                raise ValueError(f"Failed to parse YAML configuration: {exc}")

    def connect_api(self) -> StockHistoricalDataClient:
        """
        Establishes and returns a StockHistoricalDataClient connection to Alpaca.

        Returns:
            StockHistoricalDataClient: The initialized Alpaca historical data client.
        """
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key or api_key == "tu_api_key_aqui" or secret_key == "tu_secret_key_aqui":
            raise ValueError(
                "Alpaca API credentials are not set or are using placeholder values in .env. "
                "Please configure ALPACA_API_KEY and ALPACA_SECRET_KEY correctly."
            )

        print("[DataIngestor] Successfully connected to Alpaca Historical Data API.")
        return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    def fetch_market_data(self) -> pd.DataFrame:
        """
        Downloads historical daily market data for the active universe symbols
        from train_start to walk_forward_end, converts it to a DataFrame, and saves
        it to the raw market data folder.

        Returns:
            pd.DataFrame: The raw historical bar data.
        """
        start_date_str = self.research_cycle.get("train_start")
        end_date_str = self.research_cycle.get("walk_forward_end")

        if not start_date_str or not end_date_str:
            raise ValueError("Research cycle dates 'train_start' and 'walk_forward_end' must be specified in the configuration.")

        # Convert to datetime objects for Alpaca request
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")

        print(f"[DataIngestor] Fetching daily bars from {start_date_str} to {end_date_str} for symbols: {self.symbols}")
        
        client = self.connect_api()
        
        # Prepare the request parameters
        request_params = StockBarsRequest(
            symbol_or_symbols=self.symbols,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt
        )
        
        # Download data
        try:
            bars_response = client.get_stock_bars(request_params)
        except Exception as e:
            raise RuntimeError(f"Error requesting data from Alpaca API: {e}")
            
        df = bars_response.df
        
        if df.empty:
            print("[DataIngestor] Warning: Fetched DataFrame is empty. No data retrieved.")
            return df
            
        # Define output directory and file path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.normpath(os.path.join(current_dir, "../../data/raw/market_data"))
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "raw_prices.parquet")
        
        # Save raw data without modifications
        df.to_parquet(output_path)
        print(f"[DataIngestor] Saved raw market data ({len(df)} rows) to {output_path}")
        
        return df
