import os
import numpy as np
import pandas as pd

class FeatureEngineer:
    """
    FeatureEngineer handles reading raw historical market data,
    calculating quantitative features using vectorized operations,
    cleaning null values, and saving the processed features.
    """
    def __init__(self, raw_data_path: str = "data/raw/market_data/raw_prices.parquet", 
                 processed_data_path: str = "data/processed/features.parquet"):
        """
        Initializes the FeatureEngineer.

        Args:
            raw_data_path (str): Path to the raw market data parquet file.
            processed_data_path (str): Path where the processed features parquet file will be saved.
        """
        # Convert path to absolute to avoid relative path issues when executing from different directories
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.normpath(os.path.join(current_dir, "../.."))

        if not os.path.isabs(raw_data_path):
            self.raw_data_path = os.path.normpath(os.path.join(project_root, raw_data_path))
        else:
            self.raw_data_path = raw_data_path

        if not os.path.isabs(processed_data_path):
            self.processed_data_path = os.path.normpath(os.path.join(project_root, processed_data_path))
        else:
            self.processed_data_path = processed_data_path

    def load_raw_data(self) -> pd.DataFrame:
        """
        Loads the raw market data from the Parquet file.

        Returns:
            pd.DataFrame: Loaded raw market data.
        """
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw market data file not found at: {self.raw_data_path}")
            
        print(f"[FeatureEngineer] Loading raw data from {self.raw_data_path}")
        df = pd.read_parquet(self.raw_data_path)
        return df

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates quantitative features using vectorized pandas operations per symbol.

        Features calculated:
            - log_return: Logarithmic daily return of close price.
            - sma_20, sma_50, sma_200: Simple Moving Averages of 20, 50, and 200 days.
            - volatility_20: 20-day rolling standard deviation of log returns, annualized.
            - rsi_14: Relative Strength Index of 14 days using Wilder's smoothing.
            - macd: Moving Average Convergence Divergence line (EMA_12 - EMA_26).
            - macd_signal: Signal line (9-day EMA of MACD).
            - bollinger_pb: Bollinger Bands %B indicator.

        Args:
            df (pd.DataFrame): Raw historical market data with MultiIndex (symbol, timestamp).

        Returns:
            pd.DataFrame: DataFrame with calculated features.
        """
        print("[FeatureEngineer] Calculating technical features...")
        
        # Work on a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Sort by index to ensure proper chronological order per asset
        df = df.sort_index()

        # 1. Log Returns
        # Group by level='symbol' and shift to get previous day's close
        df['log_return'] = np.log(df['close'] / df.groupby(level='symbol')['close'].shift(1))

        # 2. Simple Moving Averages (SMA)
        # Using .transform() to maintain index alignment and handle grouping correctly
        df['sma_20'] = df.groupby(level='symbol')['close'].transform(lambda x: x.rolling(window=20).mean())
        df['sma_50'] = df.groupby(level='symbol')['close'].transform(lambda x: x.rolling(window=50).mean())
        df['sma_200'] = df.groupby(level='symbol')['close'].transform(lambda x: x.rolling(window=200).mean())

        # 3. Historical Volatility (20-day rolling standard deviation of log returns, annualized)
        # We multiply by the square root of 252 trading days to annualize volatility
        df['volatility_20'] = df.groupby(level='symbol')['log_return'].transform(
            lambda x: x.rolling(window=20).std()
        ) * np.sqrt(252)

        # 4. RSI (14 days) using Wilder's smoothing equivalent (ewm with com=period-1)
        def _calc_rsi(s, period=14):
            delta = s.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
            avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
            
            # Prevent division by zero
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            # Fill NaNs where avg_loss was 0. If avg_gain is also 0, RSI is 50. Otherwise, RSI is 100.
            rsi = rsi.fillna(100.0)
            rsi.loc[(avg_gain == 0) & (avg_loss == 0)] = 50.0
            return rsi

        df['rsi_14'] = df.groupby(level='symbol')['close'].transform(lambda x: _calc_rsi(x, 14))

        # 5. MACD (EMA_12 - EMA_26) and Signal line (EMA_9 of MACD)
        ema_12 = df.groupby(level='symbol')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        ema_26 = df.groupby(level='symbol')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df.groupby(level='symbol')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())

        # 6. Bollinger Bands %B (bollinger_pb)
        # Using a 20-day window and 2 standard deviations
        sma_bb_20 = df.groupby(level='symbol')['close'].transform(lambda x: x.rolling(window=20).mean())
        std_bb_20 = df.groupby(level='symbol')['close'].transform(lambda x: x.rolling(window=20).std())
        upper_band = sma_bb_20 + 2 * std_bb_20
        lower_band = sma_bb_20 - 2 * std_bb_20
        
        # Prevent division by zero if upper_band equals lower_band (e.g. flat pricing)
        df['bollinger_pb'] = (df['close'] - lower_band) / (upper_band - lower_band).replace(0, np.nan)
        df['bollinger_pb'] = df['bollinger_pb'].fillna(0.5)

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the DataFrame by removing NaN values generated by rolling calculations.

        Args:
            df (pd.DataFrame): DataFrame with features containing NaNs.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        initial_len = len(df)
        # Drop rows with NaN values
        df_cleaned = df.dropna()
        final_len = len(df_cleaned)
        
        print(f"[FeatureEngineer] Data cleaning complete. Dropped {initial_len - final_len} rows with NaNs. Remaining: {final_len} rows.")
        return df_cleaned

    def save_processed_data(self, df: pd.DataFrame) -> None:
        """
        Saves the processed features DataFrame to a Parquet file.

        Args:
            df (pd.DataFrame): Cleaned features DataFrame.
        """
        processed_dir = os.path.dirname(self.processed_data_path)
        if processed_dir:
            os.makedirs(processed_dir, exist_ok=True)
            
        df.to_parquet(self.processed_data_path)
        print(f"[FeatureEngineer] Saved processed features to {self.processed_data_path}")

    def run_pipeline(self) -> pd.DataFrame:
        """
        Executes the entire feature engineering pipeline.

        Returns:
            pd.DataFrame: Processed and cleaned features DataFrame.
        """
        try:
            raw_df = self.load_raw_data()
            features_df = self.calculate_features(raw_df)
            cleaned_df = self.clean_data(features_df)
            self.save_processed_data(cleaned_df)
            return cleaned_df
        except Exception as e:
            print(f"[FeatureEngineer] Pipeline execution failed: {e}")
            raise

if __name__ == "__main__":
    # In execution mode, instanciate and run the pipeline
    engineer = FeatureEngineer()
    processed_df = engineer.run_pipeline()
    
    print("\n--- Processed Data Summary ---")
    print(f"Total rows: {len(processed_df)}")
    print("\nFirst 5 rows of calculated features:")
    feature_cols = ['close', 'log_return', 'sma_20', 'sma_50', 'sma_200', 'volatility_20', 'rsi_14', 'macd', 'macd_signal', 'bollinger_pb']
    print(processed_df[feature_cols].head())
