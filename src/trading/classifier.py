import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

class TradingClassifier:
    """
    TradingClassifier handles reading processed market data, generating prediction targets,
    splitting data chronologically, training a RandomForest Classifier via a Pipeline,
    and evaluating the results.
    """
    def __init__(self, data_path: str = "data/processed/features.parquet"):
        """
        Initializes the TradingClassifier.

        Args:
            data_path (str): Path to the processed features parquet file.
        """
        # Convert path to absolute to avoid relative path issues when executing from different directories
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.normpath(os.path.join(current_dir, "../.."))

        if not os.path.isabs(data_path):
            self.data_path = os.path.normpath(os.path.join(project_root, data_path))
        else:
            self.data_path = data_path

        # Updated features list to include technical indicators for v2.0
        self.features_list = [
            'log_return', 'sma_20', 'sma_50', 'sma_200', 'volatility_20',
            'rsi_14', 'macd', 'macd_signal', 'bollinger_pb'
        ]
        self.model = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads the processed features from the Parquet file.

        Returns:
            pd.DataFrame: Loaded features DataFrame.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Processed features file not found at: {self.data_path}")
            
        print(f"[TradingClassifier] Loading data from {self.data_path}")
        return pd.read_parquet(self.data_path)

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates the target variable and prepares the final dataset.
        The target is defined as: 1 if next day's close price > today's close price, else 0.
        Rows containing NaNs (e.g. the last day of the dataset) are dropped.

        Args:
            df (pd.DataFrame): Dataframe containing the technical features and close price.

        Returns:
            pd.DataFrame: Dataframe with the target variable added and cleaned of NaNs.
        """
        print("[TradingClassifier] Preparing target variable (next-day close direction)...")
        
        # Ensure correct sorting by symbol and timestamp
        df = df.sort_index()

        # Shift next day's close price per symbol (look-ahead bias protection)
        next_day_close = df.groupby(level='symbol')['close'].shift(-1)
        
        # Define target: 1 if next day close is strictly greater than today's close, else 0
        df['target'] = (next_day_close > df['close']).astype(int)

        # Force target to be NaN where next_day_close is actually missing/NaN (e.g., the last row of each symbol)
        # to ensure safe removal and prevent any future data leakage or invalid labels.
        df.loc[next_day_close.isna(), 'target'] = None
        
        # Drop NaNs that occur because of the shift
        nans_before = df['target'].isna().sum()
        df = df.dropna(subset=['target'])
        
        # Convert target to integer type
        df['target'] = df['target'].astype(int)
        
        print(f"[TradingClassifier] Target variable created. Dropped {nans_before} rows with missing next-day close.")
        return df

    def split_data(self, df: pd.DataFrame, train_size: float = 0.8) -> tuple:
        """
        Splits the dataset chronologically into train and test sets.

        Args:
            df (pd.DataFrame): Cleaned dataset with target and features.
            train_size (float): Proportion of the dataset to include in the train split.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"[TradingClassifier] Splitting data chronologically (train={train_size:.0%}, test={1-train_size:.0%})...")
        
        # Sort by timestamp level of MultiIndex to ensure strict chronological splitting
        df = df.sort_index(level='timestamp')
        
        # Define features and target matrices
        X = df[self.features_list]
        y = df['target']

        split_idx = int(len(df) * train_size)
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Get split dates for logging
        train_start = X_train.index.get_level_values('timestamp').min()
        train_end = X_train.index.get_level_values('timestamp').max()
        test_start = X_test.index.get_level_values('timestamp').min()
        test_end = X_test.index.get_level_values('timestamp').max()
        
        print(f"[TradingClassifier] Train set: {len(X_train)} rows ({train_start} to {train_end})")
        print(f"[TradingClassifier] Test set:  {len(X_test)} rows ({test_start} to {test_end})")

        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self) -> None:
        """
        Runs the full model training and evaluation pipeline.
        """
        df = self.load_data()
        df = self.prepare_data(df)
        X_train, X_test, y_train, y_test = self.split_data(df)

        print("[TradingClassifier] Building pipeline with StandardScaler and RandomForestClassifier...")
        # Create pipeline with updated RandomForest hyperparameters for v2.0
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=7,
                min_samples_leaf=5,
                random_state=42
            ))
        ])

        print("[TradingClassifier] Training RandomForest model...")
        self.model.fit(X_train, y_train)
        
        print("[TradingClassifier] Generating predictions on the test set...")
        y_pred = self.model.predict(X_test)
        
        print("\n=== CLASSIFICATION REPORT ===")
        report = classification_report(y_test, y_pred, target_names=["DOWN/FLAT", "UP"])
        print(report)

        # Feature Importance Analysis
        print("=== FEATURE IMPORTANCE ===")
        importances = self.model.named_steps['rf'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.features_list,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Display as formatted text
        print(importance_df.to_string(index=False))

        # Save the trained model to disk
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.normpath(os.path.join(current_dir, "../.."))
        model_dir = os.path.join(project_root, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "rf_model_v2.joblib")
        
        joblib.dump(self.model, model_path)
        print(f"[TradingClassifier] Saved trained model to {model_path}")

if __name__ == "__main__":
    classifier = TradingClassifier()
    classifier.train_and_evaluate()
