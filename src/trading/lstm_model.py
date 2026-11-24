import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# Add project root to sys.path to ensure correct imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class LSTMClassifier(nn.Module):
    """
    LSTMClassifier defines a PyTorch recurrent neural network architecture
    for binary classification of stock direction.
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape x: [batch_size, seq_len, input_size]
        out, _ = self.lstm(x)
        # Decode the hidden state of the last time step
        # out[:, -1, :] shape: [batch_size, hidden_size]
        logits = self.fc(out[:, -1, :])
        return logits

class DeepLearningPipeline:
    """
    DeepLearningPipeline manages the training and evaluation of the LSTM model.
    It prepares sequential inputs grouped by symbol to avoid cross-asset leakage,
    splits the data chronologically, fits the neural network, and saves weights.
    """
    def __init__(self, data_path: str = "data/processed/features.parquet",
                 model_path: str = "models/lstm_model.pth",
                 lookback: int = 10,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 batch_size: int = 64,
                 epochs: int = 50,
                 lr: float = 0.001):
        
        # Resolve absolute paths
        if not os.path.isabs(data_path):
            self.data_path = os.path.normpath(os.path.join(project_root, data_path))
        else:
            self.data_path = data_path

        if not os.path.isabs(model_path):
            self.model_path = os.path.normpath(os.path.join(project_root, model_path))
        else:
            self.model_path = model_path

        self.scaler_path = os.path.join(os.path.dirname(self.model_path), "lstm_scaler.joblib")

        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        self.features_list = [
            'log_return', 'sma_20', 'sma_50', 'sma_200', 'volatility_20',
            'rsi_14', 'macd', 'macd_signal', 'bollinger_pb'
        ]
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[DeepLearningPipeline] Using device: {self.device}")

    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Loads the features parquet file, calculates the look-ahead free target,
        and cleans NaNs.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Processed features not found at: {self.data_path}")
            
        print(f"[DeepLearningPipeline] Loading features from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        # Sort index chronologically per symbol
        df = df.sort_index()

        # Shift next day's close price per symbol (look-ahead bias protection)
        next_day_close = df.groupby(level='symbol')['close'].shift(-1)
        
        # Target: 1 if next day's close > today's close, else 0
        df['target'] = (next_day_close > df['close']).astype(int)

        # Force target to be NaN where next_day_close is missing
        df.loc[next_day_close.isna(), 'target'] = None
        
        # Drop rows with NaN targets or NaN features
        df = df.dropna(subset=['target'] + self.features_list)
        df['target'] = df['target'].astype(int)
        
        return df

    def split_and_scale(self, df: pd.DataFrame) -> tuple:
        """
        Splits data chronologically (80% train, 20% test) and scales features.
        """
        print("[DeepLearningPipeline] Splitting data chronologically (80% train, 20% test)...")
        df = df.sort_index(level='timestamp')
        
        split_idx = int(len(df) * 0.8)
        split_date = df.index.get_level_values('timestamp')[split_idx]
        
        train_df = df[df.index.get_level_values('timestamp') <= split_date].copy()
        test_df = df[df.index.get_level_values('timestamp') > split_date].copy()

        print(f"[DeepLearningPipeline] Train rows: {len(train_df)} | Test rows: {len(test_df)}")

        # Fit scaler on train set and transform both sets
        scaler = StandardScaler()
        train_df[self.features_list] = scaler.fit_transform(train_df[self.features_list])
        test_df[self.features_list] = scaler.transform(test_df[self.features_list])

        # Save the fitted scaler to models/lstm_scaler.joblib
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        joblib.dump(scaler, self.scaler_path)
        print(f"[DeepLearningPipeline] Saved StandardScaler to {self.scaler_path}")

        return train_df, test_df

    def generate_sequences(self, df_set: pd.DataFrame) -> tuple:
        """
        Generates temporal sequences of lookback length grouped by symbol.
        """
        X_seqs = []
        y_seqs = []

        for symbol, group in df_set.groupby(level='symbol'):
            group = group.sort_index(level='timestamp')
            X_vals = group[self.features_list].values
            y_vals = group['target'].values
            
            if len(group) < self.lookback:
                continue
                
            for i in range(self.lookback, len(group)):
                # Window of past lookback days [i-lookback to i-1]
                X_seqs.append(X_vals[i - self.lookback : i])
                # Target of day i-1 (which represents close[i] > close[i-1])
                y_seqs.append(y_vals[i - 1])

        return np.array(X_seqs), np.array(y_seqs)

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> nn.Module:
        """
        Trains the LSTM model.
        """
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_size = len(self.features_list)
        model = LSTMClassifier(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        print(f"[DeepLearningPipeline] Training LSTM model for {self.epochs} epochs...")
        model.train()

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= len(loader.dataset)
            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch:02d}/{self.epochs:02d} | Train Loss: {epoch_loss:.6f}")

        self.model = model
        return model

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Evaluates the trained LSTM model on the test set.
        """
        print("[DeepLearningPipeline] Evaluating model on test set...")
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

        print("\n=== LSTM CLASSIFICATION REPORT ===")
        print(classification_report(y_test, preds, target_names=["DOWN/FLAT", "UP"]))

    def save_weights(self) -> None:
        """
        Saves the trained model state dict to disk using pickle to avoid torch.save import bugs.
        """
        import pickle
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        # Convert state_dict Tensors to numpy arrays for bulletproof serialization
        numpy_weights = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
        with open(self.model_path, 'wb') as f:
            pickle.dump(numpy_weights, f)
        print(f"[DeepLearningPipeline] Saved LSTM model weights to {self.model_path}")


    def run(self) -> None:
        """
        Orchestrates the entire deep learning model pipeline.
        """
        df = self.load_and_prepare_data()
        train_df, test_df = self.split_and_scale(df)
        
        X_train, y_train = self.generate_sequences(train_df)
        X_test, y_test = self.generate_sequences(test_df)
        
        print(f"[DeepLearningPipeline] Training sequences shape: {X_train.shape}")
        print(f"[DeepLearningPipeline] Test sequences shape:     {X_test.shape}")

        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)
        self.save_weights()

if __name__ == "__main__":
    pipeline = DeepLearningPipeline()
    pipeline.run()
