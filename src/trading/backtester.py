import os
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path to ensure correct imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.trading.classifier import TradingClassifier

class VectorizedBacktester:
    """
    VectorizedBacktester simulates historical trading performance of the model's predictions,
    calculates portfolio curves, and generates financial performance and risk metrics.
    """
    def __init__(self, data_path: str = "data/processed/features.parquet"):
        """
        Initializes the backtester by instantiating the classifier, loading, preparing,
        splitting, and training the model on the historical data.

        Args:
            data_path (str): Path to the processed features parquet file.
        """
        print("[VectorizedBacktester] Initializing and training model...")
        self.classifier = TradingClassifier(data_path=data_path)
        
        # Load and prepare data
        df = self.classifier.load_data()
        df = self.classifier.prepare_data(df)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = self.classifier.split_data(df)
        
        # Train model using classifier's routine
        self.classifier.train_and_evaluate()
        self.model = self.classifier.model
        print("[VectorizedBacktester] Model training complete and loaded.")

    def run_backtest(self) -> pd.DataFrame:
        """
        Runs the vectorized backtest on the test set. Calculates market and strategy returns,
        cumulative returns, and individual asset equity curves.

        Returns:
            pd.DataFrame: Backtest results per asset.
        """
        print("[VectorizedBacktester] Running backtest simulation...")
        
        # 1. Generate predictions on the test set
        y_pred = self.model.predict(self.X_test)
        
        # 2. Create results DataFrame matching X_test index (symbol, timestamp)
        results_df = pd.DataFrame(index=self.X_test.index)
        results_df['y_pred'] = y_pred
        results_df['log_return'] = self.X_test['log_return']
        
        # Ensure chronological sorting per symbol
        results_df = results_df.sort_index()
        
        # 3. Calculate market and strategy returns (Look-Ahead Bias Protection)
        # Shift log_return by -1 per symbol: the return earned tomorrow is associated with today's prediction
        results_df['market_return'] = results_df.groupby(level='symbol')['log_return'].shift(-1)
        results_df['strategy_return'] = results_df['y_pred'] * results_df['market_return']
        
        # Drop the last row of each symbol containing NaN due to the shift
        results_df = results_df.dropna(subset=['market_return'])
        
        # 4. Calculate Equity Curves (Initial portfolio = $10,000 USD)
        results_df['strategy_cum_return'] = results_df.groupby(level='symbol')['strategy_return'].cumsum()
        results_df['bh_cum_return'] = results_df.groupby(level='symbol')['market_return'].cumsum()
        
        results_df['strategy_equity'] = 10000.0 * np.exp(results_df['strategy_cum_return'])
        results_df['bh_equity'] = 10000.0 * np.exp(results_df['bh_cum_return'])
        
        return results_df

    def calculate_metrics(self, returns: pd.Series) -> dict:
        """
        Helper method to calculate quantitative performance and risk metrics.

        Args:
            returns (pd.Series): Daily log returns of the asset/portfolio.

        Returns:
            dict: Dictionary with Total Return, Volatility, Sharpe, and Max Drawdown.
        """
        if len(returns) == 0:
            return {
                "Total Return (%)": 0.0,
                "Volatility (%)": 0.0,
                "Sharpe Ratio": 0.0,
                "Max Drawdown (%)": 0.0
            }
        
        # Total return (Buy & Hold or Strategy)
        total_return = np.exp(returns.sum()) - 1.0
        
        # Annualized Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio (annualized, Rf=0)
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
        
        # Max Drawdown
        equity = np.exp(returns.cumsum())
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            "Total Return (%)": total_return * 100,
            "Volatility (%)": volatility * 100,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_dd * 100
        }

    def generate_report(self, results_df: pd.DataFrame) -> None:
        """
        Calculates and displays performance metrics grouped by asset and globally,
        and saves results to CSV.

        Args:
            results_df (pd.DataFrame): Backtest results.
        """
        # Save results to CSV
        output_path = os.path.normpath(os.path.join(project_root, "data/results/backtest_results.csv"))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path)
        print(f"[VectorizedBacktester] Backtest results saved to {output_path}")

        # Metrics list
        metrics_data = []

        # 1. Calculate metrics per symbol
        symbols = results_df.index.get_level_values('symbol').unique()
        for symbol in symbols:
            symbol_df = results_df.xs(symbol, level='symbol')
            
            bh_metrics = self.calculate_metrics(symbol_df['market_return'])
            strat_metrics = self.calculate_metrics(symbol_df['strategy_return'])
            
            metrics_data.append({
                "Symbol": symbol,
                "Type": "Strategy",
                "Total Return (%)": strat_metrics["Total Return (%)"],
                "Volatility (%)": strat_metrics["Volatility (%)"],
                "Sharpe Ratio": strat_metrics["Sharpe Ratio"],
                "Max Drawdown (%)": strat_metrics["Max Drawdown (%)"]
            })
            metrics_data.append({
                "Symbol": symbol,
                "Type": "Buy & Hold",
                "Total Return (%)": bh_metrics["Total Return (%)"],
                "Volatility (%)": bh_metrics["Volatility (%)"],
                "Sharpe Ratio": bh_metrics["Sharpe Ratio"],
                "Max Drawdown (%)": bh_metrics["Max Drawdown (%)"]
            })

        # 2. Calculate global portfolio metrics (Equally-weighted portfolio)
        # Average returns across all assets on each date
        portfolio_returns = results_df.groupby('timestamp')[['market_return', 'strategy_return']].mean()
        
        global_bh = self.calculate_metrics(portfolio_returns['market_return'])
        global_strat = self.calculate_metrics(portfolio_returns['strategy_return'])
        
        metrics_data.append({
            "Symbol": "GLOBAL",
            "Type": "Strategy",
            "Total Return (%)": global_strat["Total Return (%)"],
            "Volatility (%)": global_strat["Volatility (%)"],
            "Sharpe Ratio": global_strat["Sharpe Ratio"],
            "Max Drawdown (%)": global_strat["Max Drawdown (%)"]
        })
        metrics_data.append({
            "Symbol": "GLOBAL",
            "Type": "Buy & Hold",
            "Total Return (%)": global_bh["Total Return (%)"],
            "Volatility (%)": global_bh["Volatility (%)"],
            "Sharpe Ratio": global_bh["Sharpe Ratio"],
            "Max Drawdown (%)": global_bh["Max Drawdown (%)"]
        })

        # Display results in a clear DataFrame table
        metrics_df = pd.DataFrame(metrics_data)
        
        # Sort so we compare strategy next to buy & hold for each symbol
        metrics_df = metrics_df.sort_values(by=["Symbol", "Type"], ascending=[True, False])
        
        print("\n" + "="*70)
        print("                 BACKTEST PERFORMANCE COMPARISON")
        print("="*70)
        print(metrics_df.to_string(index=False, formatters={
            "Total Return (%)": "{:,.2f}%".format,
            "Volatility (%)": "{:,.2f}%".format,
            "Sharpe Ratio": "{:,.2f}".format,
            "Max Drawdown (%)": "{:,.2f}%".format
        }))
        print("="*70)

if __name__ == "__main__":
    backtester = VectorizedBacktester()
    results = backtester.run_backtest()
    backtester.generate_report(results)
