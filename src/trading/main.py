import os
import sys
import time
import schedule

# Ensure project root is in sys.path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.trading.data_ingestion import DataIngestor
from src.trading.feature_engineering import FeatureEngineer
from src.trading.execution import LiveExecutor

def run_daily_routine():
    """
    Executes the daily quant trading routine: Ingestion -> Feature Engineering -> Live Execution.
    """
    print("\n" + "="*80)
    print("                     STARTING QUANT TRADING DAILY CYCLE")
    print("="*80)
    
    try:
        # 1. Market Data Ingestion
        print("\n[Step 1/3] Ingesting latest market data from Alpaca...")
        ingestor = DataIngestor()
        # Note: fetch_market_data downloads and updates the raw parquet prices file
        ingestor.fetch_market_data()
        print("[Step 1/3] Market data ingestion completed successfully.")
        
    except Exception as e:
        print(f"[ERROR] Step 1/3 Ingestion failed: {e}")
        print("[WARNING] Proceeding with existing raw market data if available...")

    try:
        # 2. Feature Engineering
        print("\n[Step 2/3] Calculating technical indicators and features...")
        engineer = FeatureEngineer()
        # Note: run_pipeline updates processed/features.parquet
        engineer.run_pipeline()
        print("[Step 2/3] Feature engineering pipeline completed successfully.")
        
    except Exception as e:
        print(f"[ERROR] Step 2/3 Feature engineering failed: {e}")
        print("[WARNING] Proceeding with existing features if available...")

    try:
        # 3. Live Execution
        print("\n[Step 3/3] Generating live signals and executing trades...")
        executor = LiveExecutor()
        signals = executor.get_latest_signals()
        executor.execute_trades(signals)
        print("[Step 3/3] Signal generation and execution loop completed successfully.")
        
    except Exception as e:
        print(f"[ERROR] Step 3/3 Trade execution failed: {e}")

    print("\n" + "="*80)
    print("                     FINISHED QUANT TRADING DAILY CYCLE")
    print("="*80 + "\n")

if __name__ == "__main__":
    # Check if immediate execution was requested
    if len(sys.argv) > 1 and sys.argv[1] == "--now":
        print("[System] Running daily routine immediately (--now flag detected)...")
        run_daily_routine()
        sys.exit(0)
        
    # Otherwise, configure daily schedule (Monday through Friday at 15:45)
    execution_time = "15:45"
    print(f"[Scheduler] Configuring routine to run Monday-Friday at {execution_time} local time...")
    
    schedule.every().monday.at(execution_time).do(run_daily_routine)
    schedule.every().tuesday.at(execution_time).do(run_daily_routine)
    schedule.every().wednesday.at(execution_time).do(run_daily_routine)
    schedule.every().thursday.at(execution_time).do(run_daily_routine)
    schedule.every().friday.at(execution_time).do(run_daily_routine)
    
    print("[Scheduler] Bot initialized. Waiting for scheduled execution times...")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)
        except KeyboardInterrupt:
            print("\n[System] Scheduler stopped by user. Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"[System] Scheduler loop encountered an error: {e}")
            time.sleep(60)
