# Quantitative ML Trading Pipeline

Pipeline modular de trading cuantitativo y ejecución automatizada en *Paper Trading* sobre ETFs líquidos de EE. UU. (`SPY`, `QQQ`, `IWM`, `TLT`), integrando ingesta histórica vía **Alpaca Markets API**, ingeniería de características técnicas, modelos de clasificación (**Random Forest** y **LSTM en PyTorch**), motor de *backtesting* y orquestación diaria programada.

---

## Arquitectura del Proyecto

```text
quant-ml-trading-pipeline/
├── config/
│   └── trading_universe.yaml         # Definición del universo (SPY, QQQ, IWM, TLT) y ciclo de investigación
├── data/
│   ├── raw/market_data/              # Almacenamiento local de precios históricos en formato Parquet
│   ├── processed/                    # Features técnicos procesados
│   └── results/                      # Resultados de backtesting
├── models/                           # Pesos y escaladores entrenados (LSTM .pth, Random Forest .joblib)
├── outputs/manifests/                # Manifiestos y registros de ejecución
├── src/
│   ├── __init__.py
│   └── trading/
│       ├── __init__.py
│       ├── data_ingestion.py         # Ingesta y validación de barras históricas desde Alpaca API
│       ├── feature_engineering.py    # Cálculo de retornos logarítmicos, SMA, volatilidad, RSI, MACD y Bollinger
│       ├── classifier.py             # Entrenamiento y evaluación de modelo Random Forest
│       ├── lstm_model.py             # Arquitectura y entrenamiento de red neuronal LSTM en PyTorch
│       ├── backtester.py             # Simulación histórica de señales, métricas y exportación
│       ├── execution.py              # Generación de señales diarias y envío de órdenes (Paper Trading)
│       └── main.py                   # Orquestador diario (modo inmediato --now o programado L-V 15:45)
├── .env.example                      # Plantilla de credenciales de Alpaca API
├── .gitignore
└── requirements.txt
```

---

## Flujo del Pipeline

1. **Ingesta de Datos (`data_ingestion.py`):** Carga la configuración desde `config/trading_universe.yaml`, se autentica con Alpaca Markets mediante variables de entorno y actualiza los precios ajustados en `data/raw/market_data/`.
2. **Ingeniería de Características (`feature_engineering.py`):** Construye indicadores técnicos y estadísticos (`log_return`, `sma_20`, `sma_50`, `sma_200`, `volatility_20`, `rsi_14`, `macd`, `macd_signal`, `bollinger_pb`) en `data/processed/features.parquet`.
3. **Modelado Predictivo (`classifier.py`, `lstm_model.py`) y Backtesting (`backtester.py`):** Entrenamiento y validación fuera de muestra con modelos basados en árboles y redes neuronales recurrentes LSTM, con evaluación histórica en `data/results/`.
4. **Ejecución Automatizada (`execution.py` & `main.py`):** Carga el modelo y escalador entrenados para generar señales actuales y ejecutar órdenes de mercado en entorno *Paper Trading* de Alpaca.

---

## Universo y Ciclo de Investigación (`config/trading_universe.yaml`)

- **Activos:** `SPY`, `QQQ`, `IWM`, `TLT` (*Benchmark:* `SPY`)
- **Entrenamiento:** `2015-01-01` a `2023-12-31`
- **Validación:** `2024-01-01` a `2024-12-31`
- **Walk-Forward:** `2025-01-01` a `2026-06-30`
- **Paper Trading:** desde `2026-07-01`

---

## Instalación y Uso

1. **Clonar el repositorio y crear un entorno virtual:**
   ```bash
   python -m venv .venv
   # En Windows:
   .venv\Scripts\activate
   # En macOS / Linux:
   source .venv/bin/activate
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar credenciales:**
   Copia `.env.example` a `.env` e ingresa tus llaves de *Paper Trading* de Alpaca:
   ```bash
   # Windows PowerShell:
   Copy-Item .env.example .env
   # macOS / Linux:
   cp .env.example .env
   ```

4. **Ejecutar el pipeline:**
   - **Ejecución inmediata (1 ciclo completo):**
     ```bash
     python src/trading/main.py --now
     ```
   - **Modo programado (Lunes a Viernes a las 15:45 hora local):**
     ```bash
     python src/trading/main.py
     ```
