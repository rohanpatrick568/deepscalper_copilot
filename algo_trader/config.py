"""
config.py — Centralised configuration for the DeepScalper AlgoTrader system.

All parameters are defined here. No magic numbers should appear in other modules.
Environment variables are loaded from a .env file using python-dotenv.
Override any value by setting the corresponding environment variable before
launching the process, or by editing this file directly.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a .env file located at the project root.
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Alpaca API Credentials  (loaded from .env — never hard-code these values)
# ---------------------------------------------------------------------------
ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"   # Paper trading endpoint
ALPACA_DATA_URL: str = "https://data.alpaca.markets"        # Data API v2

# ---------------------------------------------------------------------------
# Trading Universe
# ---------------------------------------------------------------------------
# Imported lazily by other modules via tickers.py to avoid circular imports.
# Reproduced here as a reference; the canonical list lives in tickers.py.
from tickers import SP100_TICKERS  # noqa: E402  (kept for legacy equity strategy)

# V2 CHANGE: Crypto universe — BTC/USD only for v2 (expand after Sharpe > 0.5 proven)
CRYPTO_PAIRS: list = ['BTC/USD']           # Single pair for v2
TRADING_UNIVERSE: list = CRYPTO_PAIRS      # Replaces SP100_TICKERS in crypto mode

# ---------------------------------------------------------------------------
# Execution Parameters
# ---------------------------------------------------------------------------
CANDLE_TIMEFRAME: str = "1Min"   # Lumibot timestep identifier
LOOKBACK_BARS: int = 10          # V2 CHANGE: 60 → 10 (TradeMaster: backward_num_day=5)
SLEEP_TIME: str = "1M"           # Lumibot on_trading_iteration frequency

# ---------------------------------------------------------------------------
# Risk Management
# ---------------------------------------------------------------------------
STARTING_CAPITAL: float = 5_000.00    # Paper trading account size (USD)
KELLY_FRACTION: float = 0.5           # Fractional Kelly coefficient (half-Kelly for safety)
ATR_PERIOD: int = 14                  # Periods for ATR calculation
ATR_STOP_MULTIPLIER: float = 2.0      # Stop-loss = entry ± (ATR × multiplier)
ATR_TP_MULTIPLIER: float = 4.0        # Take-profit = entry ± (ATR × TP multiplier)
MAX_POSITION_PCT: float = 0.95        # V2 CHANGE: 0.03 → 0.95 (single pair; Kelly sizes within)

# ---------------------------------------------------------------------------
# Circuit Breakers
# ---------------------------------------------------------------------------
MAX_DAILY_LOSS_PCT: float = 0.05      # V2 CHANGE: 0.03 → 0.05 (crypto more volatile; 24hr rolling)
NO_TRADE_OPEN_BUFFER_MIN = None       # V2 CHANGE: Not applicable — crypto is 24/7
NO_TRADE_CLOSE_BUFFER_MIN = None      # V2 CHANGE: Not applicable — crypto is 24/7
CLOSE_ALL_EOD: bool = False           # V2 CHANGE: No end-of-day in crypto
EOD_CLOSE_BUFFER_MIN: int = 5         # Kept for legacy equity strategy compatibility

# V2 CHANGE: Crypto-specific circuit breakers
VOLATILITY_HALT_MULTIPLIER: float = 4.0  # Halt if 5-min ATR > 4× its 72-hr rolling avg (flash crash)
CONSECUTIVE_LOSS_HALT: int = 8           # Halt 30 min after 8 consecutive losing trades

# Market session constants (US Eastern Time — kept for legacy equity strategy)
MARKET_OPEN_HOUR: int = 9
MARKET_OPEN_MINUTE: int = 30
MARKET_CLOSE_HOUR: int = 16
MARKET_CLOSE_MINUTE: int = 0
MARKET_TIMEZONE: str = "US/Eastern"

# ---------------------------------------------------------------------------
# Model Architecture  (DeepScalper paper — CIKM '22, Sun et al.)
# ---------------------------------------------------------------------------
WEIGHTS_DIR: Path = Path("./weights/")   # Local directory containing .pth weight files

# --- Observation dimensions ---
MACRO_DIM: int = 11    # Macro features: z_open/high/low/close/adj + z_d_5..30 (Table 2)
LOB_DIM: int = 4       # V2 CHANGE: 5 → 4 (new dual-mode micro features: spread/imbalance/depth/mid_move)
PRIV_DIM: int = 2      # Private state: (position_flag, unrealised_pnl_pct)

# Legacy alias kept for backward compatibility
INPUT_DIM: int = MACRO_DIM  # = 11

# --- Action space (Branching Dueling Q-Network) ---
# V2 CHANGE: Binary LONG/FLAT only — Alpaca crypto has no short selling
N_DIR: int = 2         # V2 CHANGE: 3 → 2. Direction branch: 0=FLAT, 1=LONG
N_SIZE: int = 1        # V2 CHANGE: 4 → 1. Size determined externally by Kelly Criterion
ACTION_DIM: int = 2   # V2 CHANGE: Binary action — 0=FLAT, 1=LONG

# --- Encoder dimensions ---
GRU_HIDDEN: int = 128        # GRU hidden size per stream in MicroEncoder
MACRO_EMBED_DIM: int = 64    # MacroEncoder MLP output dim
FC_HIDDEN: int = 128         # FC hidden width in BDQ advantage/value heads

# Legacy aliases kept for backward compatibility
HIDDEN_SIZE: int = GRU_HIDDEN
FC_SIZE: int = FC_HIDDEN
DROPOUT_RATE: float = 0.0    # Paper does not specify dropout; set to 0

# --- Hindsight bonus (Section 4.2) ---
HINDSIGHT_HORIZON: int = 10    # V2 CHANGE: 60 → 10 (TradeMaster: forward_num_day=5 bars)
HINDSIGHT_WEIGHT: float = 0.2  # V2 CHANGE: 0.01 → 0.2 (TradeMaster: future_weights=0.2)

# --- Risk-aware auxiliary task (Section 4.4) ---
AUX_TASK_ETA: float = 1.0    # η: relative importance of volatility prediction loss

# --- Training hyperparameters (reference values — used in Colab notebooks) ---
LEARNING_RATE: float = 3e-4
BATCH_SIZE: int = 64
REPLAY_BUFFER_CAPACITY: int = 50_000
TARGET_UPDATE_FREQ: int = 100             # Soft-update target network every N steps
TAU: float = 0.01                         # Soft-update coefficient
EPSILON_START: float = 1.0
EPSILON_END: float = 0.01
EPSILON_DECAY_STEPS: int = 10_000
MIN_EPISODES: int = 200
EARLY_STOP_PATIENCE: int = 20             # Episodes without val Sharpe improvement
PER_ALPHA: float = 0.6                    # Prioritized replay priority exponent
PER_BETA_START: float = 0.4              # IS weight annealing start value
TRANSACTION_COST_LAMBDA: float = 0.0025  # V2 CHANGE: 0.0001 → 0.0025 (Alpaca crypto taker fee: 25 bps)

# V2 CHANGE: TradeMaster train/val/test split (time-ordered, no shuffling)
TRAIN_SPLIT: float = 0.70    # 70% training
VAL_SPLIT: float   = 0.10    # 10% validation (model selection)
TEST_SPLIT: float  = 0.20    # 20% test (final evaluation — never used for model selection)

# V2 CHANGE: LOB feature config
LOB_LEVELS: int = 3                        # Use top 3 bid/ask levels from Alpaca orderbook
USE_REAL_LOB_INFERENCE: bool = True        # Use real Alpaca orderbook during live inference
USE_PROXY_LOB_TRAINING: bool = True        # Use OHLCV-proxy features during training
                                           # (real historical LOB data not available for 6mo)

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
DASHBOARD_REFRESH_MS: int = 1_000        # PyQt QTimer interval (milliseconds)
MAX_TRADE_LOG_ENTRIES: int = 500         # Maximum trade log lines kept in memory (FIFO)
CONFIDENCE_THRESHOLD: float = 0.60      # Min confidence score to show in signal panel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR: Path = Path("./logs/")
LOG_LEVEL: str = "INFO"
LOG_MAX_BYTES: int = 10 * 1_024 * 1_024  # 10 MB per log file
LOG_BACKUP_COUNT: int = 5

# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------
ALPACA_MAX_REQUESTS_PER_MINUTE: int = 180  # Conservative guard (free tier cap: 200/min)
