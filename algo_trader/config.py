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
from tickers import SP100_TICKERS

# Load environment variables from a .env file located at the project root.
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Alpaca API Credentials  (loaded from .env — never hard-code these values)
# ---------------------------------------------------------------------------
ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"   # Paper trading endpoint
ALPACA_DATA_URL: str = "https://data.alpaca.markets"        # Data API v2

# Lumibot reads generic broker credential names during import, so mirror the
# Alpaca values into those keys for compatibility.
os.environ["API_KEY"] = ALPACA_API_KEY
os.environ["API_SECRET"] = ALPACA_SECRET_KEY
os.environ["ALPACA_API_KEY"] = ALPACA_API_KEY
os.environ["ALPACA_API_SECRET"] = ALPACA_SECRET_KEY
os.environ["MARKET"] = "NYSE"

# ---------------------------------------------------------------------------
# Trading Universe
# ---------------------------------------------------------------------------
# Imported lazily by other modules via tickers.py to avoid circular imports.
# Reproduced here as a reference; the canonical list lives in tickers.py.

# Equity rollout universe (pilot subset for safer staged deployment)
TRADING_UNIVERSE: list = SP100_TICKERS[:10]
CRYPTO_PAIRS: list = []  # Backward-compat alias (deprecated)

# ---------------------------------------------------------------------------
# Execution Parameters
# ---------------------------------------------------------------------------
CANDLE_TIMEFRAME: str = "1Min"   # Lumibot timestep identifier
LOOKBACK_BARS: int = 60          # Paper-aligned observation lookback
SLEEP_TIME: str = "1M"           # Lumibot on_trading_iteration frequency

# ---------------------------------------------------------------------------
# Risk Management
# ---------------------------------------------------------------------------
STARTING_CAPITAL: float = 1_000.00    # Paper trading account size (USD)
KELLY_FRACTION: float = 0.5           # Fractional Kelly coefficient (half-Kelly for safety)
ATR_PERIOD: int = 14                  # Periods for ATR calculation
ATR_STOP_MULTIPLIER: float = 2.0      # Stop-loss = entry ± (ATR × multiplier)
ATR_TP_MULTIPLIER: float = 4.0        # Take-profit = entry ± (ATR × TP multiplier)
MAX_POSITION_PCT: float = 0.03        # Paper-aligned max position cap

# Trade frequency and exit control (live execution)
MIN_HOLD_BARS: int = 3                # Minimum bars to hold after entry before model-driven exit
ENTRY_COOLDOWN_BARS: int = 2          # Bars to wait after exit before allowing a new entry

# Trailing/volatility exits
USE_TRAILING_STOP: bool = True
TRAILING_ATR_MULTIPLIER: float = 1.5  # Trail distance = max(ATR*x, floor% of price)
TRAILING_STOP_FLOOR_PCT: float = 0.003  # 30 bps minimum trailing distance

# Volatility-scaled sizing
USE_VOLATILITY_SIZING: bool = True
TARGET_ENTRY_RISK_PCT: float = 0.005  # Target stop distance as fraction of price (50 bps)
MIN_POSITION_SCALE: float = 0.30
MAX_POSITION_SCALE: float = 1.00

# ---------------------------------------------------------------------------
# Circuit Breakers
# ---------------------------------------------------------------------------
MAX_DAILY_LOSS_PCT: float = 0.03      # Daily loss halt threshold
NO_TRADE_OPEN_BUFFER_MIN: int = 15    # No entries in first 15 min after open
NO_TRADE_CLOSE_BUFFER_MIN: int = 15   # No new entries in last 15 min before close
CLOSE_ALL_EOD: bool = True            # Force-flat by market close
EOD_CLOSE_BUFFER_MIN: int = 5         # Close all positions 5 min before close

# Legacy crypto-only breakers retained for compatibility but not used in equities mode
VOLATILITY_HALT_MULTIPLIER: float = 4.0
CONSECUTIVE_LOSS_HALT: int = 8

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
LOB_DIM: int = 5       # Paper-aligned micro feature schema width
PRIV_DIM: int = 2      # Private state: (position_flag, unrealised_pnl_pct)

# Legacy alias kept for backward compatibility
INPUT_DIM: int = MACRO_DIM  # = 11

# --- Action space (Branching Dueling Q-Network) ---
# Paper-aligned direction/size branches
N_DIR: int = 3         # Direction branch: 0=SHORT, 1=FLAT, 2=LONG
N_SIZE: int = 4        # Size branch cardinality
ACTION_DIM: int = 3    # Discrete action semantics: SHORT/FLAT/LONG

# --- Encoder dimensions ---
GRU_HIDDEN: int = 128        # GRU hidden size per stream in MicroEncoder
MACRO_EMBED_DIM: int = 64    # MacroEncoder MLP output dim
FC_HIDDEN: int = 128         # FC hidden width in BDQ advantage/value heads

# Legacy aliases kept for backward compatibility
HIDDEN_SIZE: int = GRU_HIDDEN
FC_SIZE: int = FC_HIDDEN
DROPOUT_RATE: float = 0.0    # Paper does not specify dropout; set to 0

# --- Hindsight bonus (Section 4.2) ---
HINDSIGHT_HORIZON: int = 60    # Paper-aligned forward horizon
HINDSIGHT_WEIGHT: float = 0.2  # TradeMaster-aligned coaching weight

# --- Risk-aware auxiliary task (Section 4.4) ---
AUX_TASK_ETA: float = 1.0    # η: relative importance of volatility prediction loss

# --- TradeMaster DeepScalper canonical hyperparameters ---
# Source: configs/algorithmic_trading/algorithmic_trading_BTC_deepscalper_deepscalper_adam_mse.py
EPOCHS: int = 20
BATCH_SIZE: int = 64
HORIZON_LEN: int = 128
BUFFER_SIZE: int = 1_000_000

LEARNING_RATE: float = 1e-3
GAMMA: float = 0.9
REPEAT_TIMES: float = 1.0
CLIP_GRAD_NORM: float = 3.0
SOFT_UPDATE_TAU: float = 0.0
STATE_VALUE_TAU: float = 0.005
EXPLORE_RATE: float = 0.25

# TradeMaster-equivalent environment defaults used by notebooks/backtests.
TRANSACTION_COST_LAMBDA: float = 0.001

# Compatibility aliases for legacy local code paths (deprecated).
REPLAY_BUFFER_CAPACITY: int = BUFFER_SIZE
TARGET_UPDATE_FREQ: int = 1
TAU: float = SOFT_UPDATE_TAU
EPSILON_START: float = EXPLORE_RATE
EPSILON_END: float = EXPLORE_RATE
EPSILON_DECAY_STEPS: int = 1
MIN_EPISODES: int = EPOCHS
EARLY_STOP_PATIENCE: int = EPOCHS
PER_ALPHA: float = 0.6
PER_BETA_START: float = 0.4

# V2 CHANGE: TradeMaster train/val/test split (time-ordered, no shuffling)
TRAIN_SPLIT: float = 0.70    # 70% training
VAL_SPLIT: float   = 0.10    # 10% validation (model selection)
TEST_SPLIT: float  = 0.20    # 20% test (final evaluation — never used for model selection)

# LOB feature config
LOB_LEVELS: int = 3
USE_REAL_LOB_INFERENCE: bool = False       # Equities path defaults to proxy-micro features
USE_PROXY_LOB_TRAINING: bool = True

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
