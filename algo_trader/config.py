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
from tickers import SP100_TICKERS  # noqa: E402

# ---------------------------------------------------------------------------
# Execution Parameters
# ---------------------------------------------------------------------------
CANDLE_TIMEFRAME: str = "1Min"   # Lumibot timestep identifier
LOOKBACK_BARS: int = 60          # Number of 1-min bars in each DeepScalper state tensor
SLEEP_TIME: str = "1M"           # Lumibot on_trading_iteration frequency

# ---------------------------------------------------------------------------
# Risk Management
# ---------------------------------------------------------------------------
STARTING_CAPITAL: float = 5_000.00    # Paper trading account size (USD)
KELLY_FRACTION: float = 0.5           # Fractional Kelly coefficient (half-Kelly for safety)
ATR_PERIOD: int = 14                  # Periods for ATR calculation
ATR_STOP_MULTIPLIER: float = 2.0      # Stop-loss = entry ± (ATR × multiplier)
ATR_TP_MULTIPLIER: float = 4.0        # Take-profit = entry ± (ATR × TP multiplier)
MAX_POSITION_PCT: float = 0.03        # Maximum 3 % of portfolio in any single stock

# ---------------------------------------------------------------------------
# Circuit Breakers
# ---------------------------------------------------------------------------
MAX_DAILY_LOSS_PCT: float = 0.03      # Halt all trading if down 3 % on the day
NO_TRADE_OPEN_BUFFER_MIN: int = 15    # No trades in first 15 min of session (9:30–9:45 ET)
NO_TRADE_CLOSE_BUFFER_MIN: int = 15   # No trades in last 15 min of session (3:45–4:00 ET)
CLOSE_ALL_EOD: bool = True            # Force-close all positions before session end
EOD_CLOSE_BUFFER_MIN: int = 5         # Minutes before close to trigger EOD flatten

# Market session constants (US Eastern Time)
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
LOB_DIM: int = 5       # Micro/intrabar features (LOB proxy — no real LOB available)
PRIV_DIM: int = 2      # Private state: (position_flag, unrealised_pnl_pct)

# Legacy alias kept for backward compatibility
INPUT_DIM: int = MACRO_DIM  # = 11

# --- Action space (Branching Dueling Q-Network) ---
N_DIR: int = 3         # Direction branch: 0=HOLD, 1=BUY, 2=SELL
N_SIZE: int = 4        # Size branch: 0=25%, 1=50%, 2=75%, 3=100% of max notional
ACTION_DIM: int = N_DIR  # Legacy alias (direction count)

# --- Encoder dimensions ---
GRU_HIDDEN: int = 128        # GRU hidden size per stream in MicroEncoder
MACRO_EMBED_DIM: int = 64    # MacroEncoder MLP output dim
FC_HIDDEN: int = 128         # FC hidden width in BDQ advantage/value heads

# Legacy aliases kept for backward compatibility
HIDDEN_SIZE: int = GRU_HIDDEN
FC_SIZE: int = FC_HIDDEN
DROPOUT_RATE: float = 0.0    # Paper does not specify dropout; set to 0

# --- Hindsight bonus (Section 4.2) ---
HINDSIGHT_HORIZON: int = 60   # h: look-ahead bars for hindsight bonus (paper searches [30,180])
HINDSIGHT_WEIGHT: float = 0.01  # w: bonus coefficient (paper searches [1e-3, 1e-1])

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
TRANSACTION_COST_LAMBDA: float = 0.0001  # 10 bps per side (Alpaca implied cost)

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
