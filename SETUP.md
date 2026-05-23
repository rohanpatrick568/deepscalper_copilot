# AlgoTrader — DeepScalper V2 × Alpaca Crypto Paper Trading System

End-to-end algorithmic intraday trading system for **BTC/USD** using:
- **DeepScalper V2** — Branching DQN (BDQ) + PER + Hindsight Bonus + Volatility Auxiliary Task (CIKM '22, Sun et al.) — adapted for 24/7 crypto, binary FLAT/LONG actions, Kelly position sizing
- **Lumibot** — Python broker execution framework
- **Alpaca** — Paper trading API (no real money at risk); crypto endpoint, no short selling
- **Gymnasium** — RL environment interface (replaces legacy `gym`)
- **PyQt5** — Live dashboard with equity bar, positions table, confidence panel, trade log

---

## Architecture Overview

```
algo_trader/
├── main.py                          # Entry point — starts everything
├── config.py                        # All constants and parameters
├── tickers.py                       # S&P 100 ticker list (legacy equity; unused in V2)
├── requirements.txt                 # Pinned Python dependencies
├── weights/                         # Trained .pth files (one per trading pair)
│   └── README.md
├── execution/
│   ├── broker.py                    # Alpaca paper broker config (crypto endpoint)
│   ├── risk.py                      # Kelly sizing + ATR stops
│   ├── circuit_breakers.py          # Session time guards + daily loss halt (legacy equity)
│   ├── state_builder.py             # Live bars → {lob, priv, macro} observation dict for DeepScalperNet
│   └── strategy.py                  # Lumibot Strategy class (CryptoCitruitBreaker — V2 core loop)
├── dashboard/
│   ├── data_bridge.py               # Thread-safe state shared between threads
│   ├── equity_bar.py                # Top metrics bar widget
│   ├── positions_table.py           # Open positions table
│   ├── confidence_panel.py          # Per-stock signal cards
│   ├── trade_log.py                 # Trade event feed
│   └── main_window.py               # Root QMainWindow
├── colab/
│   ├── deepscalper/
│   │   ├── architecture.py          # DeepScalperNet V2 — BDQ: MacroEncoder + dual-stream GRU MicroEncoder + VolatilityHead
│   │   ├── agent.py                 # DeepScalperAgent — BDQ Double-DQN + PER + hindsight bonus + vol auxiliary task
│   │   ├── environment.py           # ScalperEnv V2 — Dict obs {lob,priv,macro}, Discrete(2) action space (FLAT/LONG)
│   │   └── utils.py                 # compute_macro_features (11) + compute_micro_features (4, dual-mode) + metrics helpers
│   ├── 01_fetch_training_data.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_train_deepscalper.ipynb
│   ├── 04_export_and_push_weights.ipynb
│   ├── 05_backtest_validation.ipynb
│   ├── 06_sharpe_diagnosis.ipynb    # V2 NEW: post-training Sharpe decomposition
│   └── 07_lob_recorder.ipynb       # V2 NEW: record real Alpaca LOB snapshots for live inference
└── tests/
    ├── conftest.py                  # Shared fixtures (make_bars, make_lob_snap, scalper_env, net_v2)
    ├── test_config.py               # Config constants and V2 parameter validation
    ├── test_utils.py                # Feature engineering (macro/micro/metrics)
    ├── test_environment.py          # ScalperEnv V2 spaces, reset, step, reward, edge cases
    ├── test_architecture.py         # Network shapes, forward pass, gradients
    ├── test_agent.py                # SumTree, PER, BDQ agent, epsilon decay, save/load
    ├── test_data_bridge.py          # Thread-safe data bridge properties and signals
    ├── test_crypto_circuit_breaker.py  # CryptoCitruitBreaker halt logic (24/7 conditions)
    ├── test_risk.py                 # Kelly sizing, ATR stop/TP calculation
    ├── test_state_builder.py        # Observation dict format, shapes, dtype, NaN safety
    └── test_circuit_breakers.py     # Legacy equity circuit breakers
```

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.13+ | Tested on 3.13.x; PyTorch 2.6+ required for 3.13 wheels |
| pip | 23+ | `pip install --upgrade pip` |
| Git | Any | |
| Alpaca account | Paper | [Sign up free](https://alpaca.markets/) |
| Google Colab | Free tier | T4 GPU recommended for training |

---

## 1 — Clone & Install

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/deepscalper_copilot.git
cd deepscalper_copilot/algo_trader

# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# For CPU-only PyTorch (saves ~1.5 GB — Python 3.13 requires PyTorch 2.6+):
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

---

## 2 — Create `.env` File

Create `algo_trader/.env` (never commit this file):

```env
ALPACA_API_KEY=PKXXXXXXXXXXXXXXXXXXXXXXXX
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

To get your keys:
1. Log in to [alpaca.markets](https://alpaca.markets/).
2. Go to **Paper Trading** → **API Keys** → Generate new key pair.

---

## 3 — Train Models (Google Colab)

Run the notebooks **in order**:

| Notebook | Purpose | Runtime |
|----------|---------|---------|
| `01_fetch_training_data.ipynb` | Pull 6 months of 1-min BTC/USD bars from Alpaca Crypto API | CPU, ~30 min |
| `02_feature_engineering.ipynb` | Compute 11 macro + 4 micro features (dual-mode: OHLCV-proxy for training); save bar-level arrays | CPU, ~15 min |
| `03_train_deepscalper.ipynb` | Train `DeepScalperNet V2` (BDQ N_DIR=2/N_SIZE=1 + PER + Hindsight + Vol Aux); 70/10/20 split | **T4 GPU**, ~4–8 h |
| `04_export_and_push_weights.ipynb` | Validate checkpoint format, copy `.pth` file and push to GitHub | CPU, ~5 min |
| `05_backtest_validation.ipynb` | Bar-by-bar greedy backtest on 20% held-out test set | CPU, ~20 min |
| `06_sharpe_diagnosis.ipynb` | Post-training Sharpe decomposition — diagnose reward signal quality | CPU, ~10 min |
| `07_lob_recorder.ipynb` | Record real Alpaca LOB snapshots (top 3 bid/ask levels) for live inference | CPU, ongoing |

### Before running notebooks:

1. Add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` to **Colab Secrets** (🔑 icon).
2. Add `GITHUB_TOKEN` (with `repo` scope) to Colab Secrets (needed for notebook 04).
3. In each notebook, set `REPO_URL` to your forked repository URL.
4. Enable **T4 GPU** for notebook 03: `Runtime → Change runtime type → T4 GPU`.

---

## 4 — Pull Trained Weights

After the training pipeline pushes weights to GitHub:

```bash
# From the repo root
git pull origin main

# Verify the BTC/USD weight file is present
python -c "
from pathlib import Path
w = list(Path('algo_trader/weights').glob('*.pth'))
print(f'{len(w)} weight file(s) found.')
"
```

Expected output: `1 weight file(s) found.`

---

## 5 — Launch the System

```bash
cd algo_trader
python main.py
```

On startup, `main.py`:
1. Loads `.env` and validates credentials against the Alpaca paper crypto API.
2. Verifies the BTC/USD `.pth` weight file exists.
3. Starts the Lumibot trading engine (`CryptoCitruitBreaker` strategy) in a background thread.
4. Opens the PyQt5 live dashboard in the main thread.

To stop: close the dashboard window or press `Ctrl+C`.

---

## 6 — Run Tests

```bash
cd algo_trader
pytest tests/ -v
```

Tests cover:
- `test_config.py` — Config constants and V2 parameter validation (31 tests)
- `test_utils.py` — Feature engineering: macro/micro features, metrics helpers (35 tests)
- `test_environment.py` — ScalperEnv V2: Gymnasium spaces, reset, step, reward shaping, edge cases (43 tests)
- `test_architecture.py` — Network shapes, forward pass, gradient flow for V2 dims (25 tests)
- `test_agent.py` — SumTree, PER buffer, BDQ agent, epsilon decay, save/load (40 tests)
- `test_data_bridge.py` — Thread-safe DataBridge properties and trade signals (26 tests)
- `test_crypto_circuit_breaker.py` — CryptoCitruitBreaker 24/7 halt logic (31 tests)
- `test_risk.py` — Kelly sizing, ATR stop/TP calculation
- `test_state_builder.py` — Observation dict format `{lob, priv, macro}`, shapes, dtype, NaN safety
- `test_circuit_breakers.py` — Legacy equity circuit breakers

---

## Key Parameters (config.py)

| Parameter | V2 Value | V1 Value | Description |
|-----------|----------|----------|-------------|
| `TRADING_UNIVERSE` | `['BTC/USD']` | S&P 100 (100 tickers) | Active trading universe |
| `LOOKBACK_BARS` | 10 | 60 | 1-min bars in observation window |
| `MACRO_DIM` | 11 | 11 | Macro features per bar (Table 2 of paper) |
| `LOB_DIM` | 4 | 5 | Micro features (dual-mode: spread/imbalance/depth/mid_move) |
| `PRIV_DIM` | 2 | 2 | Private state: (position flag, unrealised P&L %) |
| `N_DIR` | 2 | 3 | Direction branch: 0=FLAT, 1=LONG (no short selling) |
| `N_SIZE` | 1 | 4 | Size branch: 1 action — Kelly Criterion sizes externally |
| `ACTION_DIM` | 2 | 12 | Binary action space: FLAT or LONG |
| `GRU_HIDDEN` | 128 | 128 | GRU hidden units per stream in MicroEncoder |
| `MACRO_EMBED_DIM` | 64 | 64 | MacroEncoder MLP output dimension |
| `FC_HIDDEN` | 128 | 128 | BDQ head fully-connected layer width |
| `HINDSIGHT_HORIZON` | 10 | 60 | Look-ahead bars h for hindsight bonus (TradeMaster: forward_num_day=5) |
| `HINDSIGHT_WEIGHT` | 0.2 | 0.01 | Bonus coefficient ω (TradeMaster: future_weights=0.2) |
| `AUX_TASK_ETA` | 1.0 | 1.0 | Volatility auxiliary task weight η (Section 4.4) |
| `TRANSACTION_COST_LAMBDA` | 0.0025 | 0.0001 | Alpaca crypto taker fee (25 bps per side) |
| `KELLY_FRACTION` | 0.5 | 0.5 | Half-Kelly position sizing |
| `ATR_STOP_MULTIPLIER` | 2.0 | 2.0 | Stop loss = 2× ATR from entry |
| `ATR_TP_MULTIPLIER` | 4.0 | 4.0 | Take profit = 4× ATR from entry |
| `MAX_POSITION_PCT` | 0.95 | 0.03 | Max portfolio allocation per position (single pair; Kelly caps within) |
| `MAX_DAILY_LOSS_PCT` | 0.05 | 0.03 | Trading halted after 5% rolling 24-hr loss |
| `VOLATILITY_HALT_MULTIPLIER` | 4.0 | — | Halt if 5-min ATR > 4× its 72-hr rolling average |
| `CONSECUTIVE_LOSS_HALT` | 8 | — | Halt 30 min after 8 consecutive losing trades |
| `NO_TRADE_OPEN_BUFFER_MIN` | None | 15 | Not applicable — crypto is 24/7 |
| `NO_TRADE_CLOSE_BUFFER_MIN` | None | 15 | Not applicable — crypto is 24/7 |
| `CLOSE_ALL_EOD` | False | True | No end-of-day close in crypto |
| `LOB_LEVELS` | 3 | — | Top 3 bid/ask levels from Alpaca orderbook |
| `USE_REAL_LOB_INFERENCE` | True | — | Use real Alpaca orderbook during live inference |
| `USE_PROXY_LOB_TRAINING` | True | — | OHLCV-proxy LOB features during training (historical LOB unavailable) |
| `TRAIN_SPLIT` | 0.70 | — | 70% training (time-ordered, no shuffling) |
| `VAL_SPLIT` | 0.10 | — | 10% validation (model selection) |
| `TEST_SPLIT` | 0.20 | — | 20% test (final evaluation only) |
| `STARTING_CAPITAL` | 5000.00 | 5000.00 | Paper account starting capital |

---

## Alpaca Crypto Note

**V2 uses the Alpaca Crypto Trading API**, which differs from the equities API:
- **No short selling** — Alpaca crypto does not support short positions. The action space is binary: 0=FLAT, 1=LONG.
- **24/7 trading** — No market open/close buffers or end-of-day closes. Circuit breakers use rolling 24-hour windows instead.
- **Minimum balance** — Alpaca crypto paper accounts require no minimum; `STARTING_CAPITAL=5000.00` is used as the simulation starting point.
- **Taker fee** — Alpaca charges 25 bps per crypto trade. `TRANSACTION_COST_LAMBDA=0.0025` reflects this in training rewards.
- **LOB data** — Real-time top-3 bid/ask snapshots are available via `alpaca-py`. Historical LOB data is unavailable, so training uses OHLCV-proxy features (`USE_PROXY_LOB_TRAINING=True`).

**Effective June 4, 2026**, Alpaca has retired the Pattern Day Trader (PDT) rule for equities — this does not affect the V2 crypto strategy.

---

## Live Trading Checklist

Before switching to a live account:

- [ ] Complete at least 30 days of paper trading with consistent positive Sharpe
- [ ] Review backtest results from notebook 05 — target Sharpe > 1.0
- [ ] Run notebook 06 (`sharpe_diagnosis`) to confirm reward signal quality
- [ ] Verify circuit breakers trigger correctly (check `algo_trader.log`)
- [ ] Confirm `CryptoCitruitBreaker` rolling-loss and ATR-spike halts activate as expected
- [ ] Change `"PAPER": True` to `"PAPER": False` in `execution/broker.py`
- [ ] Add live API keys to `.env` (different from paper keys)
- [ ] Set `STARTING_CAPITAL` to your actual account balance in `config.py`
- [ ] Set `MAX_DAILY_LOSS_PCT` conservatively (0.01–0.02) for first live week

---

## Security Notes

- **Never commit `.env`** — it is in `.gitignore`.
- **Never hard-code API keys** in any source file.
- Weight files (`.pth`) may be large — consider [Git LFS](https://git-lfs.github.com/) (see `weights/README.md`).
- The Alpaca paper API key has no real-money access even if leaked, but rotate it immediately if exposed.

---

## License

MIT — see `LICENSE` for details.
