# AlgoTrader — DeepScalper × Alpaca Paper Trading System

End-to-end algorithmic intraday trading system for the **S&P 100** using:
- **DeepScalper** — Branching DQN (BDQ) + PER + Hindsight Bonus + Volatility Auxiliary Task (CIKM '22, Sun et al.)
- **Lumibot** — Python broker execution framework
- **Alpaca** — Paper trading API (no real money at risk)
- **PyQt5** — Live dashboard with equity bar, positions table, confidence panel, trade log

---

## Architecture Overview

```
algo_trader/
├── main.py                          # Entry point — starts everything
├── config.py                        # All constants and parameters
├── tickers.py                       # S&P 100 ticker list (100 symbols)
├── requirements.txt                 # Pinned Python dependencies
├── weights/                         # Trained .pth files (one per ticker)
│   └── README.md
├── execution/
│   ├── broker.py                    # Alpaca paper broker config
│   ├── risk.py                      # Kelly sizing + ATR stops
│   ├── circuit_breakers.py          # Session time guards + daily loss halt
│   ├── state_builder.py             # Live bars → {lob, priv, macro} observation dict for DeepScalperNet
│   └── strategy.py                  # Lumibot Strategy class (core loop)
├── dashboard/
│   ├── data_bridge.py               # Thread-safe state shared between threads
│   ├── equity_bar.py                # Top metrics bar widget
│   ├── positions_table.py           # Open positions table
│   ├── confidence_panel.py          # Per-stock signal cards
│   ├── trade_log.py                 # Trade event feed
│   └── main_window.py               # Root QMainWindow
├── colab/
│   ├── deepscalper/
│   │   ├── architecture.py          # DeepScalperNet — BDQ: MacroEncoder + dual-stream GRU MicroEncoder + VolatilityHead
│   │   ├── agent.py                 # DeepScalperAgent — BDQ Double-DQN + PER + hindsight bonus + vol auxiliary task
│   │   ├── environment.py           # ScalperEnv — Dict obs {lob,priv,macro}, MultiDiscrete([3,4]) action space
│   │   └── utils.py                 # compute_macro_features (11) + compute_micro_features (5) + metrics helpers
│   ├── 01_fetch_training_data.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_train_deepscalper.ipynb
│   ├── 04_export_and_push_weights.ipynb
│   └── 05_backtest_validation.ipynb
└── tests/
    ├── test_risk.py
    ├── test_state_builder.py
    └── test_circuit_breakers.py
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

Run the five Colab notebooks **in order**:

| Notebook | Purpose | Runtime |
|----------|---------|---------|
| `01_fetch_training_data.ipynb` | Pull 6 months of 1-min bars from Alpaca | CPU, ~30 min |
| `02_feature_engineering.ipynb` | Compute 11 macro + 5 micro features; save bar-level arrays | CPU, ~15 min |
| `03_train_deepscalper.ipynb` | Train one `DeepScalperNet` per ticker (BDQ + PER + Hindsight + Vol Aux) | **T4 GPU**, ~4–8 h |
| `04_export_and_push_weights.ipynb` | Validate checkpoint format, copy `.pth` files and push to GitHub | CPU, ~5 min |
| `05_backtest_validation.ipynb` | Bar-by-bar greedy backtest on 20% held-out val set | CPU, ~20 min |

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

# Verify all 100 weight files are present
python -c "
from pathlib import Path
w = list(Path('algo_trader/weights').glob('*.pth'))
print(f'{len(w)} weight files found.')
"
```

Expected output: `100 weight files found.`

---

## 5 — Launch the System

```bash
cd algo_trader
python main.py
```

On startup, `main.py`:
1. Loads `.env` and validates credentials against the Alpaca paper API.
2. Verifies all 100 `.pth` weight files exist.
3. Starts the Lumibot trading engine in a background thread.
4. Opens the PyQt5 live dashboard in the main thread.

To stop: close the dashboard window or press `Ctrl+C`.

---

## 6 — Run Tests

```bash
cd algo_trader
pytest tests/ -v
```

Tests cover:
- `test_risk.py` — Kelly sizing, ATR stop/TP calculation
- `test_state_builder.py` — Dict observation format `{lob, priv, macro}`, shapes, dtype, NaN safety, backward-compat shim
- `test_circuit_breakers.py` — Session time guards, daily loss halt, reset

---

## Key Parameters (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LOOKBACK_BARS` | 60 | 1-min bars in observation window |
| `MACRO_DIM` | 11 | Macro features per bar (Table 2 of paper) |
| `LOB_DIM` | 5 | Micro/intrabar features (LOB proxy) |
| `PRIV_DIM` | 2 | Private state: (position flag, unrealised P&L %) |
| `N_DIR` | 3 | Direction branch: 0=HOLD 1=BUY 2=SELL |
| `N_SIZE` | 4 | Size branch: 0=25% 1=50% 2=75% 3=100% of max notional |
| `GRU_HIDDEN` | 128 | GRU hidden units per stream in MicroEncoder |
| `MACRO_EMBED_DIM` | 64 | MacroEncoder MLP output dimension |
| `FC_HIDDEN` | 128 | BDQ head fully-connected layer width |
| `HINDSIGHT_HORIZON` | 60 | Look-ahead bars h for hindsight bonus (Section 4.2) |
| `HINDSIGHT_WEIGHT` | 0.01 | Bonus coefficient w in r_H = r_t + w·log(P_{t+h}/P_t)·pos |
| `AUX_TASK_ETA` | 1.0 | Volatility auxiliary task weight η (Section 4.4) |
| `KELLY_FRACTION` | 0.5 | Half-Kelly position sizing |
| `ATR_STOP_MULTIPLIER` | 2.0 | Stop loss = 2× ATR from entry |
| `ATR_TP_MULTIPLIER` | 4.0 | Take profit = 4× ATR from entry |
| `MAX_POSITION_PCT` | 0.03 | Max 3% of portfolio per position |
| `MAX_DAILY_LOSS_PCT` | 0.03 | Trading halted after 3% daily loss |
| `NO_TRADE_OPEN_BUFFER_MIN` | 15 | No trades first 15 min of session |
| `NO_TRADE_CLOSE_BUFFER_MIN` | 15 | No trades last 15 min of session |
| `STARTING_CAPITAL` | 5000.00 | Paper account starting capital |

---

## Alpaca Intraday Margin Note

**Effective June 4, 2026**, Alpaca has retired the Pattern Day Trader (PDT) rule:
- Minimum account balance: **$2,000** (down from $25,000 PDT threshold).
- Eligible accounts receive **4× intraday buying power**.
- This system defaults to `STARTING_CAPITAL=5000.00` to comfortably exceed the minimum.
- Set `PAPER: True` in `broker.py` (default) to trade with paper money only.

---

## Live Trading Checklist

Before switching to a live account:

- [ ] Complete at least 30 days of paper trading with consistent positive Sharpe
- [ ] Review backtest results from notebook 05 — target Sharpe > 1.0
- [ ] Verify circuit breakers trigger correctly (check `algo_trader.log`)
- [ ] Confirm weight files are up to date (retrain if market regime has shifted)
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
