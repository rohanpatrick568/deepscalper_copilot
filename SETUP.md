# DeepScalper Copilot Setup Guide

This project is now an equities-first DeepScalper stack with TradeMaster-aligned DQN training controls.

## Current System Summary

- Market mode: US equities (paper trading only)
- Action semantics: 3-direction policy (SHORT, FLAT, LONG)
- Live strategy class: EquityDeepScalper
- Agent training core: TradeMaster-style controls (uniform replay default, repeat_times, clip_grad_norm, soft_update_tau, state_value_tau, static explore_rate)
- Canonical training constants in config.py:
  - EPOCHS = 20
  - HORIZON_LEN = 128
  - BUFFER_SIZE = 1_000_000
  - LEARNING_RATE = 1e-3
  - GAMMA = 0.9
  - REPEAT_TIMES = 1.0
  - EXPLORE_RATE = 0.25

## Prerequisites

- Python 3.13+
- pip 23+
- Git
- Alpaca paper account

## Install

```powershell
cd deepscalper_copilot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r algo_trader/requirements.txt
```

## Configure Environment

Create algo_trader/.env:

```env
ALPACA_API_KEY=YOUR_PAPER_KEY
ALPACA_SECRET_KEY=YOUR_PAPER_SECRET
```

Notes:

- main.py validates keys and account connectivity before startup.
- execution/broker.py enforces PAPER=True.

## Training Pipeline (Colab)

Run notebooks in order:

1. colab/01_fetch_training_data.ipynb
2. colab/02_feature_engineering.ipynb
3. colab/03_train_deepscalper.ipynb
4. colab/04_export_and_push_weights.ipynb
5. colab/05_backtest_validation.ipynb
6. colab/06_sharpe_diagnosis.ipynb
7. colab/07_lob_recorder.ipynb (optional data utility)

Key parity notes for training:

- Notebook 03 is wired to canonical keys: epochs, buffer_size, horizon_len, repeat_times, soft_update_tau, state_value_tau, explore_rate.
- Agent update cadence uses agent.update_net().
- Active training flow no longer uses auxiliary volatility loss.

## Weights

Weights are expected in algo_trader/weights as one file per configured symbol:

```text
{SYMBOL}.pth
```

Example for AAPL:

```text
AAPL.pth
```

## Run Live Paper App

```powershell
cd algo_trader
python main.py
```

Startup behavior:

1. Validate .env credentials and Alpaca paper account
2. Validate weight file presence for TRADING_UNIVERSE
3. Start Lumibot engine thread
4. Start PyQt dashboard

Optional headless mode:

```powershell
$env:ALGO_TRADER_HEADLESS="1"
python main.py
```

## Tests

```powershell
cd algo_trader
pytest -q
```

Current baseline is green with one intentional legacy skip.

## Important Files

- algo_trader/config.py
- algo_trader/main.py
- algo_trader/execution/strategy.py
- algo_trader/colab/deepscalper/agent.py
- algo_trader/colab/deepscalper/architecture.py
- algo_trader/colab/03_train_deepscalper.ipynb
- algo_trader/HOW_TRAINING_AND_LIVE_WORK.md
