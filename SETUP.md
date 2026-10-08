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

- Python 3.12+ (Windows 11 and Colab-supported runtimes)
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
ALPACA_DATA_FEED=iex
ALPACA_ADJUSTMENT=raw
ALGO_TRADER_RUN_MODE=preflight
```

Copy [algo_trader/.env.example](algo_trader/.env.example) rather than
inventing variable names. The default universe is one symbol (`AAPL`) and all
paths resolve from the project, not the current working directory.

Notes:

- main.py validates keys and account connectivity before startup.
- execution/broker.py enforces PAPER=True and the configured feed/adjustment.
- A missing or incompatible checkpoint fails startup with the symbol and path.

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

## Safe operational workflow

All commands below are PowerShell commands and do not place orders unless the
explicit smoke-test mode is selected:

```powershell
cd .\algo_trader
python -m workflow offline       # local artifacts, no network/orders
python -m workflow preflight     # read-only account/data/calendar inspection
$env:ALGO_TRADER_RUN_MODE="dry-run"
python main.py                   # model path, broker boundary rejects orders
```

For the supervised paper smoke test, use a separate paper account, confirm
preflight output, and set the bounded mode explicitly:

```powershell
$env:ALGO_TRADER_RUN_MODE="paper-smoke"
python main.py
```

`paper-smoke` is intentionally not a default. Keep the dashboard open and
stop after the bounded test window; never use live credentials.

Normal model-driven paper mode:

```powershell
$env:ALGO_TRADER_RUN_MODE="paper"
python main.py
```

Startup behavior:

1. Validate .env credentials and Alpaca paper account
2. Validate checkpoint compatibility for the canonical universe
3. Start Lumibot engine thread
4. Start PyQt dashboard

Optional headless mode:

```powershell
$env:ALGO_TRADER_HEADLESS="1"
python main.py
```

## Local or Google Colab training/evaluation

The notebooks in `algo_trader/colab/` are thin launchers. They must export the
same versioned checkpoint/manifest format used locally. For a small one-symbol
offline smoke test:

```powershell
cd .\algo_trader
python -m pytest -q tests/test_agent.py tests/test_environment.py
python .\backtest_validation_local.py --symbol AAPL --smoke
```

In Colab, mount Drive, set `WEIGHTS_DIR` and `DATA_DIR` to Drive paths, select
`AAPL`, run notebooks 01→05, and export both the best-validation checkpoint
and its manifest. To resume, mount the same Drive directory and point notebook
03 at the saved checkpoint; do not silently switch to local paths. Download
the `.pth` and manifest into `algo_trader/weights/`, then run
`python -m workflow offline` before any paper session. Colab GPU availability,
Drive permissions, Alpaca credentials, and historical-data entitlement require
an interactive user session and are not CI checks.

Operational readiness (configuration, broker safety, state recovery) is
separate from model-performance validation (chronological validation/holdout,
costs, and acceptance gates). A passing offline smoke test does not approve a
model for trading.

## Tests

```powershell
cd algo_trader
pytest -q
```

CI also runs installation/import and tests on Windows.

## Important Files

- algo_trader/config.py
- algo_trader/main.py
- algo_trader/execution/strategy.py
- algo_trader/colab/deepscalper/agent.py
- algo_trader/colab/deepscalper/architecture.py
- algo_trader/colab/03_train_deepscalper.ipynb
- algo_trader/HOW_TRAINING_AND_LIVE_WORK.md
