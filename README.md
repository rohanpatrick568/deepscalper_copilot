# DeepScalper Copilot

Equities-first algorithmic trading project using a DeepScalper policy with a TradeMaster-aligned DQN training core, Alpaca paper execution through Lumibot, and a PyQt dashboard.

## Project Status

- Current market mode: US equities paper trading
- Direction policy: SHORT, FLAT, LONG
- Training core parity: TradeMaster-style controls now active
- Runtime validation: test suite green with one intentional legacy skip

## Repository Layout

```text
deepscalper_copilot/
├── README.md
├── SETUP.md
└── algo_trader/
    ├── main.py
    ├── config.py
    ├── HOW_TRAINING_AND_LIVE_WORK.md
    ├── backtest_validation_local.py
    ├── requirements.txt
    ├── colab/
    ├── execution/
    ├── dashboard/
    ├── tests/
    └── weights/
```

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies from algo_trader/requirements.txt.
3. Add Alpaca paper keys to algo_trader/.env.
4. Ensure weights exist for symbols in TRADING_UNIVERSE.
5. Run:

```powershell
cd algo_trader
python main.py
```

For full setup details, see SETUP.md.

## Training Pipeline

Run notebooks in order:

1. colab/01_fetch_training_data.ipynb
2. colab/02_feature_engineering.ipynb
3. colab/03_train_deepscalper.ipynb
4. colab/04_export_and_push_weights.ipynb
5. colab/05_backtest_validation.ipynb
6. colab/06_sharpe_diagnosis.ipynb

## Current Canonical Training Hyperparameters

Defined in algo_trader/config.py:

- EPOCHS = 20
- BATCH_SIZE = 64
- HORIZON_LEN = 128
- BUFFER_SIZE = 1_000_000
- LEARNING_RATE = 1e-3
- GAMMA = 0.9
- REPEAT_TIMES = 1.0
- CLIP_GRAD_NORM = 3.0
- SOFT_UPDATE_TAU = 0.0
- STATE_VALUE_TAU = 0.005
- EXPLORE_RATE = 0.25

## Documentation

- SETUP.md: installation, environment, training, run, and test steps
- algo_trader/HOW_TRAINING_AND_LIVE_WORK.md: architecture and parity details
- algo_trader/weights/README.md: checkpoint naming, loading, and verification

## Testing

```powershell
cd algo_trader
pytest -q
```

## Safety and Scope

- execution/broker.py is configured for paper trading
- main.py validates credentials and weight presence before startup
- This repository is currently intended for paper workflows and validation
