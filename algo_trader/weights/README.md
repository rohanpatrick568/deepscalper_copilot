# weights/

This directory contains pre-trained DeepScalper model weights — one `.pth` file per S&P 100 ticker.

## Contents

After a successful training run (Colab notebook `03_train_deepscalper.ipynb`) and weight export (`04_export_and_push_weights.ipynb`), this directory will contain 100 files:

```
AAPL.pth
MSFT.pth
AMZN.pth
... (one per S&P 100 ticker)
CME.pth
```

## How to Pull Weights

```bash
# Clone or update the repo (weights are tracked via Git LFS)
git lfs install
git pull origin main

# Verify all 100 files are present
python -c "
from pathlib import Path
from tickers import SP100_TICKERS
missing = [t for t in SP100_TICKERS if not (Path('weights') / f'{t}.pth').exists()]
print(f'Missing: {missing}' if missing else 'All 100 weight files present.')
"
```

## Git LFS Setup

Weight files (`.pth`) are stored using [Git Large File Storage (LFS)](https://git-lfs.github.com/) to avoid bloating the main repository history.

```bash
# One-time setup
git lfs install
git lfs track "*.pth"
git add .gitattributes
git commit -m "track .pth files with Git LFS"
```

## File Naming Convention

Each file is named `{TICKER}.pth` where `{TICKER}` matches exactly the symbol in `tickers.py` (e.g. `BRK.B.pth` for Berkshire Hathaway).

## Model Architecture

Each `.pth` file contains the `state_dict` of a `DuelingQNetwork` (see `colab/deepscalper/architecture.py`) trained on 6 months of 1-minute OHLCV data for the corresponding ticker. The state dict is loaded with:

```python
import torch
from colab.deepscalper.architecture import DuelingQNetwork
from config import INPUT_DIM, ACTION_DIM, LOOKBACK_BARS, HIDDEN_SIZE, FC_SIZE, DROPOUT_RATE

model = DuelingQNetwork(LOOKBACK_BARS, INPUT_DIM, ACTION_DIM, HIDDEN_SIZE, FC_SIZE, DROPOUT_RATE)
model.load_state_dict(torch.load("weights/AAPL.pth", map_location="cpu"))
model.eval()
```

## Retraining

To retrain all models from scratch, run the Colab notebooks in order:

1. `colab/01_fetch_training_data.ipynb`
2. `colab/02_feature_engineering.ipynb`
3. `colab/03_train_deepscalper.ipynb`
4. `colab/04_export_and_push_weights.ipynb`
