# weights/

This directory contains V2 DeepScalper model weights for crypto trading.

## Expected Files (V2)

V2 uses a BTC/USD-only universe by default, so the expected primary file is:

```
BTC_USD.pth
```

If the configured universe is expanded later, each pair should use the same naming pattern:

```
{PAIR_WITH_SLASH_REPLACED_BY_UNDERSCORE}.pth
```

Examples:

```
BTC_USD.pth
ETH_USD.pth
```

## Verify Weights

```bash
python -c "
from pathlib import Path
from config import CRYPTO_PAIRS
missing = [p for p in CRYPTO_PAIRS if not (Path('weights') / f'{p.replace('/', '_')}.pth').exists()]
print(f'Missing: {missing}' if missing else 'All crypto weight files present.')
"
```

## Model Architecture

Each `.pth` file contains a `state_dict` for `DeepScalperNet` trained with V2 dimensions:

- `MACRO_DIM = 11`
- `LOB_DIM = 4`
- `N_DIR = 2` (FLAT/LONG)
- `N_SIZE = 1`

Load example:

```python
import torch
from colab.deepscalper.architecture import DeepScalperNet
from config import MACRO_DIM, LOB_DIM, PRIV_DIM, GRU_HIDDEN, MACRO_EMBED_DIM, FC_HIDDEN, N_DIR, N_SIZE

model = DeepScalperNet(
	macro_dim=MACRO_DIM,
	lob_dim=LOB_DIM,
	priv_dim=PRIV_DIM,
	gru_hidden=GRU_HIDDEN,
	macro_embed=MACRO_EMBED_DIM,
	fc_hidden=FC_HIDDEN,
	n_dir=N_DIR,
	n_size=N_SIZE,
)
state = torch.load("weights/BTC_USD.pth", map_location="cpu")
state_dict = state.get("online_net", state)
model.load_state_dict(state_dict)
model.eval()
```

## Retraining Pipeline

Run notebooks in order:

1. `colab/01_fetch_training_data.ipynb`
2. `colab/02_feature_engineering.ipynb`
3. `colab/03_train_deepscalper.ipynb`
4. `colab/04_export_and_push_weights.ipynb`
