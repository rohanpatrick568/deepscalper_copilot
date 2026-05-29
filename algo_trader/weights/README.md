# weights Directory

This directory stores trained DeepScalper checkpoint files for the configured equities universe.

## Naming Convention

One checkpoint per symbol:

```text
{SYMBOL}.pth
```

Examples:

```text
AAPL.pth
MSFT.pth
NVDA.pth
```

If any symbol contains a slash in future extensions, loaders normalize with symbol.replace("/", "_").

## Expected Checkpoint Layout

Both layouts are supported by loaders:

1. Wrapped checkpoint with online_net key
2. Raw state_dict checkpoint

Loaders use:

```python
state_dict = ckpt.get("online_net", ckpt)
```

## Current Model Signature

DeepScalperNet forward returns:

- q_dir
- q_size

The old auxiliary volatility output is no longer part of the active model forward path.

## Verify Required Weight Files

```powershell
cd algo_trader
python -c "from pathlib import Path; from config import TRADING_UNIVERSE, WEIGHTS_DIR; missing=[s for s in TRADING_UNIVERSE if not (WEIGHTS_DIR / f'{s.replace('/', '_')}.pth').exists()]; print('Missing:'+','.join(missing) if missing else 'All required weight files present.')"
```

## Minimal Load Example

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

ckpt = torch.load("weights/AAPL.pth", map_location="cpu", weights_only=True)
state_dict = ckpt.get("online_net", ckpt)
model.load_state_dict(state_dict)
model.eval()
```

## How Weights Are Produced

Primary training/export flow:

1. colab/03_train_deepscalper.ipynb
2. colab/04_export_and_push_weights.ipynb
