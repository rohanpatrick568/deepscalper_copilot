# DeepScalper supervised paper setup

DeepScalper is paper-only. The bounded pilot universe is `AAPL`, execution is
long-only without pyramiding, and the model size branch is disabled in favor
of fixed risk-capped sizing.

## 1. Windows 11 setup

Use 64-bit Python 3.12:

```powershell
git clone --branch fix/supervised-paper-reliability --single-branch https://github.com/rohanpatrick568/deepscalper_copilot.git
cd .\deepscalper_copilot
git rev-parse --verify HEAD
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r .\algo_trader\requirements.txt
& .\.venv\Scripts\python.exe -m pip check
Copy-Item .\algo_trader\.env.example .\algo_trader\.env
```

Fill only Alpaca **paper** credentials in `.env`. Keep `ALPACA_DATA_FEED=iex`
unless the account is entitled to SIP. Runtime paths resolve from
[`algo_trader/`](algo_trader/), not the shell working directory.

## 2. Offline validation

This command uses synthetic data and requires no credentials or network:

```powershell
cd .\algo_trader
& ..\.venv\Scripts\python.exe -m workflow offline
& ..\.venv\Scripts\python.exe -m workflow dry-run
```

`dry-run` executes a real strategy decision and creates an in-memory order,
while the broker boundary remains untouched.

To expose the training choice before credentials or model weights exist:

```powershell
& ..\.venv\Scripts\python.exe -m workflow setup --location local
& ..\.venv\Scripts\python.exe -m workflow setup --location colab
```

## 3. Read-only Alpaca preflight

```powershell
& ..\.venv\Scripts\python.exe -m workflow preflight
```

Preflight reads the paper account status, blocks, buying power, positions,
open orders, broker clock/calendar, AAPL eligibility, configured feed access,
and actual completed-bar freshness. It reports model readiness separately and
never submits, replaces, cancels, or closes an order.

## 4. Explicit bounded paper round trip

Use a supervised paper account and first confirm preflight reports no AAPL
position or open AAPL order:

```powershell
& ..\.venv\Scripts\python.exe -m workflow paper-smoke `
  --confirm-paper-smoke `
  --symbol AAPL `
  --notional 20 `
  --max-orders 2 `
  --timeout-seconds 120
```

This model-independent command verifies the paper endpoint, buys at most the
configured notional using one fractional market order, sells exactly the
reconciled test quantity with one order, and succeeds only after Alpaca
confirms AAPL flat and both test orders terminal. It refuses dirty AAPL state
and leaves unrelated symbols untouched. If a price jump makes the reconciled
position exceed `MAX_ORDER_NOTIONAL`, recovery does not bypass the cap: it
reports `UNRESOLVED PAPER-SMOKE EXPOSURE` and requires supervised manual
flattening in the paper account.

## 5. Local training and evaluation

Core training supports Windows CPU. Set `--device cuda` only when the installed
PyTorch build and hardware report CUDA support; the selected device is recorded
and is never changed silently.

Prepare a portable feature file from chronological raw AAPL minute bars:

```powershell
& ..\.venv\Scripts\python.exe -m colab.deepscalper.data features `
  --input .\data\AAPL_raw.parquet `
  --output .\data\AAPL_features.npz
```

Run a small synthetic plumbing test (not a performance approval):

```powershell
& ..\.venv\Scripts\python.exe -m colab.deepscalper.training train `
  --synthetic --tiny --epochs 1 --location local --device cpu `
  --output-dir .\runs\AAPL-smoke --symbol AAPL
```

Run/restart the production-shape one-symbol workflow:

```powershell
& ..\.venv\Scripts\python.exe -m colab.deepscalper.training train `
  --data .\data\AAPL_features.npz --epochs 20 --location local --device cpu `
  --output-dir .\runs\AAPL --symbol AAPL

& ..\.venv\Scripts\python.exe -m colab.deepscalper.training train `
  --data .\data\AAPL_features.npz --epochs 20 --location local --device cpu `
  --output-dir .\runs\AAPL --symbol AAPL --resume
```

Ctrl+C cancels local execution. `latest.pth` and `training_state.json` preserve
the last completed epoch; `--resume` restores model, optimizer, step, epoch,
and best-validation state. The replay buffer is deliberately not serialized
and this fact is recorded in the manifest.

Evaluate and promote only the selected accepted model:

```powershell
& ..\.venv\Scripts\python.exe -m colab.deepscalper.training evaluate `
  --data .\data\AAPL_features.npz `
  --checkpoint .\runs\AAPL\best.pth `
  --manifest .\runs\AAPL\best.manifest.json --split holdout --device cpu

& ..\.venv\Scripts\python.exe -m colab.deepscalper.training promote `
  --checkpoint .\runs\AAPL\best.pth `
  --manifest .\runs\AAPL\best.manifest.json `
  --weights-dir .\weights --symbol AAPL
```

Promotion rejects failed gates, checksum/schema mismatch, non-production
architecture, or the wrong symbol.

## 6. Google Colab training and resume

The user must open and authorize Colab; this repository does not launch paid
compute.

1. Open [`algo_trader/colab/01_fetch_training_data.ipynb`](algo_trader/colab/01_fetch_training_data.ipynb)
   in Colab.
2. The launcher mounts/remounts Drive, checks out
   `fix/supervised-paper-reliability`, prints the checked-out SHA, and installs
   `requirements-core.txt`. Re-running the cell fetches and fast-forwards the
   same branch instead of cloning `main`.
3. Add `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` to Colab secrets only for the
   credentialed data-fetch cell; synthetic training needs neither.
4. Run notebooks 01 and 02. They call the same `deepscalper.data` package and
   persist raw data and `AAPL_features.npz` in Drive.
5. In notebook 03, leave `--location colab` explicit. Set `RESUME=True`; after
   interruption it reuses Drive-backed `latest.pth`.
6. Run notebook 05 for greedy holdout evaluation, then notebook 04 to validate
   and export the accepted best checkpoint plus manifest.
7. Download `best.pth` and `best.manifest.json` from Drive to a local staging
   directory and run the same `verify` and `promote` commands shown above.

Colab GPU availability, Drive authorization, data entitlement, and the
interactive training duration are external checks and are not claimed by CI.

## 7. Model-driven paper session

Model-driven startup requires the promoted `weights/AAPL.pth` and
`weights/AAPL.manifest.json` pair:

```powershell
$env:ALGO_TRADER_RUN_MODE = "paper"
& ..\.venv\Scripts\python.exe .\main.py
```

The dashboard wires manual position close, emergency cancel/flatten, Local /
Google Colab guidance, and bounded window-close shutdown through reconciled
strategy controls. A nonzero exit means the engine failed or Alpaca did not
confirm flat/terminal state before timeout.

## Verification

```powershell
& ..\.venv\Scripts\python.exe -m compileall -q .
& ..\.venv\Scripts\python.exe -m pytest -q
```

Operational readiness means installation, preflight, no-order enforcement,
recovery, and broker reconciliation pass. Model-performance validation is
separate: a promoted model must pass chronological validation and untouched
holdout gates after realistic costs and terminal liquidation. A plumbing
smoke test never approves trading performance.
