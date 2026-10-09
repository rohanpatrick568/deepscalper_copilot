# DeepScalper operating guide

One guide from a clean machine to a supervised Alpaca **paper** session.

Two things are verified separately and must not be confused:

| | What it proves | How it is checked |
| --- | --- | --- |
| **Operational readiness** | The software installs, decides, submits, reconciles, recovers, and shuts down correctly | Offline tests, CI, read-only preflight, bounded smoke order |
| **Model performance** | The policy is actually worth trading | Training on **real** market data, passing the acceptance gates on validation *and* an untouched holdout |

A tiny synthetic run can only ever prove the first. Synthetic artifacts are
tagged `"data_source": "synthetic"` and are refused by `promote`.

Supported local execution is **CPU** on macOS (Apple silicon) and Windows 11.
Google Colab may additionally use CUDA.

> **Only one computer may run the order-submitting engine for a paper account
> at a time.** Two engines on one account will fight over the same positions
> and protective orders. Keep virtual environments, credentials, and execution
> state local to each machine; sync source through Git and move data/models as
> files.

---

## 1. Clean setup and dependency verification

Requires Python 3.12 and Git. Use a separate virtual environment on each
computer — never sync `.venv` between machines.

### macOS (Terminal, Apple silicon)

```bash
git clone https://github.com/rohanpatrick568/deepscalper_copilot.git
cd deepscalper_copilot
git checkout fix/supervised-paper-reliability

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r algo_trader/requirements.txt

python -m pip check
python -c "import platform; print(platform.platform(), platform.machine())"
python -c "import alpaca, gymnasium, lumibot, numpy, pandas, torch; print('imports OK')"
```

### Windows 11 (PowerShell)

```powershell
git clone https://github.com/rohanpatrick568/deepscalper_copilot.git
cd deepscalper_copilot
git checkout fix/supervised-paper-reliability

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r algo_trader\requirements.txt

python -m pip check
python -c "import platform; print(platform.platform(), platform.machine())"
python -c "import alpaca, gymnasium, lumibot, numpy, pandas, torch; print('imports OK')"
```

`python -m pip check` must print `No broken requirements found`. Lumibot 3.10.0
requires NumPy `<2`, which is why NumPy is pinned to 1.26.4.

### Run the test suite

```bash
cd algo_trader
python -m pytest -q
```

```powershell
cd algo_trader
python -m pytest -q
```

Colab and other training-only environments can install
`algo_trader/requirements-core.txt` instead, which omits the desktop GUI
dependencies.

---

## 2. Choose Local or Colab, then train and evaluate

The choice is explicit and never changed silently. Both locations run the
**same** module (`colab.deepscalper.training`), the same feature and policy
definitions, the same chronological splits, and produce the same
checkpoint + manifest format.

Open the chooser without any model or broker credentials:

```bash
cd algo_trader
python main.py --training-setup
```

or from the CLI:

```bash
python -m workflow setup --location local     # or --location colab
```

### 2a. Prepare data

Real training data requires Alpaca **data** credentials (section 4). Training
itself does not: if you already have a feature file, or you copied one from
another machine, no credentials are needed.

```bash
cd algo_trader
python -m colab.deepscalper.data fetch --symbol AAPL \
  --start 2024-01-01 --end 2025-01-01 --feed iex --adjustment raw \
  --output data/AAPL_raw.parquet
python -m colab.deepscalper.data features \
  --input data/AAPL_raw.parquet --output data/AAPL_features.npz
```

### 2b. Local CPU training

```bash
cd algo_trader
python -m colab.deepscalper.training train \
  --data data/AAPL_features.npz \
  --output-dir runs/AAPL --symbol AAPL \
  --location local --device cpu --epochs 20 \
  --min-position-changes 2
```

```powershell
cd algo_trader
python -m colab.deepscalper.training train `
  --data data\AAPL_features.npz `
  --output-dir runs\AAPL --symbol AAPL `
  --location local --device cpu --epochs 20 `
  --min-position-changes 2
```

* **Resume**: re-run the same command with `--resume`. Resume refuses to
  continue if the symbol, data, model, feature schema, or policy configuration
  changed.
* **Cancel**: pass `--cancel-file runs/AAPL/CANCEL` and create that file to
  stop cleanly. Interrupting with Ctrl+C also keeps the last valid checkpoint.
* **Device**: `--device cpu` is the supported local default. `--device auto`
  falls back to CPU and reports the fallback. `--device cuda` fails loudly
  rather than silently training elsewhere.

### 2c. Evaluate and read the decision

```bash
python -m colab.deepscalper.training evaluate \
  --checkpoint runs/AAPL/best.pth --manifest runs/AAPL/best.manifest.json \
  --data data/AAPL_features.npz --split holdout --device cpu

python -m colab.deepscalper.training verify \
  --checkpoint runs/AAPL/best.pth --manifest runs/AAPL/best.manifest.json
```

**Rejection handling.** `train` exits `2` when the model is not accepted, and
`best.manifest.json` lists every failed gate under `"rejections"`, for example:

```
validation.position_changes 0 < required 2 (a flat, zero-trade policy cannot be approved)
holdout.return -0.013000 < required 0.000000
holdout.max_drawdown 0.180000 > allowed 0.100000
```

Gates cover net return after modelled costs on **both** validation and the
untouched holdout, holdout drawdown, and a positive activity requirement. A
flat, zero-trade policy can never be approved. **Do not lower the thresholds
to force an approval** — retrain, change features, or accept that the model is
not tradeable. Evaluation applies the same entry filters, minimum holding
period, cooldown, long-only rule, and protective exits that paper execution
applies; simulated protective exits approximate the runtime ATR stops on bar
closes, so this is approximate fill behaviour, not fill parity.

### 2d. Google Colab

Open `algo_trader/colab/03_train_deepscalper.ipynb` in Colab (the chooser's
**Open Colab launcher** button links to it). You start and authorize the
session yourself; nothing here launches remote compute for you.

1. **Bootstrap cell** — mounts Drive, clones/updates the repository, and checks
   out the revision in `REVISION` (set it to `fix/supervised-paper-reliability`
   or a commit SHA). Safe to re-run after a disconnect.
2. **Device cell** — `DEVICE = 'auto'` uses CUDA when present and otherwise
   reports a CPU fallback; `'cuda'` requires a GPU runtime
   (Runtime → Change runtime type → T4 GPU); `'cpu'` forces CPU.
3. **Config cell** — set `DATA` and `OUTPUT` under
   `/content/drive/MyDrive/...` so checkpoints survive a runtime restart.
4. **Training cell** — resumes automatically when `latest.pth` and
   `training_state.json` exist. Re-run it after an interruption.
5. **Verify cell** — prints `accepted`, the metrics, and any rejections.
6. **Download cell** — downloads `best.pth` and `best.manifest.json`.

---

## 3. Move artifacts between machines and load on CPU

Transfer `best.pth` and `best.manifest.json` as files (Drive, USB, scp). Do not
commit them.

Confirm the artifact loads on CPU regardless of the training device:

```bash
cd algo_trader
python -m scripts.verify_artifact_cpu \
  --checkpoint best.pth --manifest best.manifest.json
```

Import an **approved** model into the paper runtime:

```bash
python -m colab.deepscalper.training promote \
  --checkpoint best.pth --manifest best.manifest.json \
  --weights-dir weights --symbol AAPL
```

```powershell
python -m colab.deepscalper.training promote `
  --checkpoint best.pth --manifest best.manifest.json `
  --weights-dir weights --symbol AAPL
```

`promote` refuses artifacts that failed the gates, artifacts trained on
synthetic data, mismatched checksums, and stale feature/action/policy schema
versions. Changing policy semantics bumps the policy version, which forces
affected artifacts to be re-evaluated before they can be promoted again.

---

## 4. Paper credentials

Use **paper** keys only, from <https://app.alpaca.markets/paper/dashboard/overview>.
Never commit or print them.

### Shell session variables (not persisted)

macOS Terminal — lasts only for the current terminal session:

```bash
export ALPACA_API_KEY='YOUR_PAPER_API_KEY'
export ALPACA_SECRET_KEY='YOUR_PAPER_SECRET_KEY'
```

Windows PowerShell — lasts only for the current PowerShell window:

```powershell
$env:ALPACA_API_KEY = 'YOUR_PAPER_API_KEY'
$env:ALPACA_SECRET_KEY = 'YOUR_PAPER_SECRET_KEY'
```

To persist on Windows for your user account (new windows only):

```powershell
[Environment]::SetEnvironmentVariable('ALPACA_API_KEY', 'YOUR_PAPER_API_KEY', 'User')
[Environment]::SetEnvironmentVariable('ALPACA_SECRET_KEY', 'YOUR_PAPER_SECRET_KEY', 'User')
```

### The `.env` file

The project reads exactly one file: **`algo_trader/.env`**. No other location
is searched, and the current working directory is irrelevant.

**Precedence: real environment variables win.** `.env` only fills in variables
that are *not* already set in the environment. If `ALPACA_API_KEY` is exported
in your shell, the `.env` value is ignored.

```bash
cp algo_trader/.env.example algo_trader/.env   # then edit it
```

```powershell
Copy-Item algo_trader\.env.example algo_trader\.env   # then edit it
```

`algo_trader/.env` is gitignored. Keep it out of Git, screenshots, and logs.

---

## 5. Offline checks → preflight → smoke order → supervised session

Run these **in order**. Each step is a gate for the next.

### 5a. Offline (no credentials, no network)

```bash
cd algo_trader
python -m pytest -q
python -m workflow offline
python -m workflow dry-run
```

`dry-run` executes the real decision loop and reports
`"broker_submissions": 0`. The no-order modes are enforced at the broker
boundary, so offline, preflight, and dry-run cannot mutate the account through
any submission, replacement, cancellation, bulk-order, or close-position path.

### 5b. Read-only preflight (credentials, no orders)

```bash
python -m workflow preflight
```

Inspects account status and permissions, buying power, positions, open orders,
the clock/calendar, asset eligibility, and live bar freshness — **without
placing any order**. Model readiness is reported separately, so this
connectivity check does not require an approved model.

Proceed only when `"connectivity_ready": true`.

### 5c. Bounded paper smoke order (explicitly opted in, places 2 orders)

This is the first step that submits real paper orders. It is a deterministic
plumbing test and does **not** need an approved model. Run it during regular
market hours and watch it.

```bash
python -m workflow paper-smoke --confirm-paper-smoke \
  --symbol AAPL --notional 20 --timeout-seconds 120
```

```powershell
python -m workflow paper-smoke --confirm-paper-smoke `
  --symbol AAPL --notional 20 --timeout-seconds 120
```

It verifies the paper endpoint, requires an open session and a clean state for
the test symbol, uses stable client order IDs, and is capped at two orders
within the configured notional limits. It buys a small fractional notional and
sells only the position it created; unrelated holdings are untouched.

Success requires broker-confirmed terminal orders and a flat test position:

```json
{"flat_confirmed": true, "all_orders_terminal": true, ...}
```

**If it reports `UNRESOLVED PAPER-SMOKE EXPOSURE`**, it deliberately stopped
submitting rather than risk overselling. The message lists the order IDs,
client order IDs, and the remaining quantity. Open the Alpaca paper dashboard,
wait for or cancel the non-terminal order, and flatten the remaining quantity
yourself. Do not re-run the smoke test until the symbol is clean again.

### 5d. Supervised model-driven paper session

Only after 5a–5c pass **and** an approved model has been promoted:

```bash
cd algo_trader
python main.py
```

```powershell
cd algo_trader
python main.py
```

Stay at the machine. The engine is long-only, does not pyramid, uses bounded
fixed sizing with broker-held protection, halts new entries on the session loss
limit while continuing to manage open positions, and closes before the
exchange's early close.

### Stopping and recovery

* **Graceful stop** — close the dashboard window, or press **Ctrl+C**. Both run
  the same bounded, verified shutdown. Flattening is only reported as
  successful when the broker confirms it.
* **Exit status** — `0` clean; `1` engine failure; `2` shutdown not confirmed;
  `3` the engine did not stop within the bounded timeout. A non-zero status
  means **check the account manually**.
* **Emergency** — use the dashboard's cancel/flatten controls, or flatten
  directly in the Alpaca paper dashboard. The Alpaca UI is always the
  authoritative view.
* **Restart** — on startup the engine reconstructs positions, orders,
  protection, and the session-loss baseline from its ledger and reconciles them
  against Alpaca before any new entry is allowed.

### Where to look

| What | Where |
| --- | --- |
| Application log | `algo_trader/algo_trader.log` |
| Durable order/position/protection ledger | `algo_trader/runtime/execution-state.json` (or `EXECUTION_STATE_PATH`) |
| Decisions, skip reasons, feed and bar age | application log (`decision=`, `skip`, `feed`) |
| Authoritative orders and positions | Alpaca paper dashboard |
| Training runs and manifests | `algo_trader/runs/<SYMBOL>/` |
| Approved weights | `algo_trader/weights/` |

---

## Quick checklist

1. Install and `pip check` on each machine; run `pytest -q`.
2. Fetch real data, train Local or Colab, and read `"rejections"`.
3. Promote only an **accepted**, non-synthetic artifact.
4. `workflow offline` → `workflow dry-run` → `workflow preflight`.
5. `workflow paper-smoke --confirm-paper-smoke`, confirm flat.
6. `python main.py` on **one** machine, supervised.
