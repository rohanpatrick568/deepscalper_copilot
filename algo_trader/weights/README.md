# Runtime model directory

Model-driven paper mode accepts one inseparable pair per configured symbol:

```text
AAPL.pth
AAPL.manifest.json
```

Use the shared promotion command; do not copy raw weights into this directory:

```powershell
python -m colab.deepscalper.training promote `
  --checkpoint .\runs\AAPL\best.pth `
  --manifest .\runs\AAPL\best.manifest.json `
  --weights-dir .\weights `
  --symbol AAPL
```

Promotion verifies the checksum, accepted validation/holdout gates, feature and
action schema, symbol, and production architecture. Runtime validates the pair
again before connecting to Alpaca.

Unmanifested checkpoints from `a51a46b` are retained under
`quarantine/legacy-a51a46b/` for provenance only. They are not approved models
and cannot start model-driven paper mode.
