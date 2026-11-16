# exfiltration

`verity bench exfiltration` prints GLM-5.3's exfiltration capacity in GB over a monitoring period, at 10% of total
compute: the most bits about the weights that can leave over the world's year, as two parts side by side, with every
input labelled measured, modeled, simulated, assumed, missing or run, not proved. It computes neither part. It
composes the two calculators that do, so it needs no GPU and no node, only the evidence store:

- the compute audit's part, the break-even planner's bound (`benchmarks/spot_figures/breakeven/`): what the wrong
  blocks of a sampled audit can carry at δ;
- the network attester's part, `verity bench certifier_term`'s (`benchmarks/certifier_term/`).

Its code is `benchmarks/harness/verity_bench/exfiltration.py`, whose docstring has how each part is read, the records
it reads from the store, the errors and what it prints; this directory holds its runner.

```text
verity bench exfiltration                 the figure, the inputs, each part, and the two side by side a period
verity bench exfiltration --json          the same as one document: value, unit, inputs, parts, periods
verity bench exfiltration --share 0.05    read at another share of total compute
verity bench exfiltration --served-run F  beside it, the share a served two-stage run's record F measures
verity bench exfiltration --variant V     read one variant of the row, in a run that holds several
verity bench exfiltration --response-ids N the network attester's part for a class of at most N ids a response
verity bench exfiltration --scan-delta D  the network attester's errors at another δ of its scans (a fraction)
verity bench exfiltration --local-only    read this machine's catalog alone, not the store's remote
verity bench exfiltration --help          every parameter and its default
```

Under a research run (`RESEARCH_RUN_DIR` set) it also writes `run.py`'s `METRICS`, the rows of `metrics.toml`, into
the run's `result.json` through the harness kit, and keeps the document as `out/exfiltration.json`:

```text
research run --on auto --project verity --source . --cwd source --store-read -- uv run python benchmarks/exfiltration/run.py
```
