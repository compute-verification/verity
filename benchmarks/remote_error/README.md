# remote_error

The chance the remote application's cryptography is fooled: the cryptographic part of the soundness error of the
composite `verity/remote/v1` (SPOT, PoUW's Pearl+ run verdict and the inference-only policy), for a prover making at most
T SHA-512 calls. A security figure, computed from parameters, not measured.

```text
uv run python benchmarks/remote_error/run.py            # the tables
uv run python benchmarks/remote_error/run.py --json     # the same as JSON
research run --on auto --cwd source -- uv run python benchmarks/remote_error/run.py   # records metrics.toml's numbers
```

## Which error this is

A protocol's verifier gives a verdict; its soundness error is the chance it accepts a play whose claim is false. The
remote composite's error is the union bound over its subprotocols' errors (`Verity.Vocab.error`), and each splits in two:

- **Cryptography's part, this figure:** SHA-512 collisions under `ecr/sha-512` and `cr/sha-512`, LMS (n = 64) forgery
  under `euf-cma/lms-sha512-n64`, the coin trees' hiding term, and the sessions' knowledge error (the `--zk` table's
  bound, its rewinding finders, the link finders and extraction). These grow with T, so the figure pools them over the
  prover's compute.
- **Sampling's part, not this figure:** ε at δ, the chance a sample misses more than an ε share of wrong proof blocks
  (`Verity.Sampled.Live.sampleMiss`). It doesn't grow with T, and `benchmarks/sampling_plan/` evaluates it.

## Contract

- **Guarantee.** `Verity.Sampled.remote_soundness_deployedReady`, the remote composite's soundness at SPOT's live session
  (`Verity.Sampled.Live.delta`), and `Verity.remote_soundness_of_subprotocols`, the union bound it instantiates. The
  script reads their open premises and named assumptions from the locks and prints them.
- **Headline.** Lean doesn't instantiate the remote guarantee at numeric parameters: its one instance, the honest
  witness, has every error ⊤. The figure says so: the bound's formula is proved, and its value is evaluated in Python.
- **Terms.** Each term of SPOT's δ that cryptography bounds, and SPOT's forgery term, says its basis: `proved` (a closed
  term of the guarantee's δ), `assumed` (the guarantee's formula at a parameter Lean leaves free, evaluated at the stated
  one) or `open` (PoUW's and the policy's errors, which Lean leaves to the base vocabulary at the remote composite; the
  total leaves them out). Each names the Lean declarations it reads.
- **Parameters.** `PARAMS` in `run.py`, each with its value, its basis (`lean`, `code`, `assumed`) and its source. The
  commit strings a window's sample reads (`--positions-log2`) scale the binding term, which the extraction budget and the
  prover's calls also enter.
- **Arithmetic.** Exact rationals: each irrational (√, e) is bounded on the side that keeps the figure an upper bound.
- **Monitoring periods and fleets.** One play a machine a window. Over one window and each monitoring period
  (`verity.protocol.MONITORING_PERIODS`), over one datacenter (`benchmarks/census/deployments.json`'s reference, its
  machines and window length) and over the world's compute (`benchmarks/census/adversary.json`'s reference, as machines
  of the datacenter's size), at an assumed SHA-512 rate a GPU. A term a + b·t + c·t² a play pools to at most
  plays·a + b·T + c·T². Each total is given with its share a machine-window audited. The census gives no AND gates a
  machine-window, so no share is given in AND gates.
- **Metrics.** `metrics.toml`: the total and the share, each for each fleet's week, month and year, the world's month
  last.

Its tests are `tests/benchmarks/remote_error/`.
