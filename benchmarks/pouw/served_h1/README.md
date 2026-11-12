# served_h1

H1 receipts beside vLLM's int8-exact GEMM on one GPU: what the receipts cost a served Llama-3.1-8B int8 window, and
whether the GPU's receipts are the reference's bit for bit. The window here (`serve.py`) is a padded H1 + fx1 window in
the static form, with in-process coins or a development session's; it is not useful-work credit, and the gated served
window of record is vLLM's H1 path (`adapters/vllm/verity_vllm/protocols/pouw/h1/__init__.py`). `serve.py`'s docstring
states the window's tables, units, coins and credit rule.

**Hardware:** an RTX PRO 6000 node (sm_120). `arch.sh`, which the GPU jobs source, builds the kernels for the node's
card: `120a`, or `90a` on an H100 or H200. Every GPU job holds its GPU through `gpu-lease`.

## Runner

`run.sh JOB [ARGS...]` starts one job's script with the arguments and environment as given; run it from the source tree,
as a `research run`'s command. Each script's header has its full `research run` line and its knobs (environment
variables).

~~~text
research run --on MACHINE --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
  -- bash benchmarks/pouw/served_h1/run.sh serve
~~~

| job | what it runs | writes under `$OUT` |
|---|---|---|
| `serve` | `serve.py`'s window on one leased GPU for at most 28 minutes, after `noise.cu` is built for the card and the Lean verifier's coins are built; extra arguments go to `serve.py` | `serve.py`'s record (`manifest.json` and the window's files), `tree.txt`, `nvcc.txt`, `lean-coins-build.log`, `prover-put.txt` (the prover's data put in the store, or `prover.tar` and `prover.sha256` beside it when the put fails) |
| `producer_gpu` | the receipt producer's GPU tests (`benchmarks/pouw/tests/test_served_h1_producer.py`'s units-kernel tests) for each `h1u.cu` build in `BUILDS`, then `step_time.py` at each of `STEP_ROWS` | `builds.txt`, `build-*.log`, `junit-*.xml`, `pytest-*.log`, `step-*.json` |
| `j10r_gpu` | by `JOB`: `tests`, J10-R's GPU tests and `j10r_time.py` (J10-R's noise and receipts against v1's, interleaved), or `window`, a served window over J10-R's tile unit through `pouw_h1_gpu.py` | `builds.txt`, `build-*.log`, `gpu.txt`, `junit.xml`, `pytest.log`, `j10r_time.json`, `j10r_time.log`, `window.log` |
| `commit_legs` | served Commit windows one after another through `pouw_h1_gpu.py`, each over v1's or J10-R's tile unit, with in-process coins or, in its live form, coins from a gate session; the script's header names the forms and their knobs | a directory a window (`pouw_h1_gpu.json`, `stdout.log`, `stderr.log`, and a live window's session archives), the list of windows run, the gate's ledger, pins and key (`gate-*`), `builds.txt`, `build-hbc.log` |
| `preflight` | a CPU check of the node: the imports `serve.py` makes, the int8 pin's path, the `research` command and one receipt against its reference | its standard output only |

The verifier's check of a window's live coins, from the challenger's own run, is `coins.py check` (its command is in
`serve.sh`'s header).

## The Python modules

`serve.py`, `j10r_time.py` and `step_time.py` are the jobs' programs. `coins.py` (the live coins and their check),
`commit.py` (the window's tables), `credit.py` (the padded-window credit), `producer.py` (`h1u.cu`'s units kernel),
`partials.py` and `lane_refs.py` (option A's lane partials) are what they import. Other directories load some of them by
path: `benchmarks/one_stage/a5_coins.py` reads `coins.py`, `adapters/vllm/tests/protocols/pouw_h1_gpu.py` reads
`producer.py`, and `benchmarks/hash/dither/dither_a.py` reads `partials.py` and `lane_refs.py`. Their tests are the
`test_served_h1*.py` files, `test_partials.py` and `test_lane_refs.py` under `benchmarks/pouw/tests/`.
