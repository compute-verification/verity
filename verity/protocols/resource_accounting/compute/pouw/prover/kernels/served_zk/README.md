# Served Pearl-C under zero knowledge

This directory takes a served Pearl-C execution phase through the proof system: a retained served pass, committed as one
audit execution phase in the ZK statement's format, is registered, sampled once, proved hidden tile by tile under Flock
`--zk` and judged as PoUW's verifier says, with every control refused. It also measures what committing a served call's
rows as hm96-sha512 leaves costs on the GPU beside the scheme's own rows.

## Try it

Without a GPU, check the host side the row commitments' gates rest on:

~~~sh
uv run pytest tests/benchmarks/pouw/test_served_zk_rows_bench.py -q
~~~

It prints `3 passed`.

## Runner

**Hardware:** `bench`, `serve` and `zk_window` run on an RTX PRO 6000 node (sm_120), building on its CPUs and holding
one GPU through `gpu-lease`; `job` runs on a node's CPUs alone.

## Runner

`run.sh JOB [ARGS...]` starts one job's script with the arguments and environment as given; run it from the source tree,
as a `research run`'s command. Each script's header has its full `research run` line and its knobs.

~~~text
research run --on MACHINE --project verity --source . --cwd source --declared-output 'out/*' \
  --env GPU_LEASE_WHO=LANE -- bash verity/protocols/resource_accounting/compute/pouw/prover/kernels/served_zk/run.sh bench
~~~

| job | what it runs | writes under `$OUT` |
|---|---|---|
| `bench` | `rows_bench.py`, untimed: the GPU cost of committing a call's rows as hm96-sha512 leaves (`hm96_rows.cu`) beside the scheme's BLAKE3 rows, at Llama-3.1-8B's four linears | `rows-bench.json`, `rows-bench.log`, `identity.csv`, `cubins.sha256`, the `ptxas` reports |
| `serve` | the served kernel's ship built from this tree (`build.sh`), then `pearl_c_vllm/window.sh`'s smoke with its requests retained on the node, for `job` | `window/` (`window.sh`'s output and the ship), `tree.txt`, `build.log`, `lean-coins-build.log` |
| `zk_window` | `pearl_c_vllm/window.sh` with the served row's switches and `statement.py`'s rows (`MODE`: `smoke`, `window` or `build`) | `window/`, `tree.txt`, `build.log`, `nvcc-rows.txt`, `prebuilt.sha256` |
| `job` | the window through the proof system on CPUs, over a `serve` run's retained requests (`SERVED_RUN`): commit, register, the challenger's coins, the sample, each sampled tile proved under Flock `--zk` and opened, a filler tile, the outcome and every control | `summary.json` (each check and the verdict it must give), and a file or directory a step, listed in the script's header |

`job` is started through `hidden_zk/scope.sh` with `HZ_JOB=verity/protocols/resource_accounting/compute/pouw/prover/kernels/served_zk/job.sh`, which gives it a systemd
scope of its own on the node's CPUs; its header has the full command. `job` reads `hidden_zk/job.sh` for its builds.
