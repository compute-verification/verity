# The vLLM example

Llama-3.2-1B (`unsloth/Llama-3.2-1B` at `9535bd9b1d1dea6acafbdc4813b728796aeb28da`), bf16, batch-invariant eager, served
by vLLM on one RTX PRO 6000 through `verity_vllm`'s committing OpenAI endpoint, and each closed Commit window audited on
the prove path. This directory holds what is the example's: its workload, its placement and its commands. What attaches
Verity to vLLM is `integrations/vllm/`, and the audit is `benchmarks/prove_path/`'s.

## Workload

- The row `llama32-1b__bf16__rtxpro6000__tp1__b4__i256__o32__mixed__greedy__bi-eager`, a row of record of the integration
  (`integrations/vllm/workloads/llama32-1b__bf16__rtxpro6000__tp1__b4__i256__o32__mixed__greedy__bi-eager.json`): its
  Build's manifest is what the endpoint commits under.
- The load `serve.sh` sends: `BURSTS` bursts of `PER_BURST` requests, each burst one window (8 and 4 by default), or with
  `VARIED=N` and `ADMIT=release-order` network-accounting's varied traffic, both from
  `integrations/vllm/tools/serve_load.py`.

## Placement

| Party | Where |
| --- | --- |
| The row's Commit of record, and the endpoint | `vy-nebius-1`, one leased RTX PRO 6000 (`gpu-lease 1`), on vLLM's venv there |
| The prover and the Lean verifier | a CPU node, `vy-nebius-cpu-2` |
| The verifier's coins | its own, or the challenger `vy-challenger-1` through infra's gate (`PP_CHALLENGER`) |

## Commands

From the tree root, each a `research run` on its party's node (the scripts' headers list every variable). There is no
`run.sh` yet, so `verity example vllm` runs none of them: they run on three nodes, and the audit takes the serve run's `art:`
ids, so a wrapper would be more than their commands. `example.toml` puts the example at level 1 (runs) on the ladder
(`examples/README.md`): level 2 needs the Lean verifier on an auditor host, off the nodes the prover runs.

1. `commit.sh`, once per row: the row's Build and Match of record and a Commit that seals its replay bundle, the row dir
   kept on the node and put in the evidence store. The current one is run `r20261024-055630-5377`, `serve.sh`'s default
   `ROW`.

   ~~~sh
   research run --on vy-nebius-1 --project verity --source TREE --cwd source --mem-gb 192 --timeout 6h \
     --env GPU_LEASE_WHO=vllm-llama1b-commit -- bash examples/vllm/commit.sh
   ~~~

2. `serve.sh`: the endpoint (`verity-vllm serve-openai --scheme rows-sha512 --seal`) under the Commit of record's manifest
   (`ROW`), serving until `UNTIL`, each closed window's advice committed and its row dir sealed in the serve dir.

   ~~~sh
   research run --on vy-nebius-1 --project verity --source TREE --cwd source --timeout 200m --env PORT=18084 --env UNTIL=13:30 -- \
     env CHECK_SLOTS=/workspace/research/locks/gpu/slots python3 tools/verity/check/slot.py --by vllm -- \
     gpu-lease 1 --wait --mem-gb 256 -- bash examples/vllm/serve.sh
   ~~~

3. The audit of window K, honest and forged: `benchmarks/prove_path/run.sh` with `PP_SERVE_RUN` (the serve dir, or its
   `art:` id), `PP_WINDOW=K` and `PP_SERVE_ROWS` (the window's sealed row dir).

   ~~~sh
   research run --on vy-nebius-cpu-2 --project verity --source . --cwd clone --mem-gb 64 --mem-wait 3600 \
     --env PP_SERVE_RUN=art:… --env PP_WINDOW=0 --env PP_SERVE_ROWS=art:… -- bash benchmarks/prove_path/run.sh
   ~~~

The scripts reach the integration as they always have: `PYTHONPATH` holds the tree and `integrations/vllm`, and they run
`verity_vllm.pipeline.cli` and the integration's tools from `integrations/vllm/`.
