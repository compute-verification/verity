# The vLLM example

Llama-3.2-1B (`unsloth/Llama-3.2-1B` at `9535bd9b1d1dea6acafbdc4813b728796aeb28da`), bf16, batch-invariant eager, served
by vLLM on one RTX PRO 6000 through `verity_vllm`'s committing OpenAI endpoint, and a served window audited through every
subprotocol. This directory holds what is the example's: its workload, its placement and its commands. What attaches
Verity to vLLM is `integrations/vllm/`, and the chain a served window runs is `examples/one_stage/served_verdicts.sh`'s.

## Workload

- The row `llama32-1b__bf16__rtxpro6000__tp1__b4__i256__o32__mixed__greedy__bi-eager`, the one file in `workloads/`: its
  requests, its model and revision, and its pinned engine. The integration finds it only through `VERITY_WORKLOADS`, which
  `commit.sh` sets (`verity_vllm.config.row_file`); its Build's manifest is what the endpoint commits under.
- The load, `serve_load.py`: `served_verdicts.sh`'s one window of varied requests (`--varied`), or `serve.sh`'s `BURSTS`
  bursts of `PER_BURST` requests, each burst one window (8 and 4 by default), or with `VARIED=N` and `ADMIT=release-order`
  network-accounting's varied traffic.

## Placement

`placement.toml` is where each run goes and what its node holds: the node's venv, CUDA, HF cache, taps and row dirs, and each
run's `research run` flags. `run.sh` hands a run's script its node's paths as its environment, so no script names a node.

| Party | Where |
| --- | --- |
| The row's Commit of record, the endpoint, the prover and the Lean verifier | `vy-nebius-1`, one leased RTX PRO 6000 (`gpu-lease 1`) while serving |
| The verifier's coins | its own, or the challenger `vy-challenger-1` through infra's gate (`--env PP_CHALLENGER=...`) |

## Commands

`run.sh` is the one command (`verity example vllm [RUN] [--dry-run] [FLAG...]`): it starts RUN with `research run` from the
tree root on the node `placement.toml` names, and returns once it starts (`research status RUN` checks on it). FLAGs go to
`research run`; `--dry-run` prints the command.

1. `verity example vllm commit`, once per row: `commit.sh`, the row's Build and Match of record and a Commit that seals its
   replay bundle, the row dir kept on the node and put in the evidence store. `placement.toml`'s `COMMIT_RUN` names the
   one the endpoint commits under, now run `r20261024-055630-5377`.
2. `verity example vllm`: one served window through every subprotocol, `served_verdicts.sh`: the endpoint committing under
   the row's Commit of record, its requests, every closed window sealed, the window's sampled proofs (C-Flock `--zk` behind
   the Lean firewall, and the Lean verifier's verdict), PoUW, the bandwidth audit, PoUS and the network certifier where each
   has its piece, and one record of every verdict, each in `$OUT/verdicts.txt`.
3. `verity example vllm serve`: the endpoint alone, `serve.sh`, serving until `UNTIL`, each closed window's advice
   committed and its row dir sealed in the serve dir, for an audit of one window later
   (`benchmarks/prove_path/run.sh` with `PP_SERVE_RUN`, `PP_WINDOW` and `PP_SERVE_ROWS`).

The scripts reach the integration by `PYTHONPATH`, which holds the tree and `integrations/vllm`, and run
`verity_vllm.pipeline.cli` and the integration's tools from `integrations/vllm/`.
