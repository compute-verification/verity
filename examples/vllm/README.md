# The vLLM example

Llama-3.2-1B (`unsloth/Llama-3.2-1B` at `9535bd9b1d1dea6acafbdc4813b728796aeb28da`), bf16, batch-invariant eager, served
by vLLM on one RTX PRO 6000 through `verity_vllm`'s committing OpenAI endpoint, and a sampling period of that serving
audited through every subprotocol. This directory holds what is the example's: its workload, its placement and its
commands. What attaches Verity to vLLM is `integrations/vllm/`, and the chain an audit runs is
`benchmarks/one_stage/served_verdicts.sh`'s.

## Workload

- The one file in `workloads/`: its requests, its model and revision, and its pinned engine. The integration finds it only
  through `VERITY_WORKLOADS`, which `commit.sh` sets (`verity_vllm.config.row_file`); its Build's manifest is what the
  endpoint commits under.
- The load, `serve_load.py`: `served_verdicts.sh`'s one sampling period of varied requests (`--varied`), or `serve.sh`'s
  `BURSTS` bursts of `PER_BURST` requests, each burst sealed on its own (8 and 4 by default), or with `VARIED=N` and
  `ADMIT=release-order` network accounting's varied traffic.

## Placement

`placement.toml` is where each run goes and what its node holds: the node's venv, CUDA, HF cache, taps and workload dirs,
and each run's `research run` flags. `run.sh` hands a run's script its node's paths as its environment, so no script names
a node. The endpoint, the prover and the Lean verifier share one GPU node, holding one leased RTX PRO 6000 while serving;
the verifier's coins are its own, or the live challenger's on an auditor host (`--env PP_CHALLENGER=...`). `example.toml`
states the example's doneness (`examples/README.md`).

## Commands

`run.sh` is the one command (`verity example vllm [window|commit|serve] [--dry-run] [FLAG...]`): it starts the run with
`research run` from the tree root on the node `placement.toml` names, and returns once it starts (`research status RUN`
checks on it). FLAGs go to `research run`; `--dry-run` prints the command.

1. `verity example vllm commit`, once per workload: `commit.sh`, the workload's Build and Match of record and a Commit that
   seals its replay bundle, kept on the node and put in the evidence store. `placement.toml`'s `COMMIT_RUN` names the
   Commit the endpoint commits under.
2. `verity example vllm`: `served_verdicts.sh`, the endpoint committing under that Commit, its requests, every closed
   sampling period sealed, then the sampled proofs (C-Flock `--zk` behind the Lean firewall, and the Lean verifier's
   verdict), PoUW, the bandwidth audit, PoUS and the network certifier where each has its piece, with one record of every
   verdict in `$OUT/verdicts.txt`.
3. `verity example vllm serve`: the endpoint alone, `serve.sh`, serving until `UNTIL` and sealing each closed sampling
   period in the serve dir, for an audit of one later (`benchmarks/prove_path/run.sh` with `PP_SERVE_RUN`, `PP_WINDOW` and
   `PP_SERVE_ROWS`).

The scripts reach the integration by `PYTHONPATH`, which holds the tree and `integrations/vllm`, and run
`verity_vllm.pipeline.cli` and the integration's tools from `integrations/vllm/`.
