# The vLLM example: the remote protocol's sampled-proofs half

This example runs one half of the **remote protocol** (`verity/remote/v0`, in `verity/core/protocols/remote/`):
what a verifier with no hardware at the site establishes about a datacenter serving a model. The remote protocol composes
two subprotocols and accepts only when both accept:

- **integrity**, by sampled proofs' one-stage audit (`verity/core/protocols/sampled_proofs/`): every value the serving
  computes is committed under hiding commitments, the verifier's own random coins select proof blocks of a served verification
  phase, the prover proves the sampled proof blocks with C-Flock in zero knowledge behind the Lean ZK server, and the Lean verifier
  gives the verdict;
- **work**, by proof of useful work (PoUW, `verity/core/protocols/pouw/`).

This example runs the integrity half end to end. Its work half doesn't run here: PoUW's served form needs an int8 model
and the challenger's gate, and on such a workload `benchmarks/one_stage/served_verdicts.sh` runs it today only as a
development run, which credits nothing.

`example.toml` records what the example establishes. Sampled proofs' guarantee, `Flock.Guarantees.SampledKSLiveV3`, is
proved: when the verifier accepts a `--zk` session, the committed transcript is outside the sampling relation with
probability at most δ plus the proof system's errors. The example's proof level is still none, because the remote
protocol it is half of has no guarantee in Lean for it to instantiate: `Verity.Soundness` is not in Lean yet, and remote's
verifier reads one-stage v1, which takes the sessions' verdicts as evidence (`verity/core/protocols/remote/README.md`,
"The open gap"). Onsite's chain has a gap of its own: network certification's `NetTiming.Guarantees.SampledAuditAccepted`
takes `Claims.Sampling.KS` as a hypothesis, and the theorem that gives it at the deployed record,
`Claims.Sampling.sampling_assembled`, is not a lock guarantee and takes the verifier's own facts as hypotheses
(`benchmarks/prove_path/e2e_chains/README.md`).

## What runs

Llama-3.2-1B (`unsloth/Llama-3.2-1B` at `9535bd9b1d1dea6acafbdc4813b728796aeb28da`), bf16, batch-invariant eager, served
by vLLM on one RTX PRO 6000 through `verity_vllm`'s committing OpenAI endpoint: this is today's run of record, and
`run.sh` passes the workload's own model and precision (`--precision pin`). `served_verdicts.sh`'s own default, int8 on an
H200, is tonight's served run, Tue 27 Oct, 6 PM PDT. Each closed verification phase is sealed, its
values committed as the serving runs, and one verification phase is audited by sampled proofs. What attaches Verity to vLLM is
`integrations/vllm/`, and the chain an audit runs is `benchmarks/one_stage/served_verdicts.sh`'s. This directory holds
what is the example's own: its workload, its placement and its commands.

## Workload

- The one workload `workloads/SHA256SUMS` lists: its requests, its model and revision, and its pinned engine. Its bytes are
  a fixture in the evidence store (registered in `fixtures/artifacts.json`, gitignored here), fetched and checked against
  that digest by `run.sh` and by the integration (`verity_vllm.config.workload_file`). The integration finds it only
  through `VERITY_WORKLOADS`, which `commit.sh` sets (`verity_vllm.config.row_file`); its Build's manifest is what the
  endpoint commits under.
- The load, `serve_load.py`: `served_verdicts.sh`'s one verification phase of varied requests (`--varied`), or `serve.sh`'s
  `BURSTS` bursts of `PER_BURST` requests, each burst sealed on its own (8 and 4 by default), or with `VARIED=N` and
  `ADMIT=release-order` the network certifier's varied traffic.

## Placement

`placement.toml` is where each run goes and what its node holds: the node's venv, CUDA, HF cache, taps and workload dirs,
and the flags the run recorder starts each run with. `run.sh` hands a run's script its node's
paths as its environment, so no script names a node. The endpoint, the prover and the Lean verifier share one GPU node,
holding one leased RTX PRO 6000 while serving. The verifier's coins are its own, or the live challenger's on an auditor
host (`--env PP_CHALLENGER=...`). `example.toml` states the example's level on `examples/README.md`'s scale.

## Commands

`run.sh` is the one command (`verity example vllm [window|commit|serve] [--dry-run] [FLAG...]`): it starts the run with
the run recorder from the tree root on the node `placement.toml` names, and returns once it starts. FLAGs go to the
recorder; `--dry-run` prints the command.

1. `verity example vllm commit`, once per workload: `commit.sh`, the workload's Build and Match of record and a Commit that
   seals its replay bundle, kept on the node and put in the evidence store. `placement.toml`'s `COMMIT_RUN` names the
   Commit the endpoint commits under.
2. `verity example vllm`: `served_verdicts.sh`, the endpoint committing under that Commit, its requests, every closed
   verification phase sealed, then the sampled proofs (C-Flock `--zk` behind the Lean ZK server, and the Lean verifier's
   verdict), with one record of every verdict in `$OUT/verdicts.txt`. The other subprotocols (PoUW, PoUB, PoUS, memory
   wipes and network certification) run only when their commands are set (`POUW_CMD`, `BANDWIDTH_CMD`, `POUS_CMD`,
   `CERTIFIER_CMD`, `WIPE_CMD`); each one unset prints a `MISSING` line, and the record stays `pending`. The example's
   runs of record (`example.toml`'s `runs`) ran sampled proofs alone.
3. `verity example vllm serve`: the endpoint alone, `serve.sh`, serving until `UNTIL` and sealing each closed verification
   phase in the serve dir, for an audit of one later (`benchmarks/prove_path/run.sh` with `PP_SERVE_RUN`, `PP_WINDOW` and
   `PP_SERVE_ROWS`).

The scripts reach the integration by `PYTHONPATH`, which holds the tree and `integrations/vllm`, and run
`verity_vllm.pipeline.cli` and the integration's tools from `integrations/vllm/`.
