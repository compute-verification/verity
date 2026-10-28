# The vLLM demo: the remote protocol on a served H200

The vLLM demo is the runnable form of the **remote protocol** (`verity/remote/v0`, in `verity/core/protocols/remote/`):
what an auditor with no devices at the site establishes about a datacenter serving a model. One served run of
Llama-3.2-1B INT8 on one H200 gives its verdict, and the same run carries the onsite protocol's subprotocols beside it.
The remote protocol composes two subprotocols over one commitment and accepts only when both accept:

- **integrity**, by sampled proofs' one-stage audit (`verity/core/protocols/sampled_proofs/`): every value the serving
  computes is committed under hiding commitments, live coins sample proof blocks from a sealed sampling block, the
  prover proves each sampled block with C-Flock `--zk` behind the Lean firewall, and the Lean verifier gives the verdict;
- **work**, by proof of useful work (PoUW, `verity/core/protocols/pouw/`), on the served model itself: the endpoint
  serves PoUW's H1, each sampling block's coins come from a session of its own on the challenger's gate, and PoUW's
  piece checks them against that session, the gate's ledger and its pins.

PoUW's verdict on the served run credits nothing. The challenger's gate holds a development key, so every served PoUW
run is a development run. And an H1 commitment's served execution blocks aren't yet a `verity/pouw/v1` case, the form
remote's PoUW component reads. So the remote verdict can't accept on this run yet, and the run's record stays `pending`.

`example.toml` records what the demo establishes. Its run is `moved`: it has no passing run of record. Its proof level is
none. Sampled proofs' guarantee, `Flock.Guarantees.SampledKSLiveV3`, is proved, but no theorem yet carries it to the
audit-level claim `Claims.Sampling.KS`, which the audit's property `NetTiming.Guarantees.SampledAuditAccepted` takes
(`benchmarks/prove_path/README.md`). PoUW's proof layer needs `AcceptedLive`, which no served path meets, and remote's
composed soundness isn't in Lean (`verity/core/protocols/remote/README.md`).

## What runs

Llama-3.2-1B INT8 (`verity-int8/Llama-3.2-1B-INT8` at `15b0976d400d796ee589e7737d03a919526bf800`, the row's `model` and
`engine.revision`) with int8 linears (`--precision verity_int8`, the row's quantization), batch-invariant eager. vLLM
serves it on vy-dc-h200's one H200 through `verity_vllm`'s committing OpenAI endpoint. Its values are committed as it
serves (`--scheme rows-sha512`), and every closed sampling block is sealed. The chain is
`benchmarks/one_stage/served_verdicts.sh`'s, whose header gives each step:

1. the endpoint, under a GPU lease, with the committer's weights registration before the first request;
2. greedy requests from the certifier's traffic generator (`serve_load.py --varied`), closed into one sampling block;
3. sampled proofs on that sampling block: its advice and request Programs, the circuit of the served execution blocks
   at each class, and the records audit (C-Flock `--zk` on the sampled proof blocks behind the Lean firewall, and the
   Lean verifier's verdict);
4. PoUW's piece, and each onsite subprotocol's piece where it has one;
5. one record of every piece's verdict (`$OUT/record/verdicts.json`), `pending` while any subprotocol is missing.

Each step's verdict line is in `$OUT/verdicts.txt`. What attaches Verity to vLLM is `integrations/vllm/`; this
directory holds what is the demo's own: its placement and its commands.

## The onsite protocol on the same run

Onsite (`verity/onsite/v0`, in `verity/core/protocols/onsite/`) is remote's superset: it adds PoUS, the bandwidth audit
(PoUB, `verity/remote-bandwidth/v0`) and network certification. The memory wipe has its piece on the same run too, and
isn't yet a component of either composite. Each runs on the served run as its piece where `served_verdicts.sh` names
its command (`POUS_CMD`, `BANDWIDTH_CMD`, `CERTIFIER_CMD`, `WIPE_CMD`). A piece that doesn't run is named `MISSING` and
gets no verdict.

## The auditor's devices: not placed

The auditor's devices are the verifier, the network certifier and the memory challenger. None of them is placed: the Lean
verifier runs on the endpoint's node beside the prover, and the network certifier and the memory challenger, which onsite's
pieces read, may run there too. Being placed needs each of them on a machine the prover doesn't control. The live coins
are the challenger's, an auditor host reached through its gate: PoUW's always, and sampled proofs' when `PP_CHALLENGER`
names it. Otherwise the Lean verifier's own coins sample the proof blocks.

## Workload and placement

- The row: `placement.toml`'s `row`, one of the integration's rows (`integrations/vllm/workloads/`), with its requests,
  its model and revision, and its pinned engine. Its bytes are a fixture in the evidence store, which `run.sh` fetches and
  checks against that directory's `SHA256SUMS`. The endpoint commits under the Build's manifest in the row dir that
  `placement.toml`'s `COMMIT_RUN` names.
- The load, `serve_load.py`: `served_verdicts.sh`'s one sampling block of varied requests (`--varied`), or `serve.sh`'s
  `BURSTS` bursts of `PER_BURST` requests, each burst sealed on its own (8 and 4 by default), or with `VARIED=N` and
  `ADMIT=release-order` the network certifier's varied traffic.
- `placement.toml`: where each run goes and what its node holds. That is the node's venv, CUDA, HF cache, taps and row
  dirs, and the flags the run recorder starts each run with, among them the gate PoUW's coins come from. `run.sh` hands a
  run's script its node's paths as its environment, so no script names a node.
- `workloads/` keeps a bf16 RTX PRO 6000 row, which the demo doesn't serve.

## Commands

`run.sh` is the one command (`verity example vllm [window|commit|serve] [--dry-run] [FLAG...]`). It starts the run with
the run recorder from the tree root, on the node `placement.toml` names, and returns once the run starts. FLAGs go to the
recorder (`--env MAIN_SHA=SHA`, `--env PP_CHALLENGER=...`); `--dry-run` prints the command.

1. `verity example vllm`: `served_verdicts.sh`, the served run through every subprotocol, as above.
2. `verity example vllm commit`, once per row: `commit.sh`, the row's Build, Match and a Commit that seals its replay
   bundle, kept on the node and put in the evidence store.
3. `verity example vllm serve`: the endpoint alone, `serve.sh`, serving until `UNTIL` and sealing each closed sampling
   block in the serve dir, for an audit of one later (`benchmarks/prove_path/run.sh` with `PP_SERVE_RUN`, `PP_WINDOW` and
   `PP_SERVE_ROWS`).

The scripts reach the integration by `PYTHONPATH`, which holds the tree and `integrations/vllm`, and run
`verity_vllm.pipeline.cli` and the integration's tools from `integrations/vllm/`.
