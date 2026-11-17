# The vLLM served run: the remote protocol on a served H200

The served run is the runnable form of the **remote protocol**'s H1 development executable (`verity/remote/v0`, H1's
period as its work, which no configuration names, in `verity/protocols/compose/verifier/remote/`; the composite of record, `verity/remote/v1`, has Pearl+'s run instead):
what an auditor with no devices at the site establishes about a datacenter serving a model. One served run of
Llama-3.2-1B INT8 on one H200 gives its verdict, and the same run carries the onsite protocol's subprotocols beside it.
The remote protocol composes three subprotocols, tied to one record, and accepts only when all three accept:

- **integrity**, by SPOT's one-stage audit (`verity/protocols/proofs/spot/`): every value the serving
  computes is committed under hiding commitments, live coins sample proof blocks from a sealed execution phase, the
  prover proves each sampled block with Flock `--zk` behind the Lean ZK server, and the Lean verifier gives the verdict;
- **work**, by proof of useful work (PoUW, `verity/protocols/resource_accounting/compute/pouw/verifier/`), on the served model itself: the endpoint
  serves PoUW's H1, each execution phase's coins come from a session of its own on the challenger's gate, and PoUW's
  piece checks them against that session, the gate's ledger and its pins;
- **the inference-only policy** (`verity/protocols/workload_classification/training_detection/inference_only/verifier/`): the registered circuit divides into inference
  blocks within their gate and fresh-input budgets. The served run doesn't run it: the served run doesn't produce the
  declared division yet.

PoUW's verdict on the served run credits nothing. The challenger's gate holds a development key, so every served PoUW
run is a development run. And an H1 commitment's served execution blocks aren't yet a `verity/pouw/period/v1` case, the form
remote's PoUW subprotocol reads. So the remote verdict can't accept on this run yet, and the run's record stays `pending`.

Its run is `moved` (`demos/README.md`, `remote.toml`'s row): it has no passing run of record. Its proof level
is none. SPOT's guarantee, `Flock.Guarantees.SampledKSLiveV3`, is proved: when the verifier accepts a `--zk`
session, the committed trace is outside the sampling relation with probability at most δ plus the proof system's
errors. But the remote protocol's composed theorem, `Verity.remote_soundness_of_subprotocols`, takes its three subprotocols'
targets as open premises, and no Lean statement carries the acceptance of remote's one-stage form, one-stage v2, which runs the proof check
itself, to `SampledKSLiveV3`'s event (`verity/protocols/compose/verifier/remote/README.md`, "What's open"). PoUW's proof layer needs `AcceptedLive`, which no served path
meets. Onsite's chain has a gap of its own: network certification's `NetTiming.Guarantees.SampledAuditAccepted` takes
`Claims.SamplingGeneric.Compiled.KS` as a hypothesis, and the theorem that gives it at the deployed record,
`SamplingDeploy.ks_compiled`, is not a lock guarantee and takes the verifier's own facts as hypotheses
(`benchmarks/prove_path/e2e_chains/README.md`).

## What runs

Llama-3.2-1B INT8 (`verity-int8/Llama-3.2-1B-INT8` at `15b0976d400d796ee589e7737d03a919526bf800`, the workload's `model`
and `engine.revision`) with int8 linears (`--precision verity_int8`, the workload's quantization), batch-invariant eager.
vLLM serves it on the endpoint's H200 node through `verity_vllm`'s committing OpenAI endpoint. Its values are committed as
it serves (`--scheme rows-sha512`), and every closed execution phase is sealed. The chain is
`benchmarks/one_stage/served_verdicts.sh`'s, whose header gives each step:

1. the endpoint, under a GPU lease, with the committer's weights registration before the first request;
2. greedy requests from network certification's traffic generator (`serve_load.py --varied`), closed into one execution phase;
3. SPOT on that execution phase: its advice and request Programs, the circuit of the served execution blocks
   at each class, and the records audit (Flock `--zk` on the sampled proof blocks behind the Lean ZK server, and the
   Lean verifier's verdict);
4. PoUW's piece, and each onsite subprotocol's piece where it has one;
5. one record of every piece's verdict (`$OUT/record/verdicts.json`), `pending` while any subprotocol is missing.

The run's exit is its record's verdict. 0 means the compiled Lean composite of record accepted, and only a composite
computed off the served node, on an auditor device, is of record. 2 means pending: no composite of record yet. That is
what a window run on the served node ends with, since its own composite is a development check; it is not a crash. 1
means a rejection, a refused composite, or a failed step.

Each step's verdict line is in `$OUT/verdicts.txt`. What attaches Verity to vLLM is `adapters/vllm/`; this
directory holds the served run's own: its placement and its commands.

## The onsite protocol on the same run

Onsite (`verity/onsite/v0`, in `verity/protocols/compose/verifier/onsite/`) is remote's superset: it adds PoUS, PoUB
(`verity/remote-bandwidth/v0`) and network attestation. The memory wipe has its piece on the same run too, and
isn't yet a subprotocol of either composite. Each runs on the served run as its piece where `served_verdicts.sh` names
its command (`POUS_CMD`, `BANDWIDTH_CMD`, `CERTIFIER_CMD`, `WIPE_CMD`). A piece that doesn't run is named `MISSING` and
gets no verdict.

## The auditor's parties

The auditor's parties are the verifier, the challenger, the network attester and the memory attester. The
challenger is an auditor device reached through its gate, and its live coins are PoUW's and, by default, SPOT's,
V3c's audit, the SPOT audit of record, among them (`placement.toml`'s `[run.window.coins.gate]`; the launch sends the
deployment's challenger key record, `challenger-key.json` at the tree's root). Only `run.sh window --development`
takes development coins instead: a local challenger session on the endpoint's node samples the proof blocks, which
shows the path but is no auditor's coins, and the run is a development run (`DEVELOPMENT=1`), which its record says.
The verifier flips no coins of its own. The other three are not
placed: the Lean verifier runs on the endpoint's node beside the prover, and the network attester and the memory
attester, which onsite's pieces read, may run there too. Being placed needs each of them on a machine the developer
doesn't control.

## Workload and placement

- The workload: the integration's one (`adapters/vllm/workloads/`) that `placement.toml`'s `row` names, with its
  requests, its model and revision, and its pinned engine. Its bytes are a fixture in the evidence store, which `run.sh`
  fetches and checks against that directory's `SHA256SUMS`. The endpoint commits under the workload's Build of record,
  which `verity-vllm predict row` writes from the workload alone before serving. The traced Build in the workload dir
  that `placement.toml`'s `COMMIT_RUN` names is its cross-check (`predict compare --predicted`), and the endpoint serves
  only when they agree.
- The load, `serve_load.py`: `served_verdicts.sh`'s one execution phase of varied requests (`--varied`, or with
  `LOAD=fixed` one full execution phase of fixed length), or `serve.sh`'s `BURSTS` bursts of `PER_BURST` requests, each
  burst sealed on its own (8 and 4 by default), or with `VARIED=N` and `ADMIT=release-order` the network attester's
  varied traffic.
- `placement.toml`: where each run goes and what its node holds. That is the node's venv, CUDA, HF cache, taps and
  workload dirs, and the flags the run recorder starts each run with, among them the gate PoUW's coins come from.
  `run.sh` hands a run's script its node's paths as its environment, so no script names a node. It holds no machine's
  address: the gate's value names the variable that holds the challenger host's (`VY_CHALLENGER_1_HOST`), which
  `run.sh` fills from the environment the run is launched in, and without it the window run starts nothing and says so.
- `workloads/` keeps a bf16 RTX PRO 6000 workload, which the served run doesn't serve.

## Commands

`run.sh` is the one command (`bash adapters/vllm/tools/served_run/run.sh [window|commit|serve] [--dry-run] [--development] [FLAG...]`). It starts the run with
the run recorder from the tree root, on the node `placement.toml` names, and returns once the run starts. FLAGs go to the
recorder (`--env MAIN_SHA=SHA`); `--dry-run` prints the command, and `--development` takes development coins.

1. `run.sh commit`, once per workload: `commit.sh`, the workload's Build and Match of record and a Commit
   that seals its replay bundle. The workload dir stays on the node, the only copy: the bundle and the Commit's records
   under `commit/` hold values (the committed store with its salts, the weights key, the requests and served tokens)
   and `match/spec.json` holds the prompts, so nothing stores the dir. The run's record holds only the files
   `published.sh` lists, each with the reason it holds nothing of a request (the node and the tree's commit, each
   stage's flags, exit code and verdict word, the sealed bundle's manifest digest, the GPU lease's times, the row's
   target), and the run names every file it left on the node; to publish another file, give it an entry there.
   `placement.toml`'s `COMMIT_RUN` names the Commit whose traced Build the endpoint's Build of record is held to.
2. `run.sh` (`window`, the default): `served_verdicts.sh`, the served run through every subprotocol, as above.
3. `run.sh serve`: the endpoint alone, `serve.sh`, serving until `UNTIL` and sealing each closed execution
   phase in the serve dir, for an audit of one later (`benchmarks/prove_path/run.sh` with `PP_SERVE_RUN`, `PP_WINDOW` and
   `PP_SERVE_ROWS`).

The scripts reach the integration by `PYTHONPATH`, which holds the tree and `adapters/vllm`, and run
`verity_vllm.pipeline.cli` and the integration's tools from `adapters/vllm/`.
