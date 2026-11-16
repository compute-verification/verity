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
  blocks within their gate and fresh-input budgets. The served run checks it on the served node: once step 3 has
  counted the window's circuit at the policy's class (`IO_CLASS`), `inference_only.py verdict` judges the window's
  blocks against the policy written before the serve, and the Lean `inference-only-check --count` counts one block's
  gates, which must agree. That answer is the served node's own. The compose off the node carries it into the record
  as the node wrote it and doesn't check it again; only a composite (`COMPOSITE` set) checks the policy's case, in the
  product's Lean on the compose host, and the placed window composes none.

PoUW's verdict on the served run credits nothing. The challenger's gate holds a development key, so every served PoUW
run is a development run. And an H1 commitment's served execution blocks aren't yet a `verity/pouw/period/v1` case, the form
remote's PoUW subprotocol reads. So the remote verdict can't accept on this run yet, and the run's record stays `pending`.

Its run is `moved` (`demos/README.md`, `remote.toml`'s row): it has no passing run of record. Its proof level
is none. SPOT's guarantee, `Flock.Guarantees.SampledKSLiveV3`, is proved: when the verifier accepts a `--zk`
session, the committed trace is outside the sampling relation with probability at most δ plus the proof system's
errors. But the remote protocol's composed theorem, `Verity.remote_soundness_of_subprotocols`, takes its three subprotocols'
targets as open premises, and no Lean statement carries the acceptance of remote's one-stage form, one-stage v2, which runs the proof check
itself, to `SampledKSLiveV3`'s event (`verity/protocols/compose/verifier/remote/README.md`, "What's open"). PoUW's proof layer needs `AcceptedLive`, which no served path
meets. Onsite's chain has a gap of its own: network attestation's `NetTiming.Guarantees.SampledAuditAccepted` takes
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
2. greedy requests from network attestation's traffic generator (`serve_load.py --varied`), closed into one execution phase;
3. SPOT on that execution phase: its advice and request Programs, the circuit of the served execution blocks
   at each class, and the records audit (Flock `--zk` on the sampled proof blocks behind the Lean ZK server, and the
   Lean verifier's verdict);
4. PoUW's piece, and each onsite subprotocol's piece where it has one;
5. one record of every piece's verdict (`$OUT/record/verdicts.json`), `pending` while any subprotocol is missing.

The placed window (`[run.window.plain]`) is remote with no accounting piece: it serves no H1
(`H1_GATE` empty), so PoUW's piece is `MISSING`, and it runs no V3c (`V3C_CLASS` empty, so step 3's audit is the SPOT
audit of record) and no composite (`COMPOSITE` empty). Its record is composed off the node by `run.sh compose`. The
tied window (`run.sh window --tie`, `[run.window.tie]`) serves the BF16 row (`tied_row`) under Pearl+
(`PEARLPLUS_TIE`, `POUW=pearlplus`), its linears committed at V3c's class `4,32,4` (4 slots, prompts 32:32), sets no
`COMPOSITE`, and defers V3c's audit off the node (`V3C_AUDIT_MEM_GB` above node A's memory) to `run.sh copy` and
`run.sh audit`. Its sampling design is a setting of the launch's environment (`[run.window.tie]`), by default the benchmark's
`stratified-rate:1/2097152`.

`served_verdicts.sh`'s exit is its record's verdict: 0 when the compiled Lean composite of record accepted, which only
a composite computed off the served node, on an auditor device, is; 2 pending; 1 a rejection, a refused composite or a
failed step. A window on the served node can only end pending, so `window.sh` exits 0 when it does, its last line
`pending: composed off this node` (or, under a deferral, audited and composed off it), and with `PREFLIGHT_ONLY=1`
exits 0 when the preflight passed.

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
the SPOT audit of record among them (`placement.toml`'s `[run.window.coins.gate]`, held to the challenger's key record,
the tree's copy at `tests/benchmarks/pouw/fixtures/challenger-key.json`, and to main's first-parent line, which
`run.sh` reads from origin's `main` at launch). Only `run.sh window --development` takes development coins instead: a
local challenger session on the endpoint's node samples the proof blocks, which shows the path but is no auditor's
coins, and the run is a development run (`DEVELOPMENT=1`), which its record says. The verifier flips no coins of its
own. The other three are not placed: the Lean verifier runs on the endpoint's node beside the prover and again on the
compose node, a CPU pod of the research pool, and the network attester and the memory attester, which onsite's pieces
read, may run on the endpoint's node too. Being placed needs each of them on a machine the developer doesn't control.

## Workload and placement

- The workload: the integration's one (`adapters/vllm/workloads/`) that `placement.toml`'s `row` names, with its
  requests, its model and revision, and its pinned engine. Its bytes are a fixture in the evidence store, which `run.sh`
  fetches and checks against that directory's `SHA256SUMS`. The endpoint commits under the workload's Build of record,
  which `verity-vllm predict row` writes from the workload alone before serving. The row's traced Build, made by
  `run.sh setup` and named in `placement.toml`'s `builds`, is its cross-check (`predict compare --predicted`), and the
  endpoint serves only when they agree.
- The load, `serve_load.py`: `served_verdicts.sh`'s one execution phase of varied requests (`--varied`, or with
  `LOAD=fixed` one full execution phase of fixed length).
- `placement.toml`: where each run goes and what its node holds. The window serves on node A (`vy-spot-glm53`, eight
  H200s shared by card) and its record is composed on `vy-nebius-cpu-14`. A node's table holds its venv, CUDA, HF
  cache, taps, traced Builds and Flock's build, which `run.sh setup` leaves under the served run's directory, and a
  run's table the flags the run recorder starts it with. `run.sh` hands a run's script these as its environment, so no
  script names a node. No file or command line holds a machine's address: `window.sh` reads the gate's
  (`VY_CHALLENGER_1_HOST`) from `CHALLENGER_ENV`, a mode-600 env file on the node. The window's card, cores and end
  are `CARD`, `CORES` and `END`, which `run.sh` fills from the environment the run is launched in.
- `workloads/` keeps a bf16 RTX PRO 6000 workload, which the served run doesn't serve.

## Commands

`run.sh` is the one command (`bash adapters/vllm/tools/served_run/run.sh [window|compose|setup|copy|audit] [--dry-run] [--development] [--tie] [FLAG...]`).
It starts the run with the run recorder from the tree root, on the node `placement.toml` names, and returns once the
run starts. FLAGs go to the recorder (`--timeout 200m` to end with the row); `--dry-run` prints the command.

1. `CARD=N CORES=A-B END=UTC run.sh setup [--tie]`, once per row and node: `setup.sh` makes what the window reads,
   each step skipped where its result is there unless `STEPS` names it: the INT8 pin, the taps, Flock's build, and the
   row's predicted Build, held to its pin (INT8: `test_predict.py`'s; BF16: `served_verdicts.sh`'s `BUILD_OF_RECORD`),
   its traced Build and their compare. Its last line names the run id to add to `builds`. A tap's bytes change when it
   is rebuilt even where its sources don't, so a rebuilt tap is not live until its exactness test passes on the card
   (`fa_tap_exactness_gpu.py`, the guarded build against the default build's record): only then does setup install
   both taps and name the commit to set `TAPS_AT` to.
2. `CARD=N CORES=A-B END=UTC run.sh [--tie]` (`window`, the default): `window.sh`, which runs `served_verdicts.sh` on
   that card under its own lease and packs the compose hand-off into the run's `out/handoff/`; 75 with nothing run when
   the card is in use or `END` leaves under an hour (or under `LEASE_MIN`, a short window's length in minutes, which
   also bounds its lease). The gate gives coins only to a commit on `main`'s first-parent line, so from a branch not yet
   landed `run.sh` refuses: land it first, or take development coins with `--development`. It refuses a commit that
   changed the taps' sources since `TAPS_AT`.
3. `HANDOFF=DIR run.sh compose`, DIR holding the window run's `out/handoff/`: `compose.sh`, the window's record composed
   off the served node, its SPOT Lean verifier run there on the audit. `run.sh` refuses it from any commit but the
   window's (the hand-off's `COMPOSE_TREE`).
4. A tied window's deferred audit, between the window and the compose: `SERVED_RUN=<window run id> AUDIT_NODE=<node>
   run.sh copy` sends its serve dir's deferral (its files, held to their `inputs.sha256`) from node A to the same path on
   the audit's node, then `SERVED_RUN=… AUDIT_NODE=… run.sh audit` runs `served_verdicts.sh v3c` there at the window's
   commit (`V3C_PEER` the audit's node; `V3C_AUDIT_TIMEOUT` and `V3C_WARM_JOBS` from the launch) and packs the compose
   hand-off into its `out/handoff/`.
5. `run.sh settings MODEL [-- CMD...]`, for a workload that `verity run` starts on node A: node A's settings for the
   row that serves MODEL (its venv, taps, traced Build, weights, peer and disk floor, and the tied form's for the BF16
   base) as `NAME=value` lines, which name no address; with `-- CMD`, CMD run under them wherever its environment sets
   none, after the node's env files (`env_files`, each mode 600) are read and each `${NAME}` they or the environment
   set is filled in, the INT8 pin linked and the venv's CUDA headers found. `bindings/` holds `verity run --bindings`
   for each demo's configuration, whose addresses are such `${NAME}`s.

The scripts reach the integration by `PYTHONPATH`, which holds the tree and `adapters/vllm`, and run
`verity_vllm.pipeline.cli` and the integration's tools from `adapters/vllm/`.
