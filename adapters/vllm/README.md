# verity-vllm

The vLLM integration (`verity_vllm`) serves vLLM under Verity. It pins one vLLM engine (`config.VLLM`) and serves it
through a committing OpenAI-compatible endpoint. While vLLM serves, the endpoint commits every value the proofs need where
vLLM computes it. It groups the engine's steps into windows and seals each closed window for a later audit. Before
serving, `predict` derives the workload's Programs from the workload alone, the Build of record the endpoint commits
under. After serving, the registered compiler compiles a window's advice into the request Programs its committed rows
must pass. The subprotocols install on the engine as adapters, and PoUW's useful-work arms run at every linear layer.
Verity itself knows nothing of vLLM.

## Entry points

- **`verity-vllm serve-openai`** (`verity_vllm/pipeline/serve_openai.py`) is the committing endpoint, on the in-process
  engine. It commits every engine step under `rows-sha512`, with the committer the Build of record's manifest builds
  (`--required-manifest`, which `verity-vllm predict row` writes). A window opens at the first step after the engine was
  idle and closes when it is idle again or after `--window-steps`. Each window is one Commit, and `--seal` seals it.
  `--protocols pouw:<scheme>` serves PoUW's H1 arm on the endpoint's own Commit, one H1 window per window, and
  `--remote-verifier` proves the engine's store to PoUB's verifier as it serves. Its module docstring documents every
  flag and record. The public release leaves out this endpoint and the window circuit, which read the network
  certifier's encoding, and that encoding stays private, so it ships without a runnable served window.
- **`verity demo vllm`** (`demos/vllm/run.sh`) starts the served-verdicts launcher,
  `benchmarks/one_stage/served_verdicts.sh`, on the node `demos/vllm/placement.toml` names. It runs the Build of
  record, then the endpoint with `--seal` under a GPU lease, and a load. Then come sampled proofs on one sealed window:
  its advice through the registered compiler (`tools/advice_compile.py`), Flock `--zk` behind the Lean ZK server, and the
  Lean verifier's verdict. PoUW, PoUS, the bandwidth audit and the network certifier follow where each has its piece,
  and it ends with one record of every verdict. The launcher's header gives each step, and `demos/vllm/README.md`
  says what the run establishes.
- **`verity-vllm COMMAND`** (`verity_vllm/pipeline/cli.py`, or `python -m verity_vllm.pipeline.cli`): every tool here is a
  subcommand of this one command line. `verity-vllm --help` lists them all. `COMMANDS` maps each to its module, whose
  docstring documents it. The commands to start from:

| command | does |
| --- | --- |
| `serve-openai` | the committing endpoint |
| `row run\|stage\|chain` | a workload's Build, Match and Commit (`chain` also publishes its record) |
| `build`, `match`, `commit` | one stage of one workload |
| `tp-capture`, `tp-match`, `tp-commit` | the same stages at a tensor-parallel size above 1 |
| `predict row\|compare\|emit` | the Build of record from the workload alone; held to a traced Build; a switched workload's Build |
| `advice` | a served run's advice, from its step trace and Commit record |
| `commit-replay`, `serving-rows` | a sealed Commit's replay on CPU; its serving rows and registration |
| `records show\|programs\|fetch\|publish` | the circuit database |
| `neutrality --tier fast\|heavy` | rebuilds the workloads of record's Programs and fails when a digest moved |
| `block-counts` | a day's inference blocks, judged against its declared c, k and n |
| `workload-file` | each workload made present and checked |

**`verity_vllm.LLM`** (`verity_vllm/llm.py`) is the pinned engine as a library, constructed and called like `vllm.LLM`:

~~~python
from vllm import SamplingParams
from verity_vllm import LLM

llm = LLM("unsloth/Llama-3.2-1B", revision="9535bd9b1d1dea6acafbdc4813b728796aeb28da")
outputs = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=16))
~~~

`model` and `revision` (its full commit sha) must be pinned in `manifests/checkpoints.json`. The class docstring
documents each option it takes: `tensor_parallel_size`, `max_num_seqs`, `max_num_batched_tokens`,
`long_prefill_token_threshold`, `max_model_len`, `gpu_memory_utilization`, `kv_cache_memory_bytes`,
`enable_chunked_prefill`, `enable_prefix_caching`, `async_scheduling`, `enforce_eager`, `execution`, `engine_class`,
`dtype`, `quantization`, `compilation_config`, `attention_backend`, `attention_config`, `moe_backend`, `seed`,
`protocols` and `protocols_config`. Any other option raises `TypeError`. `LLM.from_spec` builds the same engine from an
engine spec (`verity_vllm/engine/spec.py`). Outside code imports only the public modules (`PUBLIC` in
`tests/test_vllm_public_api.py`), and from each only its `__all__`.

## Replay

Under the two-stage design (`GLOSSARY.md`), serving commits only what each replay block reads and writes at its edges.
After the first stage samples replay blocks, the developer replays each sampled one and registers its interior, and the
second stage samples proof blocks inside it. Here a replay block is one request, from its first engine step to its
finish inside one served window. The parts, in the order they run:

- **The boundary commitment at serve** (`verity_vllm/pipeline/boundary_rows.py`). For each request that finishes in a
  window, the engine core keeps two hiding rows: its ingress (the prompt's length, the sampling inputs and seed, the
  prompt's token ids) and its egress (how many ids were served, why it finished, the ids). The requests of one size
  class, prompt and output lengths rounded up to powers of two, are one registration. A window closes after its
  length, when no request is running, or at stop, and a thread of its own commits it. A request still running is
  carried to the next window, so a replay block never crosses one. A scheduler that caches prefixes commits the
  same rows; it is refused only with history tables, which hold a request's latent rows from position 0, where a
  request served from the cache computes none before its hit's end.
- **The schedule** (`verity_vllm/engine/step_trace.py`). The in-process engine core records what the scheduler decided
  at every step: each arrival, each step's rows in the model runner's order with their first positions and lengths,
  preemptions, cache hits and finishes. Recording changes nothing the engine does. The schedule reaches an audit as
  advice: `verity_vllm/pipeline/advice_windows.py` commits an advice's encoding as hiding rows, with only their root
  and the length in bits public and the registered compiler's digest bound beside them.
- **The replay** (`benchmarks/serve_tax/replay.py`). A served model's kernels need not be batch invariant: a request's
  bits can depend on the requests that shared its steps. So the replay re-runs the whole window, not the sampled
  request alone. It starts the served engine again in one process from the trace's header, adds every request under
  its served engine id, prompt, sampling inputs and seed right before the step it arrived before, and lets the
  scheduler decide the rest as it did. It then compares its own step trace and every sampled token with the served
  ones, and `replay.json` says where they differ.
- **What keeps it bit for bit.** Both sides run the engine profile of `verity_vllm/engine/env.py`, which gives each
  switch's reason: Model Runner V2, the engine core in-process, Inductor's reduction tiling fixed. Both run with
  `PYTHONHASHSEED=0` and request ids not randomized, because the scheduler iterates sets of request ids. A model whose
  sparse attention selects positions by a top-k over more keys than it keeps is served and replayed with the
  canonical top-k plugin (`benchmarks/dsa_topk/`), since the stock kernels' selection can vary from run to run.
- **The interior registration** (`verity_vllm/pipeline/interior_rows.py`). The replay takes each sampled request's
  logits rows and sampled ids at every rank's sampler (`verity_vllm/engine/sampler_capture.py`, `replay.py
  --capture`). `pieces` cuts them into the proof blocks Flock proves, and `interior` commits those after the boundary
  registration's digest. Until a size class's request Definition lands, the replay block is a stand-in: instances of
  one check template, in the two-stage serve one token-select check for each served id. So the interior holds the
  token selection and not the model's layers.
- **The stages around it** (`benchmarks/one_stage/two_stage.py`). `first` registers each size class's boundary
  commitment, names the window's advice registration (`--advice`) and samples the class's replay blocks once on the
  challenger's coins. `resume` gives the replay side the same stream for the interior registration and the second
  stage. The verifier refuses every two-stage run at its last check (`FORM` in
  `verity/protocols/spot/verifier/second_stage.py`) until the soundness proof covers that form.

The commands are `benchmarks/serve_tax/run.sh`'s, and its header documents each variable. `ARMS=two-stage` serves with
the boundary commitment, which it writes under `PRIVATE/two-stage/boundary`. `REPLAY=LOAD REPLAY_FROM=ARM_DIR
REPLAY_CAPTURE=ID,...` replays that load's window alone and captures the named requests. A replay block that is a span
of positions inside a request, and a served prefix-cache hit, are not on this path.

What a replay costs is measured, and no figure is kept in the tree. `replay.json` holds the served window's seconds
beside the replay's and the engine's start. `benchmarks/serve_tax/tp_restore.sh` serves, restores a fresh engine from a
snapshot taken at a window's edge and replays to the next edge; its `restore.json` holds the seconds of the build, the
restore and the replay. For an engine in one process, `benchmarks/serve_tax/restore.py` and `rebuild.py` also write
the replay's busy GPU seconds beside serving's for the same steps; a tensor-parallel replay has no such counter. The
planner prices a sampled replay block's replay from the model's `replay` card (`benchmarks/spot_figures/cards/`) in
`benchmarks/spot_figures/total_cost.py`.

## Where to look

One line for each subpackage of `verity_vllm/`, and for what sits beside it:

~~~text
verity_vllm/
  llm.py             LLM, the pinned engine as a library
  config.py          workload files, checkpoint pins, protocol names; settings.py: every environment variable it reads
  engine/            the pinned engine: its build, profiles, served switches (served.py); hooks.py owns every runtime patch
  pipeline/          stage drivers (Build, Match, Commit; tp/ for TP > 1), the endpoint (serve_*.py), cli.py, circuit_db.py
  program/           Programs: the torch frontend (frontend/) lowers the served model onto registered Definitions
    predict/         the Build of record from the workload alone, the registered compiler (steps.py), the window circuit
  query/             the query of record (required.py), which cuts each Call into proof blocks; the required-value manifest
  correspondence/    where each Call lives at runtime: its module, graph node, slot and view
  observe/           the observer's capture of an engine run, folded back into a Program (fold/) for the Match
  acquire/           value acquisition: the acquisition plan, and the hooks and native taps it installs
  commit/            the Commit: the GPU committer (committer/) and the rows-sha512 scheme (step_rows.py)
  commitments/       the vllm-v1 scheme's hashlib reference, PROTOCOL.md and vectors (vllm_v1/); its leaf tree (leaves.py)
  check/             the checks: the Match (match/), replay, coverage, weights of record; a Commit's verdict (verdict.py)
  protocols/         the subprotocols' adapters: sampled proofs, PoUS, PoUB's registered store (remote_bandwidth/)
    pouw/            PoUW: the executor at each linear layer, and its H1, Pearl-C and lead arms
  properties/        property harnesses: non-interference, kernel census, golden and holdout corpora, tap exactness
  records/           typed stage records, one for each kind one stage writes and another reads
  ops/, pins/        pod scripts (pod_bootstrap.sh builds a GPU node's runtime at the pin, vllm_env.sh the environment
                     of another build of engine.stock.BUILDS beside it); each workload's served pin per device
workloads/           workload files, each a fixture in the evidence store, read through config.workload_file
manifests/           checkpoint pins (checkpoints.json), workloads of record (records.json), Definitions, capture maps
tests/               the suite verity-vllm, a directory for each subpackage; lint/ holds the ratchets (lint/README.md)
tools/               scripts run by path: advice_compile.py, ratchet_gate.py, suite_parts.py
~~~

Imports point down a fixed layer order, with `pipeline` at the top and `config` at the bottom:
`tests/lint/test_p09_layering.py` states the order and holds the package to it. `tests/regression/` holds the reference
regression, skipped unless `VERITY_REGRESSION=1`.

## Running

`uv sync --all-packages --extra torch-cpu` installs it with CPU-only torch. The `torch` extra is the CUDA build, and
vLLM serves only on a GPU node, whose runtime `verity_vllm/ops/pod_bootstrap.sh` builds. Tests and tools read fixtures
relative to this directory, so run them from here. The suite runs remotely like every suite
(`tools/verity/check/suites.py`), and `python3 tools/suite_parts.py launch` splits it into parallel `research run` jobs.

## Vocabulary

Serialized field names (`member`, `op_path`, `output_member`, `query: Q_module_v1`, `weights_pin`, `vu_*`) keep their
older spelling, since a header string inside a digested document changes its digest. Prose uses these words:

- **Call**: one application of a Definition inside a Program. **Value**: an input or output of a Call.
- **Required values**: the values on proof-block boundaries, the prescribed inputs and what a protocol requires. The
  **required-value manifest** (`manifest.json`) lists them, and the **acquisition plan** says where to take each.
- **Runtime occurrence**: one execution of a Call, keyed by `op_path`, `output_member`, invocation, step, request and
  rank. **Value acquisition** takes a required value's bytes there; **value observation** does the same for the Match.
- **Workload**: what the code calls a row (`verity-vllm row`, `config.RowConfig`, a row id): its requests and sampling,
  with the checkpoint, engine arguments and target that serve them.
- **Family** and **kind** name circuit Definitions and query families, many of them serialized fields.
