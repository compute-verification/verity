# verity-vllm

The vLLM integration (`verity_vllm`), Verity's frontend for vLLM. It pins one vLLM engine, compiles the workload that engine
serves to its circuit, commits the values of that circuit the proofs need while vLLM serves, and installs the subprotocols'
hooks on the engine. Verity itself knows nothing of vLLM.

The one entry point is `examples/vllm/run.sh` (`verity example vllm`), which serves a workload on a GPU node and audits it
through every subprotocol (`examples/vllm/README.md`).

## What it does

- **Pinned engine.** vLLM at one commit (`config.VLLM`), built by `engine.vllm_adapter.build_engine` from a checkpoint pinned
  in `manifests/checkpoints.json` and its engine profile (`engine/profiles/`). Every Commit reads back, on every rank, the
  switches that select kernels (`engine/served.py`) and fails closed unless each is vLLM's default or declared with a reason.
- **Workloads.** Each file in `workloads/` (`verity-vllm/workload/v2`, read as `config.RowConfig`, written by `verity-vllm
  workload`) states one workload with the runtime that serves it: checkpoint and revision, requests and sampling, the vLLM
  commit, engine arguments, kernel environment and target. The code still says "row" for a workload (`verity-vllm row`).
  A workload file's bytes are a fixture in the evidence store: the tree keeps its SHA-256 (`workloads/SHA256SUMS`) and its
  art id (`fixtures/artifacts.json`, the file gitignored). Every reader takes it through `config.workload_file` (outside
  code: `verity_vllm.pipeline.workload_file`, shell: `verity-vllm workload-file`), which fetches an absent one by its id and
  refuses one whose digest differs. A new or changed file is a new fixture (`research data put --kind fixture/v1 --file F
  --preserve`), registered, gitignored and listed in `SHA256SUMS` (`tests/pipeline/test_workload_file.py` refuses one that isn't).
- **Build: the circuit.** The torch frontend (`program/frontend/`) lowers the served model's exported graph by lowering rules
  into a Program of registered Definitions (`verity.ml.definitions.registry`), with its runtime correspondence: each Call's
  module, graph node, slot and view. `program/predict/` derives the same Programs ahead of time from the workload alone.
- **Query: proof blocks.** The query of record, `Q_word_v1{X=16,W=32}` (`query/word.py`, on `verity.ml.circuits`), cuts every
  Call's circuit into proof blocks of bounded committed width. The required-value manifest (`query/manifest/`) lists every
  value on a proof block's boundary, with the prescribed inputs (weights, prompt ids) and what a protocol requires.
- **Match.** A capture of the engine's run (`observe/`) is folded back into a Program (`observe/fold/`) and compared with the
  Build's (`check/match/`), so the circuit committed to is the one the engine ran.
- **Commit.** While vLLM serves, value acquisition (`acquire/`: hooks and native taps) takes each required value where it is
  computed, and the GPU committer (`commit/committer/`) commits it under the scheme `rows-sha512` (`commit/step_rows.py`;
  `vllm-v1`, `commit/scheme.py`, by name). The checks (`check/`) end in one verdict, `verdict.json` (`check/verdict.py`).
- **Serve.** `verity-vllm serve-openai` is an OpenAI-compatible endpoint on the in-process engine that commits every step and,
  with `--seal`, seals each closed sampling period for a later audit.
- **Records.** The circuit database (`pipeline/circuit_db.py`, `verity-vllm records`) keeps, per workload of record
  (`manifests/records.json`), the exact circuit it served and the evidence that it is faithful, never committed values.

## Layout

~~~text
verity_vllm/
  llm.py          LLM, the pinned engine
  config.py       workload files, checkpoint pins, protocol names; settings.py: every environment variable
  engine/         the vLLM engine: build, profiles, served switches, hooks (hooks.py), engine specs, a built engine's handle
  program/        Programs: the torch frontend and its lowering rules, lifting, ahead-of-time prediction (predict/)
  query/          the verification query, its proof blocks and the required-value manifest
  correspondence/ runtime correspondence: Call to module, graph node, slot and view
  observe/        value observation, and the fold of a capture back into a Program
  acquire/        value acquisition: the acquisition plan, hooks and native taps
  commit/         commitment: trees, leaf rules, the GPU committer
  check/          checks to verdicts: Match, replay, execution extent, weights of record, batch invariance
  properties/     property harnesses: non-interference, kernel census, golden corpus, tap exactness
  protocols/      the subprotocols' hooks on the engine
  pipeline/       stage drivers (Build, Match, Commit), the CLI, the endpoint, the circuit database; tp/ for TP > 1
  records/ collectives/ ops/ pins/   typed stage records; TP collective sites; pod bootstrap; per-device pins
tests/ tools/     tests mirror the package (tests/regression: VERITY_REGRESSION=1); scripts run by path
workloads/ manifests/   workload digests (SHA256SUMS); checkpoint pins, workloads of record, Definitions, capture maps
~~~

Imports point down `pipeline` > `check` | `properties` > `acquire` > `observe` | `commit` > `engine` > `query` |
`correspondence` > `program` > `collectives` > `config` (`tests/lint/test_p09_layering.py`).

## Python API

Outside code imports only `verity_vllm`, `verity_vllm.engine.spec`, `verity_vllm.engine.handle`,
`verity_vllm.engine.served_seed`, `verity_vllm.program.predict.window_weights`, `verity_vllm.program.predict.window_rows` and
`verity_vllm.pipeline.workload_file`, each only its `__all__`, and the modules still on its `KNOWN` list (`tests/test_vllm_public_api.py`). `verity_vllm.LLM` is a pinned vLLM engine, constructed and called like `vllm.LLM`:

~~~python
from vllm import SamplingParams
from verity_vllm import LLM

llm = LLM("unsloth/Llama-3.2-1B", revision="9535bd9b1d1dea6acafbdc4813b728796aeb28da")
outputs = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=16))
~~~

`model` and `revision` (the full commit sha) must be pinned in `manifests/checkpoints.json`, with the snapshot on the host.
`LLM.from_spec(path)` builds the same engine from an engine spec (`verity-vllm/engine-spec/v0`, `engine/spec.py`):
`model`, `revision` and an `[engine]` table of exactly the options below. Supported options:

| option | default | sets |
| --- | --- | --- |
| `tensor_parallel_size` | 1 | GPUs the model is sharded over |
| `max_num_seqs` | 8 | requests the scheduler runs together |
| `max_num_batched_tokens` | vLLM's | tokens per scheduler step |
| `max_model_len` | the pin's | context length |
| `gpu_memory_utilization` | the profile's | fraction of device memory the engine may take |
| `kv_cache_memory_bytes` | vLLM's sizing | the KV cache's size in bytes, in place of sizing it from free memory |
| `enable_chunked_prefill` | vLLM's, on | False holds a prefill that doesn't fit a step instead of cutting it into chunks |
| `enable_prefix_caching` | off | True lets a request read the KV pages another computed for the same prefix |
| `async_scheduling` | vLLM's, on | False schedules each step once the previous step's output is in |
| `enforce_eager` | True | eager execution; False compiles with torch.compile and CUDA graphs |
| `execution` | `bi-eager`, or `bi-compiled` with `enforce_eager` False | the execution label (`engine.served.EXECUTIONS`) |
| `dtype` | the profile's | served dtype override |
| `quantization` | the checkpoint's | vLLM quantization override, such as `"fp8"` |
| `compilation_config` | vLLM's | compilation and CUDA-graph settings |
| `attention_backend` | vLLM's choice | the attention backend, such as `"FLASH_ATTN"` |
| `attention_config` | vLLM's | attention settings, such as `{"flash_attn_version": 2}` |
| `moe_backend` | vLLM's `"auto"` | the MoE backend, such as `"triton"` |
| `seed` | 0 | the engine seed |
| `protocols` | none | protocol options installed on the engine: `"pouw:<scheme>,pous:<scheme>"` or `{name: scheme}` |
| `protocols_config` | none | each enabled protocol's settings, `{name: {...}}`; never hashed |

Any other option, or a value of the wrong type, raises `TypeError` naming it. `generate` returns vLLM's `RequestOutput` list
unchanged and appends a `GenerateRecord` to `llm.records`; `llm.pin` is the resolved `EnginePin`, `llm.engine_record` the
engine as built, `llm.vllm` the wrapped `vllm.LLM` and `llm.model` the served model, which protocol options install on.

## Protocol hooks

Each subprotocol that runs on the engine has one adapter in `verity_vllm/protocols/`: sampled proofs (`sampled_proofs.py`),
PoUW (`pouw/`) and PoUS (`pous.py`), enabled on their own or together (`config.parse_protocols`, off by default, refused at
TP > 1). An adapter states `NAME`, `schemes()`, `refusals(scheme, ctx)` and `install(scheme, target)` (the contract is
`verity_vllm/protocols/interface.py`); the protocol logic stays in its `verity.core.protocols` package. At a linear layer's `quant_method.apply` PoUS wraps PoUW,
which executes the GEMM (`protocols.SITES`), and sampled proofs commits outside every site. Every option first commits to the
plaintext weights, then installs innermost first. `verity_vllm/protocols/remote_bandwidth/` is the node's registered store
for PoUB, with the hooks that keep it current while vLLM serves. The network certifier is on the
node's uplink.

Every runtime patch, an adapter's or a benchmark arm's, goes through `engine/hooks.py`: `patch`, `wrap`, `append` and `service`
under a `Hooks(tag=...)`, each undone exactly by `uninstall`, with `hooks.live(tag)` listing what a tag installed. Nothing else
replaces an attribute of a vLLM, torch or Triton object. Under TP > 1 the model lives in the rank workers, reached through
vLLM's `worker_extension_cls` (`pipeline/tp/rank_worker.py`). `engine.arrivals.run_requests_arrivals` times a run per
request and per step, on an injectable clock.

## Commands

Every tool is a `verity-vllm` command (`python -m verity_vllm.pipeline.cli`; `verity-vllm --help` lists them all), among them:

| command | does |
| --- | --- |
| `row run WORKLOAD ROLE REPO REV` | a workload's Build, Match and Commit (`row chain` also publishes its record) |
| `serve-openai` | the committing OpenAI-compatible endpoint |
| `records show\|programs\|fetch\|publish` | the circuit database |
| `predict compare\|emit` | ahead-of-time Programs, compared with a traced Build, and a switched workload's Build |
| `neutrality --tier fast\|heavy` | re-traces the workloads of record and fails when a Program digest moved |
| `rejudge RUN` | a finished TP Commit's GPU-free gates, recomputed from what it saved |

## Running

`uv sync --all-packages --extra torch-cpu` installs it with CPU-only torch (the `torch` extra is the CUDA build; vLLM runs
only on a pod). Tests and tools read fixtures relative to this directory, so run them from here. The suite is `verity-vllm`
(`tools/verity/check/suites.py --list`); it runs remotely like every suite, and `python3 tools/suite_parts.py launch` splits it
into parallel `research run` jobs. `verity_vllm/ops/pod_bootstrap.sh` builds a pod's runtime (Python 3.12, torch cu129, the
pinned vLLM wheel, the native taps); on a pod without a GPU, or with `--cpu`, it skips the taps.

## The engine pin

A move of the vLLM commit rewrites, in one change: the wheel URL, its sha256 and its readiness check in
`verity_vllm/ops/pod_bootstrap.sh`; `config.VLLM`, which every workload file's `engine.vllm` states and `TargetProfile` checks
against the running build; benchmarks' `VLLM_PIN` (`verity_bench.vllm.presets`) and each benchmark spec's `vllm_pin`; and
PoUW's serving venv (`benchmarks/pouw/pearl_c_vllm/setup.sh`). `git grep` of the old short sha lists the rest.

## Vocabulary

Serialized field names (`member`, `op_path`, `output_member`, `query: Q_module_v1`, `weights_pin`, `vu_*`) keep their old
spelling, since a header string inside a digested document changes its digest; prose uses these words:

- **Call**: one application of a Definition inside a Program; **Value**: an input or output of a Call.
- **Required values**: the values on proof-block boundaries, the prescribed inputs and what a protocol requires, listed by
  the **required-value manifest** (`manifest.json`); the **acquisition plan** says where to take each.
- **Runtime occurrence**: one execution of a Call, keyed by `op_path`, `output_member`, invocation, step, request and rank.
  **Value acquisition** takes a required value's bytes there; **value observation** does the same for the Match.
- **Family** and **kind** name circuit Definitions and query families, many of them serialized fields.
