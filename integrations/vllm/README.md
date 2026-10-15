# verity-vllm

The Verity x vLLM integration: the Torch/vLLM-derived Programs (`verity_vllm.program.registry`, `verity_vllm.program.frontend`), the
runtime capture and Match/Commit machinery (`verity_vllm`), and the qualification sweep (`verity_vllm/ops`,
`workloads/`).  Promoted from `veritor@0903f60d6` (release `r18-int-3`) as one working closure; see the Notion page
"Veritor → Verity migration" for the decisions and the evidence index.

## Layout

One distribution (`verity-vllm`), one package (`verity_vllm`), one subpackage per job; `tests/` mirrors the package.  Naming
(Draft 3.1): core Verity nouns get the shortest names (`program`, `query`, `commit`), integration concepts carry their
qualifier (`runtime` correspondence, `weights_of_record`), implementation artefacts have descriptive compound names
(`native_collect`, `poc_*` for v1 leftovers).  Build / Match / Commit are the stage drivers in `pipeline/`, not layers.
Imports point down the layer order `pipeline` > `check` | `properties` > `acquire` | `protocol_options` > `observe` | `commit` >
`engine` > `query` | `correspondence` > `program` > `collectives` > `config` (lint P9).

~~~text
verity_vllm/
  pipeline/           stage drivers      build.py match.py commit.py row*.py report.py hot.py layout.py cli.py llm.py research_tools.py tp/
  engine/             the vLLM engine    build.py capture.py pinned.py run_facts.py code_identity.py vllm_adapter.py engine_driver.py engine_profile.py arrivals.py rank_worker.py profiles/
  program/            Program            registry/ frontend/ lifting/ verity/ml/kernels/ model.py workload.py compact.py dtypes.py
  query/              verification query module_body.py word.py cross_call.py required.py program_view.py manifest/ compare.py
  correspondence/     runtime corresp.   runtime.py reader_for_*.py resolve*.py capture_identities.py
  observe/            value observation  observer.py events.py log.py storage.py contract.py runtime_tree.py chunks.py fold/
  acquire/            value acquisition  plan.py install.py stage.py gate.py compiled.py sources/ *_tap_src/
  protocol_options/   protocol options   __init__.py (selector, composition) interface.py sampled_proofs.py pous.py pouw.py
  commit/             commitment         merkle.py hashing.py identity.py binding.py padding_steps.py hidden_*.py committer/
  check/              checks -> verdicts match/ replay/ executed_prefix.py verdict.py gates.py weights_of_record.py root_policy.py ...
  properties/         property harnesses noninterference.py census.py kernel_allowlist.py golden.py holdout.py admission.py ...
  collectives/        collectives        collective_record.py collective_sites.py embedding_shard.py
  config.py target_family.py             run-definition locations; the target-family preflight
  ops/                pod operations     pod_bootstrap.sh pod_gate.sh pod_*_tap.sh canary.sh *_negatives.sh known_roots.json
tests/<subpackage>/   mirrors the package; tests/regression = the R19 reference harness (opt in with VERITY_REGRESSION=1)
workloads/ manifests/ fixtures/ data/ docs/    records of the qualification table; never rewritten by the tool
~~~

**`pipeline/`** holds the stage drivers (Build / Match / Commit): `build.py`, `match.py`, `commit.py`, `hot.py`, `report.py`,
`global_program.py` (the rooted workload Program of a Build), `m1_capture.py`, `layout.py` (Build / run directory layout),
`cli.py` (the one CLI, `verity-vllm`), `row.py` / `row_*.py` (`verity-vllm row`: a sweep row's
stages), `llm.py` (the public `verity_vllm.LLM`), `timeline.py` / `spans.py` / `telemetry/`, `admission_bound.py` / `admission_planner.py`,
`research_outputs.py` / `research_result.py`, `source_identity.py`, `release_json.py`, `workload.py` /
`coverage_workloads.py`, `experiment.py`, `research_tools.py` (the `vllm.build` / `vllm.match` / `vllm.commit` Tools of
`tools/research`), and `tp/` (the rank-aware capture, commit, match and fold-match drivers).  Entrypoints:
`verity-vllm <command>` (`python -m verity_vllm.pipeline.cli <command>`) and `verity_vllm.LLM`.  Never: Verity semantics -- a check that
lives here is misfiled.

**`engine/`** is the vLLM engine we attach to: `build.py` (the engine under κ and what it reads back), `capture.py` (the
step observer and run header; `pinned.py` its host buffers), `run_facts.py` (profile manifest, versions), `code_identity.py` (executed code objects),
`vllm_adapter.py` (workloads and request driving; the stages import the engine's names from it), `engine_driver.py`,
`engine_profile.py`, `arrivals.py`, `rank_worker.py` (the TP rank worker extension), and `profiles/` (kappa: capture profiles and workload JSON; one generic
constructor + a facts table).  Never: commit, decide what is required.

**`program/`** implements *Program*: Definitions (`registry/`, the registered library; importing it registers the on-demand
primitive families through `verity.ml.circuits.codec.register_lazy_family`), lowering (`frontend/`: the torch frontend, `derive`, `rules/`
= lowering rules, the vLLM meta export, `target_profile`, the correspondence emitter `emit.py`, the collective export ops),
lifted / padded Programs (`lifting/`), `workload.py` / `compact.py`, `model.py` (specified computations such as bf16(sqrt(H))),
and `verity/ml/kernels/`, the independent kernel models the Definitions wrap (numpy / C++ / CUDA models, twins, relation checkers and
their tables).  Entrypoints: `program.frontend.derive`, `program.registry`.  Never: module paths as identity, hook sets,
proof-unit policy, model-name tables (`registry/quarantine/` is the by-name residue and only shrinks; the lanes core
binds were promoted into `registry/`).
`registry/fa3_check_inf.py` is FA3's
per-iteration `Check_inf` chain (`Attention_v4`), bound under `TargetProfile.fa3_construction = "check-inf-per-iteration"` (the default).

**`query/`** implements the *verification query*: `word.py` (`Q_word_v1{X=16,W=32}`, the query of record: every Call's circuit cut
into proof units of bounded committed width, on core's `verity.ml.circuits.cut`; `verity/partition/v1` names the partition; its unit rules are
shared across processes and runs through `manifest build --unit-rule-cache DIR`, env `VERITY_UNIT_RULE_CACHE`), `module_body.py`
(Q_module_body_v1, for A/B), `cross_call.py` (the opt-in cross-Call recompute check, `manifest build --cross-call-check`),
`program_view.py`, `manifest/` (the
required-value manifest: `format.py` the shared vocabulary, `verify.py` the rebuild-and-compare check of record),
`query_artifact.py` (query artifact identities), `compare.py` (a v2 manifest against a v1 one), `required.py` (v1
`IDENTITY_KEY` rows).  Boundary values, proof-unit partitions and dead gates are core's `verity.ml.circuits.boundary`, `verity.ml.circuits.partition`
and `verity.ml.circuits.liveness`.  Never: execution data, hooks, model names, anything read from a run.

**`correspondence/`** implements *runtime correspondence* as data: `runtime.py` (runtime-correspondence/v1),
`reader_for_query.py` / `reader_for_acquire.py`; and *occurrence resolution*: `resolve.py`, `resolve_decomp.py`.  Never:
verification policy, commitment positions, requiredness, verdicts.

**`observe/`** implements *value observation* (the Match-side capture): `observer.py` (the TorchDispatchMode), `events.py` /
`log.py` / `storage.py`, `contract.py` (the one observation contract), `triton_adapter.py`, `runtime_tree.py`, `chunks.py`
(chunk attribution), and `fold/` (raw log -> Program: `fold.py`, `tree.py`, `memory.py`, `views.py`, `resolver.py`,
`patterns/` (one module per kernel family, the TP collectives included), `patterns_prefix.py`).  Never: commit, decide what is
required, name a model.

**`acquire/`** implements *value acquisition*: `plan.py` (required values + correspondence -> acquisition plan), `install.py`,
`stage.py`, `compiled.py` (the manifest of `Execution=compiled` rows), `sources/` (`hidden_source.py` / `moe_source.py` /
`compiled_source.py` / `compiled_kernel_source.py` / `partial_source.py` / `norm_scale_source.py` / `router_softmax_source.py` /
`vocab_range_source.py`, attached by `taps.py`), `fa2_plane_classes.py`, `committer_api.py`, and the shipped tap sources (`fa2_tap_src/`,
`fa3_tap_src/`, `norm_tap_src/`, `router_tap_src/`).  `gate.py` is the CPU gate (one row's acquisition plan, off-pod).  Four taps are
on by default; `NORM_TAP=0`, `GUARDED_MAX_TAP=0`, `ROUTER_TAP=0` or `VOCAB_TAP=0` turns one off, for A/B only.  The guarded-max tap
(`query/guarded_max.py`) runs on every row, whose attention is FA2 or FA3: the tap builds of `pod_fa2_tap.sh FA2_TAP_ROW_GUARD=1` /
`pod_fa3_tap.sh FA3_TAP_ROW_GUARD=1` write `GuardNegInfZero(m)` into word 3 of the stream's ROW entry, and each key block's
`max * scale` into the MS class appended after FIN (`VERITY_MAT_SRC=123`), which lengthens every stream identity of the manifest.  The
others attach only where the row's manifest names their family (`pipeline/row_records.py` `applicable_taps`): the norm-scale tap
(`query/norm_scales.py`), the router-softmax tap on MoE rows (`query/router_softmax.py`: the ordered router's committed softmax per
token, from a tapped `topk_softmax`) and the vocabulary-range source on TP rows (`query/vocab_range.py`: a TP rank's `tok - START` and
range bits per token, from the fused embedding's ids, no kernel change).  `ops/pod_bootstrap.sh` builds their libraries.
Never: decide what is required; never a model-name list.

**`commit/`** implements *commitment*: `merkle.py` (trees over canonical ids, `path_shape` / `fold_path`, openings), `hashing.py`
(`H(tag, *parts)`), `identity.py`, `binding.py` (binding map + coverage), `padding_steps.py`, `hidden_stream.py`, `scheme.py` (the commitment scheme `vllm-v1`: every leaf and root rule, over `verity.ml.commitments.vllm_v1`), and `committer/`, the GPU committer (`native_host.py`, the collector of record;
`native_collect.py` + `native_*.cu` / `.cpp`, `native_jit.py`, `leafhash.py`, `flush_points.py`, `hidden_gpu_src/`).  Never:
runtime metadata in the identity tuple.

**`check/`** implements *checks -> verdicts*: `match/` (`global_match.py`, `global_match_fast.py`, `program_compare.py`,
`batch_decomp.py`, `rank_match.py`, `xrank_collectives.py`), `replay/` (`driver.py` with `sampled_replay`, `replay*.py`,
`stoch_recompute.py`, `compiled_kernel_check.py`, `vu_query.py`), `gates.py` G1..G8, execution extent (`executed_prefix.py`),
`verdict.py` (with `commit_verdict`), `value_check.py` / `compiled_value_check.py` / `oracle_compare.py`, `kernel_identity.py`,
`fold_compare.py`, `operand_provenance.py`, and input provenance (`weights_of_record.py`: checkpoint basis, `weights_pin`;
`root_policy.py`); `poc_*.py` are v1 leftovers.  Never: acquire or commit anything.

**`properties/`** holds the property harnesses: `noninterference.py`, `census.py` + `kernel_allowlist.py`, `golden.py` +
`golden/`, `holdout.py`, `admission.py` (differential admission tests), `quarantine_lint.py`, `protected.py`,
`fa_tap_exactness.py`, `norm_tap_exactness.py`, `router_tap_exactness.py`, `vocab_range_exactness.py`.  Never: a per-run check.

**`collectives/`** holds the collective record (`collective_record.py`), the communicator hook sites (`collective_sites.py`)
and the embedding shard layout (`embedding_shard.py`).

**`ops/`** is pod operations: `pod_bootstrap.sh` (which also installs the `verity-vllm` launcher), `pod_fa2_tap.sh` / `pod_fa3_tap.sh` / `pod_norm_tap.sh` / `pod_router_tap.sh` / `pod_hidden_gpu.sh`, `canary.sh`, `compiled_commit.sh`, the
`*_negative*.sh` runs, `cov_pod.sh`, the gates (`pod_gate.sh`, `verify_lane.sh`), `known_roots.json`.  Never: Python that a layer
above would import.

The stdlib-only core of `verity_ir` (`types refs defs program layout evaluate query codec parts query_codec`, `gateset`
as `query_ast`) is `verity.ml.circuits` in `verity`, and its `plan` is `verity.core.protocols.sampled_proofs.plan` in `verity/core/protocols/sampled_proofs`; the on-demand primitive families
veritor hardcoded in `codec.py` are registered by `verity_vllm/program/registry/__init__.py` through `verity.ml.circuits.codec.register_lazy_family`.

## Vocabulary (frozen; Notion "vLLM integration v2 — Draft 3" §12)

Prose, docstrings and Notion use these words.  Serialized field names (`member`, `op_path`, `output_member`, `query: Q_module_v1`,
`weights_pin`, ...) are not renamed: a header string inside a digested document changes its digest.  Subcircuit template,
subcircuit, backend family and backend mean what the root README's Glossary says.

Core (Verity semantics, `verity`):

- **Definition** — a registered function with a total evaluator on encodings (`verity_vllm.program.registry`).
- **Call** — one application of a Definition inside a Program (retires "Program row", "instance" in prose; `instances.json.gz` stores Calls).
- **Value** — an input or output of a Call (retires "Member" in prose; the serialized `member` / `output_member` field keeps its name).
- **Program** — the sequence of Calls with its dataflow; `program_digest` covers the dataflow only.
- **verification query** — Q: which sets of Calls are **proof units** (`VU` / `vu_*` in serialized and code names); `Q_module_v1` is called **Q_module_body_v1** in prose
  (units = module bodies, submodule calls excluded; the serialized id stays until a labelled decision).
- **boundary values** — `⋃_{V∈Q(P)} (in(V) ∪ out(V))`; **required values** = boundary values ∪ prescribed inputs (weights, prompt ids)
  ∪ protocol-required values (sampled ids); **commitment** — the position-identified Merkle roots over the required values;
  **verification obligation** — what a verifier must be able to establish from the commitment.

Integration (this tree):

- **lowering rule** (was Rule) — turns an exported-graph node into a Call (`verity_vllm/program/frontend/rules`).
- **runtime correspondence** — Call ↔ (module path, fx node, slot, view) in the served model; a **runtime occurrence** is one execution of a
  Call, its **occurrence key** (`op_path`, `output_member`, `invocation`, step, request, rank) the serialization.
- **value acquisition** — taking a required Value's bytes at its occurrence (hooks, taps); **value observation** — the same for the
  Match record; **occurrence resolution** (was Attribution) — mapping an observed occurrence back to its Call / request.
- **input provenance** (was input binding / Roots) — where each prescribed input comes from (`weights_of_record`, `named_parameters`).
- **execution evidence** (was runtime evidence) — what the run wrote (`timeline.jsonl`, capture log, tokens.json).
- **check / verdict** — one named check over the record, PASS / FAIL with evidence; **required-value manifest** (never bare "manifest")
  — the cached `RequiredValues(P, Q)` population (`manifest.json`); **acquisition plan** — which occurrences to acquire, derived from it.

Checks (prose name → serialized name): runtime match (`global_match`, GM-01 G1..G8), local replay (`C2` / `sampled_replay`), boundary
linkage (`linkage`), checkpoint binding (`weights_pin`), execution extent (`executed_prefix`), required-value coverage (`coverage`),
Program/source identity (`identity`).  Build / Match / Commit are development tool names (the stages of `verity-vllm row`), not layers.
Research terms (Tool, Derivation, Attempt, Artifact, Snapshot, Replica, Label) belong to `tools/research` and never name Verity semantics.

**v2 is the integration** (since the flip on `lane/vllm-v2-default`, Draft 3.1 §12.3): the required-value manifest of record is
`Q_word_v1{X=16,W=32}` over the Program + runtime correspondence: every Call boundary committed, the header naming the
Program's `verity/partition/v1` object, which the Commit checks and its binding map records, and the strict word check the Build passed
(`query.word_check`), without which the Commit refuses the manifest (`verity_vllm.query`: `verity-vllm manifest build | build-global`,
built at the end of a PASS Build by `verity-vllm row`); value acquisition follows the acquisition plan derived from it
(`verity_vllm.acquire`; TP rank committers still select by `native_host`'s class tables); the verdict of record is `<row>/verdict.json`
(`vllm-verdict/v1`, `verity_vllm.check.verdict`).  `commit_summary.json` and the `finalize.py` board are no longer written (P5).  The
v1 engine is deleted (recoverable at git tag `v1-reference`); the frozen R19 reference records on R2 that `tests/regression` compares
against (`VERITY_REGRESSION_ORACLE=frozen`) are records, not code.

Layers and where they live (Draft 3 §12.1):

| layer | responsibility | today |
| --- | --- | --- |
| Program | Definitions, Calls, dataflow, `program_digest` | `verity_vllm/program/registry`, `verity_vllm/program/frontend` (lowering rules) |
| query | Q(P), proof units, boundary and required values | `verity_vllm/query` (`module_body`, `word`), `verity_vllm/pipeline/manifest.py` (`verity-vllm manifest`: the manifest of record), `verity_vllm/check/poc_required_interface.py` |
| runtime correspondence | Call ↔ module path / fx node / slot | `verity_vllm/correspondence`, `verity_vllm/program/lifting`, `verity_vllm/correspondence/reader_for_query.py`, `verity_vllm/correspondence/runtime.py` |
| acquisition | acquisition plan, taps, hooks, collector | `verity_vllm/acquire` (`plan.py` ROUTE_OF_FAMILY, `install.py`, `stage.py`); hazard / analytic owners in `commit/committer/native_collect.py`, `commit/committer/native_host.py`, `commit/committer/hidden_gpu_src/` |
| observation + resolution | capture log, fold, occurrence resolution | `pipeline/m1_capture.py`, `observe/fold/fold.py`, `check/match/batch_decomp.py` |
| input provenance | weights of record, prompt ids, checkpoint pin | `check/weights_of_record.py`, `check/root_policy.py`, `manifests/checkpoints.json` |
| commitment | position-identified roots, openings, binding map | `commit/binding.py`, `commit/scheme.py`, `commit/committer/leafhash.py` |
| checks → verdicts | the seven checks above, one verdict of record per row | `check/match/global_match.py`, `check/replay/driver.py` (`sampled_replay`), `check/executed_prefix.py`, `query/manifest/verify.py`, `check/verdict.py` (`verdict.json`) |

`tests/test_no_by_name_rules.py` pins every remaining by-name rule (model names, module-path predicates, family tables) in
`tests/by_name_allowlist.json` with the phase that retires it; the list only shrinks.  `tests/regression/` is the R19 reference
regression: the current tree's checks against the frozen v1 records (`expected/<row>.json` re-baselined on v2; labelled decisions in
`fixtures.toml` are the recorded reasons the two differ).

## Python API

`verity_vllm.LLM` is a pinned vLLM engine, constructed and called like `vllm.LLM`:

~~~python
from vllm import SamplingParams
from verity_vllm import LLM

llm = LLM("unsloth/Llama-3.2-1B", revision="9535bd9b1d1dea6acafbdc4813b728796aeb28da")
outputs = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=16))
~~~

`model` and `revision` (the full commit sha) must be pinned in `manifests/checkpoints.json` with the snapshot on the host.
The engine is built by `engine.vllm_adapter.build_engine` from that pin's engine profile, as every `verity-vllm` stage
builds it: dtype, tokenizer and attention come from the profile unless explicitly overridden, and prefix caching is off.
`LLM.from_spec(path)` builds the same engine from an engine spec, a config v0 document (`verity_config`) of schema
`verity-vllm/engine-spec/v0` with `model`, `revision` and an `[engine]` table of exactly the options below
(`pipeline/engine_spec.py`).  An unknown or missing key, or a value of the wrong type, refuses before any engine is
built, and the spec's canonical SHA-256 is `engine_record["spec_digest"]`.
Supported keyword options:

| option | default | sets |
| --- | --- | --- |
| `tensor_parallel_size` | 1 | GPUs the model is sharded over |
| `max_num_seqs` | 8 | requests the scheduler runs together |
| `max_num_batched_tokens` | vLLM's | tokens per scheduler step |
| `max_model_len` | the pin's | context length |
| `gpu_memory_utilization` | the profile's | fraction of device memory the engine may take |
| `enforce_eager` | True | eager execution; False compiles with torch.compile and CUDA graphs |
| `dtype` | the profile's | served dtype override |
| `quantization` | the checkpoint's | vLLM quantization override, such as `"fp8"` |
| `compilation_config` | vLLM's | compilation and CUDA-graph settings |
| `seed` | 0 | the engine seed |
| `protocols` | none | protocol options installed on the engine: `"pouw:<scheme>,pous:<scheme>"` or `{name: scheme}` |
| `protocols_config` | none | each enabled protocol's settings, `{name: {...}}`; never hashed |

Any other option raises `TypeError` naming it and this list; an engine built elsewhere is not accepted.  `generate(prompts,
sampling_params=None, use_tqdm=True)` returns vLLM's `RequestOutput` list unchanged and appends a `GenerateRecord` (sampling
parameters, prompt and generated token ids per request) to `llm.records`; `llm.pin` is the resolved `EnginePin` and
`llm.engine_record` the engine as built (the `vllm.LLM` keyword arguments, the profile's environment flags, the execution vLLM
reports, the protocol composition record).  `llm.vllm` is the wrapped `vllm.LLM` (for driving its engine step by step) and
`llm.model` the served model in this process, the object protocol options install on.

## Hook interface

What a caller may attach to the engine, in what order, and what it may read back.  The names below are the surface other
slices build on (protocol options, the benchmarks harness); vllm owns them, and a change to one is announced to its users.

**One mechanism.**  Every runtime patch goes through `engine/hooks.py`: `patch`, `wrap`, `append` (a library's hook list)
and `service` (an in-process thread), each undone exactly by `uninstall`.  A component takes a `Hooks(tag=...)`, makes
every entry through it, and uninstalls it newest first; `hooks.live(tag)` lists what a component has installed.  Nothing
else replaces an attribute on a vLLM, torch or Triton object.

**What an add-on registers.**  Two kinds:

- *A protocol option* is one adapter module in `protocol_options/` with `NAME`, `schemes()`, `refusals(scheme, ctx)` and
  `install(scheme, target) -> Installed`, plus the optional `requires_build`, `EXECUTES`, `keeps_weight_copy` and
  `commit_weights` (the contract is `protocol_options/interface.py`).  It finds its sites by type, never by module name,
  and patches only through `target.hooks(entry)`.  Its protocol logic stays in its `verity.core.protocols` package.
- *A benchmark arm or experiment* patches only through a `Hooks` tagged by its id (`bench:<arm>`).  The harness
  (`verity_bench.vllm.runner`, benchmarks') calls its `install` before the engine is built and its `ready` once
  `llm.vllm` and `llm.model` exist; it refuses an arm whose `install` added a live entry outside its tag, and uninstalls
  the arm's hooks when the engine closes.

**Order.**  At each site the nesting is `protocol_options.SITES`, outermost first: at `LinearBase.quant_method.apply`,
POUS wraps PoUW, which executes (a site has at most one executor, `EXECUTES`).  Sampled proofs commits at module
boundaries, outside every site.  A composition runs in two phases: every option first commits to the plaintext weights
(`commit_weights`, model untouched), then each installs in `config.PROTOCOLS` order, innermost first.  Uninstall is the
reverse, and an install that raises unwinds every hook already made.  Where it happens depends on the stage: `LLM` installs right
after the engine is built; a Match capture installs the executors and wrappers only, no sampled proofs (`at_capture`); a Commit
installs after its committers register the weights of record and after its warm-ups (`at_commit`).  An arm's hooks are
installed before the engine is built; how they nest with a protocol option at a site in `SITES` is not stated yet, so
an arm patches a site of its own or runs without protocol options.

**Multi-GPU.**  Under `tensor_parallel_size > 1` the model, its weights and every kernel live in the rank worker processes,
so a hook installed in the driver sees nothing.  The seam is vLLM's `worker_extension_cls`: `engine/rank_worker.py` mixes
methods into every rank's worker, and the driver runs them on each rank with `LLM.collective_rpc`.  The TP capture and
Commit drivers (`pipeline/tp/`) install per-rank observers and committers this way, and `collectives/collective_sites.py`
is the one finder of the collective sites they hook.  Protocol options do not compose across ranks yet: `refusals`
refuses any set at TP > 1.  The planned shape is the same seam: one `install` per rank with `Target.rank`, a per-rank
`Composition`, and a record that lists the ranks; that refusal lifts in the change that wires it.

**Per-step timing.**  A caller that times the engine drives it with
`engine.arrivals.run_requests_arrivals(llm.vllm, requests, params_for, clock=..., request_timing=True, step_timing=True)`:

- `timing["requests"][rid]` carries `arrival_s`, `first_output_s` and `time_to_first_token_s`;
- `timing["steps"]` holds one entry per `step()` call: `iter`, `s` (the clock's duration of that call), `outputs`, `forward`
  and `forward_index`, so a step that only drained a finish isn't counted as a forward.  It also has `prefill_tokens` and
  `decode_tokens`, what that step's schedule computed, read from the in-process scheduler (None when it isn't reachable);
- `wall_s`, `engine_steps` and the token rates are the whole run's;
- `on_step(outputs)` is called after every step, for a caller's own per-step reads;
- the protocol options' own counters are `llm.protocols.info()`.

The durations are host time around `step()`, and include device time only where vLLM synchronizes inside the step.  A
caller that needs device time synchronizes around the run itself, as the harness does.  `clock` is injectable, so tests
measure on a stub clock.  The capture's timing-only mode (`engine.capture.Capture`, its `step_times` with `forward_s` and
`sample_s`) times an observed run, not a served one.

## Protocol options

PoUW, POUS and sampled proofs are protocol options (`verity_vllm/protocol_options/`), each enabled on its own and any of them
in one run.  A set is `name:scheme` entries, one per protocol (`config.parse_protocols`), off by default: `LLM(protocols=...)`
installs PoUW and POUS on a serving engine; a row takes its set from `verity-vllm row --protocols` (`PROTOCOLS`) or its
workload's target, every stage reads it as the target's `protocols` (`TargetProfile.protocols`; unset is sampled proofs
alone, the Commit as it has always run), `run_row` refuses a set it cannot run before any stage, and the Commit installs
the others once its committers registered the weights of record (the replay reads live weights through what an option
released: `protocol_options.weights_view`).  Adapters install innermost first (PoUW replaces the linear GEMMs, POUS decodes the
weights they read, sampled proofs commits what the model computed) and uninstall in reverse, every patch and service through
`engine/hooks.py` tagged with its option.  The protocol logic is `verity.core.protocols.pouw`, `verity.core.protocols.pous` and `verity.core.protocols.sampled_proofs`';
the adapters (`pouw.py`, `pous.py`, `sampled_proofs.py`) are thin.  A Commit admits only options its Program describes (POUS's
bit-identical decode; an executor whose scheme has a circuit in its protocol package, `protocol_options.CIRCUITS` and
`traced_as`: PoUW's `ncp-v2`), so its verdict is of record as usual and carries the set
(`verdict.protocols`) and the composition record's digest.
PoUW is refused beside sampled proofs until its modeled circuit lands (its `traced_as` is None).  PoUW's settings are `coins` (the epoch's challenger,
`verity.core.protocols.pouw.coins.Challenger`, held in the auditor's process: install opens its salt coin
after the weights' root and refuses without one; a pipeline that is its run's own auditor supplies one,
`pouw.challenged`), `retain_steps` and `transcript_dir`. Its weight operands are formed per forward from the weight each call is handed, so
PoUW keeps no weight-derived copy and POUS composes with it. A call holds the formed operands of the one layer it
executes (3 B per weight of that layer: 1.41 GB for a 70B gate_up), plus under `FORM_BYTES` (256 MiB) of host work. PoUW executes `ncp-v1` and `ncp-v1-shift24` on the host in numpy, and its
placeholders are listed in its `info()`.

**PoUW is not auditable yet.**

- There is one epoch per install, with no epoch rotation.
- Activation rows are kept only in memory, for the last `retain_steps` units (0 by default), and are never persisted.
- There is no opener for a drawn tile.
- The transcript's `leaves` grow without bound for the life of the install: 32 B per 16×16 tile, about 0.84 MB per
  token at 70B.

## Running

~~~sh
uv sync --all-packages --extra torch-cpu     # CPU-only torch; the `torch` extra is the CUDA build; vLLM only on a pod
cd integrations/vllm
uv run pytest tests/program verity_vllm        # the CPU oracle; tests that need CUDA / vLLM / a pod tree skip or fail by name
~~~

Tests and tools read fixtures relative to this directory: run them from here.  The repository's merge gate (`check`,
`tools/verity/check/check.py`) does not collect them: `integrations/vllm` is outside the root `pyproject.toml` testpaths.  Every tool is a `verity-vllm` command
(`verity-vllm --help`); a sweep row runs with `verity-vllm row run <row-id> <ROLE> <hf-repo> <revision>`.

The pod runtime (Python 3.12, torch cu129, vLLM `d9105ea80`, native taps, `hidden_gpu`) is built by `verity_vllm/ops/pod_bootstrap.sh`
and exercised through `research run --source <sha> --on <machine>`.  On a pod without a GPU (`nvidia-smi` absent, or `--cpu`)
it skips the native taps and writes the same `readiness.json` with `"mode": "cpu"`; `--dry-run` prints the resolved mode.

## Moving the engine pin

The engine is vLLM at one commit, today `d9105ea80`, and vllm owns which commit that is.  The pin is stated in several
places, and a move rewrites all of them in one PR:

- the wheel URL and its sha256 in `ops/pod_bootstrap.sh` (step 2), and its readiness check `vllm_commit_<sha9>`;
- the engine profiles named for it (`engine/profiles/vllm_<sha9>_*`), whose κ pins were recorded from that build;
- benchmarks' `VLLM_PIN` (`verity_bench.vllm.presets`), the `vllm_pin` of each benchmark spec, and PoUW's
  serving venv check (`benchmarks/pouw/pearl_c_vllm/setup.sh`).

`git grep <old sha9>` lists the rest: docstrings and tests that cite the old build's source.

**Who asks.** Any user of the integration asks @vllm, which files the ask as a VLLM issue.  The ask names the target
commit, why the move is needed (a fix, a model, a kernel), and the user's paths it touches.  vllm decides, and does the
move on one branch.

**What must pass.**  Each of the following is a run on the new pin, and each run id goes in the move's PR body:

1. *Bootstrap:* `ops/pod_bootstrap.sh` with the new wheel's sha256 reaches readiness PASS on a GPU pod.
2. *Tokens:* greedy tokens identical to the old pin's for each checkpoint a user serves, through `verity_vllm.LLM`.  A
   difference is named, and it stands only if the owner of every Definition it touches accepts it.
3. *circuits:* the kernel census against `properties/kernel_allowlist.py`, and a Build/Match/Commit GO on the rows of
   record.  Every Definition of a served kernel is bound to the pinned build's kernels, so new kernels mean new profiles.
4. *compute-accounting:* PoUW's device gate with its executor installed at `LinearBase.quant_method.apply`.
5. *memory-accounting:* POUS's install and its bit-identical decode gate.
6. *proofs:* a sampled-proofs Commit whose verdict PASSes.
7. *benchmarks:* the quoted overheads re-run from the specs at the new `vllm_pin`.  The pin is part of each result's
   identity, so a number measured at one pin is never compared with a number measured at another.

A user whose code the move doesn't reach may answer "not affected" instead of a run id.  The move lands once every user
has answered.

**How users hear.**  When the PR is marked ready, vllm announces the move to each user's handle
(`research msg send --kind announce --to ...`).  The announcement pins the merge commit (`git:<sha>`) and the wheel's
sha256 (`--ships`), and each user closes it with a status line once its pods run the new pin.
