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
Imports point down the layer order `pipeline` > `check` | `properties` > `acquire` > `observe` | `commit` > `engine` >
`query` | `correspondence` > `program` > `collectives` > `config` (lint P9).

~~~text
verity_vllm/
  pipeline/           stage drivers      build.py match.py commit.py report.py hot.py layout.py cli.py research_tools.py admission_*.py tp/
  engine/             the vLLM engine    vllm_adapter.py engine_driver.py engine_profile.py arrivals.py rank_worker.py profiles/
  program/            Program            registry/ frontend/ lifting/ kernels/ model.py workload.py compact.py dtypes.py
  query/              verification query module_body.py required.py program_view.py manifest/ compare.py
  correspondence/     runtime corresp.   runtime.py reader_for_*.py resolve*.py capture_identities.py
  observe/            value observation  observer.py events.py log.py storage.py contract.py runtime_tree.py chunks.py fold/
  acquire/            value acquisition  plan.py install.py stage.py gate.py compiled.py sources/ *_tap_src/
  commit/             commitment         merkle.py hashing.py identity.py binding.py padding_steps.py hidden_*.py committer/
  check/              checks -> verdicts match/ replay/ executed_prefix.py verdict.py gates.py weights_of_record.py root_policy.py ...
  properties/         property harnesses noninterference.py census.py kernel_allowlist.py golden.py holdout.py admission.py ...
  collectives/        collectives        collective_record.py collective_sites.py embedding_shard.py
  config.py target_family.py             run-definition locations; the target-family preflight
  ops/                pod operations     pod_bootstrap.sh row_pod.sh run_row_v2.sh pod_*_tap.sh canary.sh known_roots.json
tests/<subpackage>/   mirrors the package; tests/regression = the R19 reference harness (opt in with VERITY_REGRESSION=1)
workloads/ manifests/ fixtures/ data/ docs/    records of the qualification table; never rewritten by the tool
~~~

**`pipeline/`** holds the stage drivers (Build / Match / Commit): `build.py`, `match.py`, `commit.py`, `hot.py`, `report.py`,
`global_program.py` (the rooted workload Program of a Build), `m1_capture.py`, `layout.py` (Build / run directory layout),
`cli.py` (the one CLI), `timeline.py` / `spans.py` / `telemetry/`, `admission_bound.py` / `admission_planner.py`,
`research_outputs.py` / `research_result.py`, `source_identity.py`, `release_json.py`, `workload.py` /
`coverage_workloads.py`, `experiment.py`, `research_tools.py` (the `vllm.build` / `vllm.match` / `vllm.commit` Tools of
`tools/research`), and `tp/` (the rank-aware capture, commit, match and fold-match drivers).  Entrypoints:
`python -m verity_vllm.pipeline.cli`, `python -m verity_vllm.pipeline.m1_capture`.  Never: Verity semantics -- a check that
lives here is misfiled.

**`engine/`** is the vLLM engine we attach to: `vllm_adapter.py`, `engine_driver.py`, `engine_profile.py`, `arrivals.py`,
`rank_worker.py` (the TP rank worker extension), and `profiles/` (kappa: capture profiles and workload JSON; one generic
constructor + a facts table).  Never: commit, decide what is required.

**`program/`** implements *Program*: Definitions (`registry/`, the registered library; importing it registers the on-demand
primitive families through `verity.ir.codec.register_lazy_family`), lowering (`frontend/`: the torch frontend, `derive`, `rules/`
= lowering rules, the vLLM meta export, `target_profile`, the correspondence emitter `emit.py`, the collective export ops),
lifted / padded Programs (`lifting/`), `workload.py` / `compact.py`, `model.py` (specified computations such as bf16(sqrt(H))),
and `kernels/`, the independent kernel models the Definitions wrap (numpy / C++ / CUDA models, twins, relation checkers and
their tables).  Entrypoints: `program.frontend.derive`, `program.registry`.  Never: module paths as identity, hook sets,
VU policy, model-name tables (`registry/quarantine/` is the by-name residue and only shrinks).

**`query/`** implements the *verification query*: `module_body.py` (Q_module_body_v1), `program_view.py`, `manifest/` (the
required-value manifest: `format.py` the shared vocabulary, `verify.py` the rebuild-and-compare check of record),
`query_artifact.py` (query artifact identities), `compare.py` (a v2 manifest against a v1 one), `required.py` (v1
`IDENTITY_KEY` rows).  Boundary values, VU partitions and dead gates are core's `verity.ir.boundary`, `verity.ir.partition`
and `verity.ir.liveness`.  Never: execution data, hooks, model names, anything read from a run.

**`correspondence/`** implements *runtime correspondence* as data: `runtime.py` (runtime-correspondence/v1),
`reader_for_query.py` / `reader_for_acquire.py`; and *occurrence resolution*: `resolve.py`, `resolve_decomp.py`.  Never:
verification policy, commitment positions, requiredness, verdicts.

**`observe/`** implements *value observation* (the Match-side capture): `observer.py` (the TorchDispatchMode), `events.py` /
`log.py` / `storage.py`, `contract.py` (the one observation contract), `triton_adapter.py`, `runtime_tree.py`, `chunks.py`
(chunk attribution), and `fold/` (raw log -> Program: `fold.py`, `tree.py`, `memory.py`, `views.py`, `resolver.py`,
`patterns*.py`, `collective_pattern.py`).  Never: commit, decide what is required, name a model.

**`acquire/`** implements *value acquisition*: `plan.py` (required values + correspondence -> acquisition plan), `install.py`,
`stage.py`, `compiled.py` (the manifest of `Execution=compiled` rows), `sources/` (`hidden_source.py` / `moe_source.py` /
`compiled_source.py` / `compiled_kernel_source.py` / `partial_source.py`), `fa2_plane_classes.py`, `committer_api.py`, and the
shipped tap sources (`fa2_tap_src/`, `fa3_tap_src/`).  `gate.py` is the CPU gate (one row's acquisition plan, off-pod).
Never: decide what is required; never a model-name list.

**`commit/`** implements *commitment*: `merkle.py` (trees over canonical ids, `path_shape` / `fold_path`, openings), `hashing.py`
(`H(tag, *parts)`), `identity.py`, `binding.py` (binding map + coverage), `padding_steps.py`, `fasttree.py` / `semantic_layout.py`,
`hidden_stream.py` + `hidden_engine.py`, and `committer/`, the GPU committer (`native_host.py`, the collector of record;
`native_collect.py` + `native_*.cu` / `.cpp`, `native_jit.py`, `leafhash.py`, `flush_points.py`, `hidden_gpu_src/`).  Never:
runtime metadata in the identity tuple.

**`check/`** implements *checks -> verdicts*: `match/` (`global_match.py`, `global_match_fast.py`, `program_compare.py`,
`batch_decomp.py`, `rank_match.py`, `xrank_collectives.py`), `replay/` (`sampled_replay.py`, `replay*.py`,
`stoch_recompute.py`, `compiled_kernel_check.py`, `vu_query.py`), `gates.py` G1..G8, execution extent (`executed_prefix.py`),
`verdict.py` + `commit_verdict.py`, `value_check.py` / `compiled_value_check.py` / `oracle_compare.py`, `kernel_identity.py`,
`fold_compare.py`, `operand_provenance.py`, and input provenance (`weights_of_record.py`: checkpoint basis, `weights_pin`;
`root_policy.py`); `poc_*.py` are v1 leftovers.  Never: acquire or commit anything.

**`properties/`** holds the property harnesses: `noninterference.py`, `census.py` + `kernel_allowlist.py`, `golden.py` +
`golden/`, `holdout.py`, `admission.py` (differential admission tests), `quarantine_lint.py`, `protected.py`,
`fa_tap_exactness.py`.  Never: a per-run check.

**`collectives/`** holds the collective record (`collective_record.py`), the communicator hook sites (`collective_sites.py`)
and the embedding shard layout (`embedding_shard.py`).

**`ops/`** is pod operations: `pod_bootstrap.sh`, `row_pod.sh` / `row_pod_tp2.sh` / `run_row_v2.sh` / `tp_stage.sh` (the row
runner and its stages), `pod_fa2_tap.sh` / `pod_fa3_tap.sh` / `pod_hidden_gpu.sh`, `canary.sh`, `compiled_commit.sh`, the
`*_negative*.sh` runs, `cov_pod.sh`, the gates (`pod_gate.sh`, `verify_lane.sh`), `known_roots.json`.  Never: Python that a layer
above would import.

The stdlib-only core of `verity_ir` (`types refs defs program layout evaluate query codec parts query_codec`, `gateset`
as `query_ast`, `plan`) is `verity.ir` / `verity.verification.plan` in `packages/verity`; the on-demand primitive families
veritor hardcoded in `codec.py` are registered by `verity_vllm/program/registry/__init__.py` through `verity.ir.codec.register_lazy_family`.

## Vocabulary (frozen; Notion "vLLM integration v2 — Draft 3" §12)

Prose, docstrings and Notion use these words.  Serialized field names (`member`, `op_path`, `output_member`, `query: Q_module_v1`,
`weights_pin`, ...) are not renamed: a header string inside a digested document changes its digest.

Core (Verity semantics, `packages/verity`):

- **Definition** — a registered function with a total evaluator on encodings (`verity_vllm.program.registry`).
- **Call** — one application of a Definition inside a Program (retires "Program row", "instance" in prose; `instances.json.gz` stores Calls).
- **Value** — an input or output of a Call (retires "Member" in prose; the serialized `member` / `output_member` field keeps its name).
- **Program** — the sequence of Calls with its dataflow; `program_digest` covers the dataflow only.
- **verification query** — Q: which sets of Calls are **verification units (VU)**; `Q_module_v1` is called **Q_module_body_v1** in prose
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
Program/source identity (`identity`).  Build / Match / Commit are development tool names (the stages of `row_pod.sh`), not layers.
Research terms (Tool, Derivation, Attempt, Artifact, Snapshot, Replica, Label) belong to `tools/research` and never name Verity semantics.

**v2 is the integration** (since the flip on `lane/vllm-v2-default`, Draft 3.1 §12.3): the required-value manifest of record is
`Q_module_body_v1` over the Program + runtime correspondence (`verity_vllm.query`: `python -m verity_vllm.pipeline.cli build | build-global`,
built at the end of a PASS Build by `row_pod.sh`); value acquisition follows the acquisition plan derived from it
(`verity_vllm.acquire`; TP rank committers still select by `native_host`'s class tables); the verdict of record is `<row>/verdict.json`
(`vllm-verdict/v1`, `verity_vllm.check.verdict`).  `commit_summary.json` and the `finalize.py` board are no longer written (P5).  The
v1 engine is deleted (recoverable at git tag `v1-reference`); the frozen R19 reference records on R2 that `tests/regression` compares
against (`VERITY_REGRESSION_ORACLE=frozen`) are records, not code.

Layers and where they live (Draft 3 §12.1):

| layer | responsibility | today |
| --- | --- | --- |
| Program | Definitions, Calls, dataflow, `program_digest` | `verity_vllm/program/registry`, `verity_vllm/program/frontend` (lowering rules) |
| query | Q(P), VUs, boundary and required values | `verity_vllm/query` (`module_body`, `cli`: the manifest of record), `verity_vllm/check/poc_required_interface.py`; `verity_vllm/observe/vu_canonical.py` |
| runtime correspondence | Call ↔ module path / fx node / slot | `verity_vllm/correspondence`, `verity_vllm/program/lifting`, `verity_vllm/correspondence/reader_for_query.py`, `verity_vllm/correspondence/runtime.py` |
| acquisition | acquisition plan, taps, hooks, collector | `verity_vllm/acquire` (`plan.py` ROUTE_OF_FAMILY, `install.py`, `stage.py`); hazard / analytic owners in `commit/native_collect.py`, `experimental/commit_integ/native_host.py`, `sweep/hidden_gpu_src/` |
| observation + resolution | capture log, fold, occurrence resolution | `bench/m1_capture.py`, `fold.py`, `experimental/cb_a/batch_decomp.py` |
| input provenance | weights of record, prompt ids, checkpoint pin | `commit/weights_of_record.py`, `root_policy.py`, `manifests/checkpoints.json` |
| commitment | position-identified roots, openings, binding map | `commit/binding.py`, `commit/leafhash.py`, `experimental/fa2_commit/engine.py` |
| checks → verdicts | the seven checks above, one verdict of record per row | `experimental/cb_a/global_match.py`, `commit/sampled_replay.py`, `commit/executed_prefix.py`, `commit/manifest_verify.py`, `commit/verdict.py` (`verdict.json`) |

`tests/test_no_by_name_rules.py` pins every remaining by-name rule (model names, module-path predicates, family tables) in
`tests/by_name_allowlist.json` with the phase that retires it; the list only shrinks.  `tests/regression/` is the R19 reference
regression: the current tree's checks against the frozen v1 records (`expected/<row>.json` re-baselined on v2; labelled decisions in
`fixtures.toml` are the recorded reasons the two differ).

## Running

~~~sh
uv sync --all-packages --all-extras          # torch (CPU) comes from the `torch` extra; vLLM only on a pod
cd integrations/vllm
uv run pytest tests/program verity_vllm        # the CPU oracle; tests that need CUDA / vLLM / a pod tree skip or fail by name
~~~

Tests and tools read fixtures relative to this directory: run them from here.

The pod runtime (Python 3.12, torch cu129, vLLM `d9105ea80`, native taps, `hidden_gpu`) is built by `verity_vllm/ops/pod_bootstrap.sh`
and exercised through `research run --source <sha> --on <machine>`.  On a pod without a GPU (`nvidia-smi` absent, or `--cpu`)
it skips the native taps and writes the same `readiness.json` with `"mode": "cpu"`; `--dry-run` prints the resolved mode.
