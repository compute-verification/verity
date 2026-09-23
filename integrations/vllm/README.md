# verity-vllm

The Verity x vLLM integration: the Torch/vLLM-derived Programs (`verity_vllm.registry`, `verity_vllm.frontend`), the
runtime capture and Match/Commit machinery (`verity_capture`), and the qualification sweep (`verity_capture/sweep`,
`workloads/`).  Promoted from `veritor@0903f60d6` (release `r18-int-3`) as one working closure; see the Notion page
"Veritor → Verity migration" for the decisions and the evidence index.

## Layout

The tree keeps veritor's repo-root-relative shape so the 100+ `__file__`-relative path computations and the
`PYTHONPATH=.` convention keep working: the integration root plays the role the veritor root and `vllm-poc/` played.

~~~text
verity_vllm/            was vllm-poc/verity_vllm, plus the vLLM-only half of verity_ir/:
  registry/               verity_ir/registry     definitions of the served program (b1, prims, lifted, hopper, fp8, ...)
  frontend/               verity_ir/frontend     Torch -> Program derivation (rules/, derive, module_query)
  lifting/                verity_ir/{lifted_record, lifted_correspondence, lifted_dump, continuation}
  correspondence/         verity_ir/correspondence{,_compiled,_program,_tp2,_tp2_compiled} -> core, compiled, program, tp2, tp2_compiled
  ir/                     verity_ir/{accounting, boundary, partition, query_artifact, workload, sampling_event,
                                     store_nan_scan, lazy, profile_descriptor, descriptor_equivalence}
verity_vllm_adapter/    was vllm-poc/verity_vllm_adapter        (same for _numerics)
verity_capture/         was verity_capture (experimental/ included: every subpackage is in the import closure)
manifests/              was vllm-poc/manifests
workloads/              was out/gen/sweep/workloads  (the 96 row manifests of the qualification table)
fixtures/               was vllm-poc/fixtures (MUFU tables, cos/sin) and vllm-poc/results/GUEST-* -> fixtures/results/
data/                   repo-root fixtures: out/gen/hf_configs -> hf_configs, out/census -> census, out/contract -> contract,
                        docs/data/rec -> rec, out/capture/logs -> logs
tests/ir/               was verity_ir/tests
~~~

Removed after the promotion (cleanup 1, 2026-09-22): `e2e/`, `record_v5/`, `scripts/`, `tests/poc/`, `verity_vllm_sampler/` and the
parts of the other packages unreachable from the campaign path (`verity_capture/sweep/*.sh`, `pod_bootstrap.sh`) and the test suite.

The stdlib-only core of `verity_ir` (`types refs defs program layout evaluate query codec parts query_codec`, `gateset`
as `query_ast`, `plan`) is `verity.ir` / `verity.verification.plan` in `packages/verity`; the on-demand primitive families
veritor hardcoded in `codec.py` are registered by `verity_vllm/registry/__init__.py` through `verity.ir.codec.register_lazy_family`.

Not migrated: `verity_ir/experimental` (no production consumer), `src/veritor` and the other pre-`verity_ir` stacks, the veritor pod scripts that
`research run` replaces (`pod_up.sh`, `ship.sh`, `pod_release.sh`, `pod_ready*.sh`, `preflight.sh`).

## Vocabulary (frozen; Notion "vLLM integration v2 — Draft 3" §12)

Prose, docstrings and Notion use these words.  Serialized field names (`member`, `op_path`, `output_member`, `query: Q_module_v1`,
`weights_pin`, ...) are not renamed: a header string inside a digested document changes its digest.

Core (Verity semantics, `packages/verity`):

- **Definition** — a registered function with a total evaluator on encodings (`verity_vllm.registry`).
- **Call** — one application of a Definition inside a Program (retires "Program row", "instance" in prose; `instances.json.gz` stores Calls).
- **Value** — an input or output of a Call (retires "Member" in prose; the serialized `member` / `output_member` field keeps its name).
- **Program** — the sequence of Calls with its dataflow; `program_digest` covers the dataflow only.
- **verification query** — Q: which sets of Calls are **verification units (VU)**; `Q_module_v1` is called **Q_module_body_v1** in prose
  (units = module bodies, submodule calls excluded; the serialized id stays until a labelled decision).
- **boundary values** — `⋃_{V∈Q(P)} (in(V) ∪ out(V))`; **required values** = boundary values ∪ prescribed inputs (weights, prompt ids)
  ∪ protocol-required values (sampled ids); **commitment** — the position-identified Merkle roots over the required values;
  **verification obligation** — what a verifier must be able to establish from the commitment.

Integration (this tree):

- **lowering rule** (was Rule) — turns an exported-graph node into a Call (`verity_vllm/frontend/rules`).
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
`Q_module_body_v1` over the Program + runtime correspondence (`verity_vllm.query`: `python -m verity_vllm.query.cli build | build-global`,
built at the end of a PASS Build by `row_pod.sh`); value acquisition follows the acquisition plan derived from it
(`verity_vllm.acquire`, `ACQUIRE_ENGINE=v2`, the default); the verdict of record is `<row>/verdict.json` (`vllm-verdict/v1`,
`verity_capture.commit.verdict`).  `commit_summary.json` and the `finalize.py` board are no longer written (P5).  The v1 engine
(`ACQUIRE_ENGINE=v1`; `verity_capture.commit.required_manifest` by-name tables) is selectable for one more release as the
**regression oracle only** -- the oracle proper is the frozen R19 reference records on R2 that `tests/regression` compares against
(`VERITY_REGRESSION_ORACLE=frozen`), not the v1 code.

Layers and where they live (Draft 3 §12.1):

| layer | responsibility | today |
| --- | --- | --- |
| Program | Definitions, Calls, dataflow, `program_digest` | `verity_vllm/registry`, `verity_vllm/frontend` (lowering rules) |
| query | Q(P), VUs, boundary and required values | `verity_vllm/query` (`module_body`, `cli`: the manifest of record), `verity_vllm/required.py`; `verity_capture/vu_canonical.py` |
| runtime correspondence | Call ↔ module path / fx node / slot | `verity_vllm/correspondence`, `verity_vllm/lifting`, `verity_vllm/query/correspondence.py`, `verity_vllm/runtime_correspondence.py` |
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
uv run pytest tests/ir verity_capture        # the CPU oracle; tests that need CUDA / vLLM / a pod tree skip or fail by name
~~~

Tests and tools read fixtures relative to this directory: run them from here.

The pod runtime (Python 3.12, torch cu129, vLLM `d9105ea80`, native taps, `hidden_gpu`) is built by `pod_bootstrap.sh`
and exercised through `research run --source <sha> --on <machine>`.
