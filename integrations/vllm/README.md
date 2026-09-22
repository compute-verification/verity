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

## Running

~~~sh
uv sync --all-packages --all-extras          # torch (CPU) comes from the `torch` extra; vLLM only on a pod
cd integrations/vllm
uv run pytest tests/ir verity_capture        # the CPU oracle; tests that need CUDA / vLLM / a pod tree skip or fail by name
~~~

Tests and tools read fixtures relative to this directory: run them from here.

The pod runtime (Python 3.12, torch cu129, vLLM `d9105ea80`, native taps, `hidden_gpu`) is built by `pod_bootstrap.sh`
and exercised through `research run --source <sha> --on <machine>`.
