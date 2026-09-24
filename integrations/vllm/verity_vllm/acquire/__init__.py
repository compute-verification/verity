"""verity_vllm.acquire: value acquisition derived from the required-value manifest + runtime correspondence (v2, Phase 4a).

An ACQUISITION PLAN is derived from the required-value manifest and the runtime correspondence (`plan.acquisition_plan`), and exactly
the plan's hooks are installed (`install.install_from_plan`, `install.apply_plan_to_committer`).  TP rank committers (`tp/worker.py`)
have no plan yet: they still select their modules by `native_host`'s class tables.

- `correspondence`  `owning_module(call)` / `return_slot(call, port)` / `arg_slot(call, i)` over the ONE reader of where a Call lives at runtime
                    (`verity_vllm.runtime_correspondence`, the `runtime-correspondence/v1` record) when the Build carries it, else the
                    manifest's v1-annotation reading -- `source` recorded in the plan's `correspondence` block, never silent
- `runtime_tree`    the runtime's module tree (`RuntimeTree`: module paths, MRO classes, buffers, runner facts; live engine or meta device)
- `plan`            `acquisition_plan(manifest, correspondence) -> AcquisitionPlan` (sites per Value, policy, unsupported, residuals; `acquisition-plan/v1`)
- `install`         hooks from the plan (module path -> hook kind -> slot); the committer adapter
- `stage`           the plan at a GPU stage's entry
- `gate`            the CPU gate (the plan of a row, and plan-vs-plan comparison) as a CLI / function
"""
