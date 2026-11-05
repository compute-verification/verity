"""verity_vllm.acquire: value acquisition derived from the required-value manifest + runtime correspondence.

An ACQUISITION PLAN is derived from the required-value manifest and the runtime correspondence (`plan.acquisition_plan`), and exactly
the plan's hooks are installed (`install.install_from_plan`, `install.apply_plan_to_committer`).  TP rank committers (`pipeline/tp/rank_worker.py`)
have no plan yet: they still select their modules by `native_host`'s class tables.

- `plan`            `acquisition_plan(manifest, correspondence) -> AcquisitionPlan` (sites per Value, policy, unsupported, residuals; `acquisition-plan/v1`),
                    over `verity_vllm.correspondence.reader_for_acquire`: the ONE reader of where a Call lives at runtime (the
                    `runtime-correspondence/v1` record when the Build carries it, else the manifest's v1-annotation reading -- `source`
                    recorded in the plan's `correspondence` block, never silent) and the runtime's module tree (`RuntimeTree`)
- `install`         hooks from the plan (module path -> hook kind -> slot); the committer adapter
- `stage`           the plan at a GPU phase's entry
- `gate`            the CPU gate (the plan of a row, and plan-vs-plan comparison) as a CLI / function
- `sources`         the acquisition sources the plan's hooks feed (attention, MoE, compiled kernels, rank partials, ...)
- `served`          the served-compiled Program and its required-value manifest, with `served_device`, `served_moe`,
                    `served_artifact` and Inductor's kernels as templates (`inductor_templates`, `inductor_texts`)
"""
