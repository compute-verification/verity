"""verity_vllm.acquire: value acquisition derived from the required-value manifest + runtime correspondence (v2, Phase 4a).

v1 decides WHERE to observe a Value by module CLASS tables (`native_host.ACQUIRE_CLASSES`, `ACQUIRE_INPUT_CLASSES`, the `HAZARD_LEAVES` /
flush leaf tables) plus the manifest-driven consumer-input rule for promoted Values.  v2 derives an ACQUISITION PLAN from the
required-value manifest and the runtime correspondence (`plan.acquisition_plan`) and installs exactly the plan's hooks
(`install.install_from_plan`, `install.apply_plan_to_committer`).  v1 stays the default; `switch.engine()` reads `ACQUIRE_ENGINE`.

- `correspondence`  the tiny adapter over today's annotations (`owning_module(call)`, `return_slot(call, port)`, `arg_slot(call, i)`) and
                    the runtime module tree (`RuntimeTree`); the parallel correspondence lane replaces the adapter with one import
- `plan`            `acquisition_plan(manifest, correspondence) -> AcquisitionPlan` (sites per Value, policy, unsupported, residuals; `acquisition-plan/v1`)
- `install`         hooks from the plan (module path -> hook kind -> slot); the collector adapter behind the switch
- `v1_decision`     v1's hook decision as a pure function of the same inputs (the CPU gate compares it to the plan's site set)
- `gate`            the site-set comparison (v1 vs plan) as a CLI / function
"""
