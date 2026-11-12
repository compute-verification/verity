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
- `gate`            the CPU gate of a row: its plan (a function), and whether every call-boundary identity of its manifest
                    has a source its Commit attaches (a CLI)
- `sources`         the acquisition sources the plan's hooks feed (attention, MoE, compiled kernels, rank partials, ...)
- `served`          the served-compiled Program and its required-value manifest, with `served_device`, `served_moe`,
                    `served_artifact` and Inductor's kernels as templates (`inductor_templates`, `inductor_texts`)
- `compiled`        the required-value manifest of an `Execution=compiled` workload, from the serving process's compiler artifacts
- `committer_api`   the committer interface the Δ_commit harness drives, and the coverage declaration every committer emits
- `fa2_plane_classes`  the FA2/FA3 hidden planes a step is expected to hold, derived per attention layer
- `h1_commit`       the Commit in H1's serving engine: its committer, its int8 source, its step-input port and H1's tap
- `as_loaded`       the MXFP4 Marlin experts' weights as loaded, which a TP rank's Commit registers under their names
- `fa2_tap_src`, `fa3_tap_src`, `norm_tap_src`, `router_tap_src`
                    the taps' kernels and the builds of their torch extensions (FA2, FA3, the norm scale, the router softmax)
"""
