# verity_vllm.program.frontend

The Torch-to-circuit translator: a traced torch model becomes a Program in which every Call names a registered
Definition. `derive` exports the model and walks its graph (the walk's state is `derive_context`), `rules/` holds the
rules the walk applies, `targets` and `gemm_targets` bind a rule to the kernels of the served hardware, `target_profile`
is the target a Program is derived for, and `emit` writes a finished derivation's runtime correspondence. `train_step`
derives one training step as a Program. Each module's docstring says what it holds.

## What other packages import

A module of this directory that another package of the adapter, a benchmark or a tool imports from declares those
names in its `__all__`; a name outside that list is the translator's own. The package itself gives no name: each is
imported from its module, and importing a module loads only what that module imports. From the rest of the adapter:
`derive.derive_report`, the entry point (`pipeline`); `target_profile`'s `TargetProfile`, `DEFAULT_TARGET`,
`ArtifactIdentity`, `parse_target_arg` and `profile_mismatches` (`engine`, `observe`, `pipeline`, `program`, `query`);
the specs, launch contexts and tensor-core steps of `targets` (`program`, `observe`, `acquire`, `pipeline`);
`gemm_targets`' `gemm_target_for`, `try_gemm_target_for`, `UnregisteredGemmTarget` and `ADA_TUNED_GEMM` (`acquire`,
`observe`, `pipeline`, `program`); the two served rules `seed_rule` and `stop_rule`, each with its name, parameters,
Definition and check (`program`, `check`, `pipeline`); what a build needs of vLLM and of the export, in `vllm_meta`,
`export_compat`, `export_ops`, `triton_capture`, `loader_parity`, `relabel`, `inputs_trace` and `boolean` (`pipeline`,
`engine`, `observe`); which kernels a loaded model applies, in `kernel_readback` (`pipeline`, `engine`, `protocols`);
`collective_kinds` (`pipeline`, `query`); `dsv4_leaves.FP8_DS_MLA_QB` (`observe`); and
`marlin_moe_leaf.mxfp4_marlin_layers` (`acquire`). From outside the adapter: `benchmarks/network_sim` reads
`derive.derive_report` and `derive_examples.GreedyServe`, `benchmarks/harness` reads `kernel_readback.int8_exact_of`,
`benchmarks/one_stage`, `benchmarks/spot_proved` and `tools/verity/circuit_check` read specs of `targets`, and
`archive/benchmarks/training_slowdown` reads `derive.derive_report`. `target_profile` also passes on four names of
`kernel_readback` that its own list leaves out: `applied_kernels`, `fp8_block_gemm_kernel_of`, `int8_exact_of` and
`moe_experts_class_of`. The rule files and the tests also read private names of `derive`, `strided_views` and
`outcomes`; those are not part of what the package offers. `rules/` and `train_step` are imported from outside the
package too and are not listed here.
