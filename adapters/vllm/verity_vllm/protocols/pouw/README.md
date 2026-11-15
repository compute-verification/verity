# verity_vllm.protocols.pouw

PoUW as a protocol option on vLLM: the option at each linear layer and its hooks on the engine. What runs a matrix
multiply under PoUW, commits it and manages its window is PoUW's serving package
(`verity.protocols.resource_accounting.compute.pouw.prover.serving`, one directory a scheme). This package hands each call to it through
`run_matmul` (``ncp-v1``, ``ncp-v2``, Pearl-C and the lead) or `observe_matmul` (H1), and imports nothing of PoUW
except the names that package exports. The device code is the PyTorch adapter's (`verity_torch.pouw`): Pearl-C's run
(`PearlCDeviceRun`, `pearl_c_weight_roots`, `hook_steps`) and H1's receipts and device tables (`Receipts`,
`h1_tables_available`) come from there, and importing it hands the serving package its tensors and H1's device. The
package's docstring states the option, and each module's docstring says what the module holds:

- the package: the option (`install`, `commit_weights` and the selector's names), the executor at each linear's
  ``apply``, the lead's GEMM on the int8 arm and the arm's registration with H1 (`use_int8_arm`);
- `h1_hooks`: H1's hooks on the served engine: the linears it records, their weights' roots, each linear's observer
  and the model runner's admissions;
- `h1_ranks`: H1 on a multiprocess engine, with each rank's calls recorded in its own worker;
- `h1_settings`: the pouw settings of H1's window and of the auditor device's challenger.

## Interface

- **Provides** these names to code outside the package. The package states no `__all__`. It is PoUW's adapter module
  (`verity_vllm.protocols.interface`): the selector reads `NAME`, `EXECUTES`, `schemes`, `executes`, `refusals`,
  `requires_build`, `keeps_weight_copy`, `cuts`, `commit_weights`, `install`, `fills` and `from_engine` from it by
  name, and other code imports `Settings`, `challenged`, `circuit_run`, `h1_run`, `linear_types`, `ncp_linear_op` and
  `quantize`. `h1_hooks`, `h1_ranks` and `h1_settings` give the names in their modules' `__all__`. Not part of the
  interface, and read today all the same: the package's `_named`, by the adapter's `acquire` and `pipeline` packages.
- **Loaded by:** the adapter's selector (`verity_vllm.protocols`), its `pipeline`, `acquire`, `check.replay` and
  `program.predict` packages; and the benchmarks under `benchmarks/pouw/`, `benchmarks/harness/`,
  `benchmarks/one_stage/` and `benchmarks/prove_path/`.
- **Loads:** `verity.protocols.resource_accounting.compute.pouw.prover.serving` (its exports only), `verity_torch.pouw` (its exports only)
  and `verity.protocols.proofs.spot`;
  the adapter's `protocols` (the selector and `interface`), `engine` and `program.frontend`; and numpy, torch and vLLM
  inside the functions that use them.
