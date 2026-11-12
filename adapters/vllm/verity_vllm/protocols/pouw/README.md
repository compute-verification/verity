# verity_vllm.protocols.pouw

PoUW's adapter on vLLM: the executor at each linear layer, with the `ncp-v1` and `ncp-v2` executors, Pearl-C on the
device (`pearl_c`) and the lead (`lead`). The package's docstring states the adapter, and each module's docstring says
what the module holds. H1 (`h1`) is a package of its own inside this one and states its own interface.

## Interface

- **Provides** these names to code outside the package, each in its module's `__all__`. The package is PoUW's adapter
  module (`verity_vllm.protocols.interface`): the selector reads `NAME`, `EXECUTES`, `schemes`, `executes`, `refusals`,
  `requires_build`, `keeps_weight_copy`, `commit_weights`, `install`, `fills` and `from_engine` from it by name, and
  other code imports `Settings`, `challenged`, `circuit_run`, `h1_run`, `linear_types` and `quantize`. `circuit` gives
  `C`, `CallRecord`, `ROW_BIAS_FAMILY`, `ROW_FAMILY`, `Read`, `RowsDigest`, `UNIT_TAG`, `WeightRecord`, `call_leaf`,
  `calls_root`, `inputs_digest`, `program`, `sides`, `unit_leaf`, `units_root` and `weight_inputs_digest`, and `native`
  gives `BLOCK`, `Native`, `NativeWeight`, `commit_call` and `dequant`, both to `ncp-v2`'s replay
  (`check.replay.pouw_circuit`). `credit_gate` gives `RECORDS`, `Trace`, `UNTRACED`, `reasons` and `verdict`, and
  `device` gives `nvcc`, both to H1. `pearl_c` gives `CALL_CAPACITY`, `CALL_STEPS`, `EPOCH_STEPS`, `FLAG_ROWS`,
  `GRAPH_ROWS`, `JOB_STEPS`, `Kernel`, `LEAF_CAPACITY`, `PearlCDeviceRun`, `PearlCExecutor`, `ROW_ALIGN`, `SCHEDULES`,
  `SCHEMES`, `SERIAL`, `call_index`, `call_keys`, `epoch_of`, `excluded`, `manifest_epoch`, `manifest_schema`,
  `new_salt_key`, `refusals`, `rotated`, `schedule_of`, `scheme_of`, `screen_into`, `splits`, `weight_roots` and
  `write_manifest`. Of its modules, `grouped` gives `ROTATED`, `gemm_grouped`, `group` and `module`, `manifest` gives
  `manifest_epoch` and `session_head`, `screen` gives `launch`, `scatter_into` and `screen_into`, `whole` gives
  `WholeStep`, and `graphs` gives `GRAPH_ROWS`, which is also `pearl_c`'s. Not part of the interface, and read today
  all the same: the package's `_named`, by the adapter's `acquire` and `pipeline` packages and H1; and, by the
  benchmarks under `benchmarks/pouw/`, `pearl_c`'s `_Keys`, `_Uploads`, `_bytes`, `_constant_keys`, `_constants`,
  `_pipeline`, `_steps`, `_write` and `_zeros` and `pearl_c.host`'s `_Driver`.
- **Loaded by:** the adapter's selector (`verity_vllm.protocols`), its `pipeline`, `acquire` and `check.replay` packages
  and H1; and the benchmarks under `benchmarks/pouw/`, `benchmarks/harness/` and `benchmarks/one_stage/`.
- **Loads:** `verity.protocols.pouw.verifier`, `verity.protocols.pouw.prover`, `verity.protocols.spot.device`,
  `verity.commitments`, `verity.circuits` and `verity.definitions`; the adapter's `protocols.interface` and
  `query.manifest.format`, and H1 with its `settings`, `window`, `hooks` and `requests`; numpy at import; torch and vLLM
  inside the functions that use them; and Triton in `pearl_c.screen_kernels`, which `pearl_c.screen` loads inside its
  functions.
