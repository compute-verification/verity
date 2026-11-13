# verity_vllm.protocols.pouw

PoUW's adapter on vLLM: the executor at each linear layer, with the `ncp-v1` and `ncp-v2` executors, Pearl-C on the
device (`pearl_c`) and the lead (`lead`). The package's docstring states the adapter, and each module's docstring says
what the module holds. H1 (`h1`) is a package of its own inside this one and states its own interface.

## Interface

- **Provides** these names to code outside the package. Each is in its module's `__all__`, except the package's and
  `pearl_c.host`'s, which state none. The package is PoUW's adapter module (`verity_vllm.protocols.interface`): the
  selector reads `NAME`, `EXECUTES`, `schemes`, `executes`, `refusals`, `requires_build`, `keeps_weight_copy`, `cuts`,
  `commit_weights`, `install`, `fills` and `from_engine` from it by name, and other code imports `Settings`,
  `challenged`, `circuit_run`, `h1_run`, `linear_types` and `quantize`. `circuit` gives `CallRecord`, `ROW_BIAS_FAMILY`,
  `ROW_FAMILY`, `Read`, `RowsDigest`, `UNIT_TAG`, `WeightRecord`, `call_leaf`, `calls_root`, `inputs_digest`, `program`,
  `sides`, `unit_leaf`, `units_root` and `weight_inputs_digest`, and `native` gives `BLOCK`, `Native`, `NativeWeight`,
  `commit_call` and `dequant`, both to `ncp-v2`'s replay (`check.replay.pouw_circuit`). `credit_gate` gives `RECORDS`,
  `Trace`, `UNTRACED`, `outside`, `reasons` and `verdict`, and `device` gives `nvcc`, both to H1. `pearl_c` gives
  `CALL_CAPACITY`, `CALL_STEPS`, `EPOCH_STEPS`, `FLAG_ROWS`, `GRAPH_ROWS`, `JOB_STEPS`, `Kernel`, `LEAF_CAPACITY`,
  `PearlCDeviceRun`, `PearlCExecutor`, `ROW_ALIGN`, `SCHEDULES`, `SCHEMES`, `SERIAL`, `call_index`, `call_keys`,
  `epoch_of`, `excluded`, `manifest_epoch`, `manifest_schema`, `new_salt_key`, `refusals`, `rotated`, `schedule_of`,
  `scheme_of`, `screen_into`, `splits`, `weight_roots` and `write_manifest`. Of its modules, `pearl_c.window` gives
  `KEEP`, `SERVED_SCHEMA`, `UNKNOWN_VERSION`, `VERSION`, `a_value`, `b_value`, `coins_of`, `macs`, `record`, `reuse`,
  `served_passes`, `version_of`, `write` and `y_value`, to the adapter's `pipeline` and `program.predict` packages and
  to the benchmarks. The others give to the benchmarks alone: `pearl_c.served_y` gives `DeviceTrees`, `HM96_Y`,
  `commit_y`, `retain_y`, `y_binding` and `y_domain`; `pearl_c.grouped` gives `ROTATED`, `gemm_grouped`, `group` and
  `module`; `pearl_c.screen` gives `excluded`, `excluded_host`, `launch`, `scatter_into` and `screen_into`;
  `pearl_c.host` gives `joined`; `pearl_c.manifest` gives `manifest_epoch`; `pearl_c.lanes` gives `whole_options`;
  `pearl_c.whole` gives `WholeStep`; and `pearl_c.graphs` gives `GRAPH_ROWS`, which is `pearl_c`'s too. Not part of the
  interface, and read today all the same: the package's `_named`, by the adapter's `acquire` and `pipeline` packages and
  H1; and, by the benchmarks under `benchmarks/pouw/`, `pearl_c`'s `_Keys`, `_Uploads`, `_bytes`, `_constant_keys`,
  `_constants`, `_pipeline`, `_steps`, `_write` and `_zeros` and `pearl_c.host`'s `_Driver`, `_Ring` and `_write_host`.
- **Loaded by:** the adapter's selector (`verity_vllm.protocols`), its `pipeline`, `acquire`, `check.replay` and
  `program.predict` packages and H1; and the benchmarks under `benchmarks/pouw/`, `benchmarks/harness/`,
  `benchmarks/one_stage/` and `benchmarks/prove_path/`.
- **Loads:** `verity.protocols.accounting.pouw.verifier`, `verity.protocols.accounting.pouw.prover`,
  `verity.protocols.proofs.spot.device`, `verity.protocols.proofs.spot.verifier`, `verity.commitments` and
  `verity.circuits`; the adapter's `protocols.interface`, `query.manifest.format`, `commit.lead_rows`,
  `records.pearl_served` and `errors`, and H1 with its `settings`, `window`, `hooks` and `requests`; numpy at import;
  torch and vLLM inside the functions that use them; and Triton, at import in `pearl_c.screen_kernels` and inside
  functions in `pearl_c.screen`, which loads `pearl_c.screen_kernels` there too.
