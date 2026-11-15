# verity.protocols.resource_accounting.compute.pouw.prover.serving

PoUW's serving side: what runs beside a served model to commit its matrix multiplies. There is one directory a scheme:
`h1/` (H1, a package with its own README), `lead/` (the lead) and `pearl_c/` (Pearl-C's schemes, schedule, manifest,
served y and window), with `ncp.py` for ``ncp-v1`` and ``ncp-v2``. Beside them are the calls' operands (`operands`), the
device libraries (`device`, `native`, `row_table_kernels`), a call's circuit records (`circuit`), H1's credit gate
(`credit_gate`), and what a workload hands in: its int8 arm (`int8_arm`), and its tensors and H1's device (`tensors`).
The package's docstring states the two calls, and each module's docstring says what it holds.

The workload's device code is not here: Pearl-C's run on the device, H1's receipts and the device halves of H1's rows,
tables, records and trace, and the lead's words on the device are the device adapter's, which hands its
implementations in when it is imported.

## Interface

- **Provides** a workload's adapter the names in the package's `__all__`, and only those; an adapter imports no deeper
  module. There are two calls per matrix multiply:
  - `run_matmul(run, site, x, weight, bias=None)` runs one under PoUW, served and committed by ``run``, a `MatmulRun`:
    `NcpRun`, `NcpCircuitRun` or `LeadRun` (whose ``x`` is a `QuantizedRows`), or a workload's own (Pearl-C's,
    the device adapter's `PearlCDeviceRun`).
  - `observe_matmul(run, site, x, weight, y, *, x_scale=None, weight_scale=None, z=None)` records one the workload
    served itself, by ``run``, a `MatmulObserver` (`H1Run`), after the workload has handed H1 its int8 arm
    (`use_int8_arm`).

  ``site`` is the linear's ordinal in the forward pass. Beside the calls are the names a serve loop needs to manage a
  window (its coins, salt session, gate, draw and outcome), the schemes' names, settings and refusals, the hand-ins
  (`use_int8_arm`, `use_tensors`, `use_h1_device`), and the row-table kernels' build and table helpers the adapter's
  step-root and history tables run (`row_table_nvcc`, `row_table_host_lib`, `hm_tables`,
  `leaf_prefixes`, `table_width`, `u64_words`, …). The package defines the calls, their types and four wrappers that
  look up their targets at each call (`in_process_coins`, `verifier_coins`, `lead_in_process_certifier`,
  `ready_verifier`). Every other exported name is read from its module at each lookup, never cached, and a function
  that imports a name sees its module's current attribute.
- **Loaded by:** the serving engine's adapter (its PoUW protocol package and its `pipeline`, `acquire`, `program` and
  `commit` packages) and the device adapter; PoUW's prover tests
  (`tests/verity/protocols/resource_accounting/compute/pouw/prover/`), SPOT's prove path, and PoUW's `kernels/served_zk/`; and the
  benchmarks under `benchmarks/pouw/`, `benchmarks/one_stage/`, `benchmarks/harness/`, `benchmarks/hash/`,
  `benchmarks/commitments/` and `benchmarks/serve_tax/`, which import its modules directly.
- **Loads:** `verity.circuits`, `verity.commitments`, `verity.errors`, the rest of `verity.protocols.resource_accounting.compute.pouw`
  and SPOT (`verity.protocols.proofs.spot`); numpy; and `verity_flock`, inside the functions that use it. It loads no
  tensor library, nothing under `adapters/` and no serving engine. Where it needs the workload, it asks: `int8_arm.arm()` raises
  `NotImplementedError("compute-accounting/pouw-serving-out-of-adapter")` until the workload hands in its arm, and
  `tensors.tensors()` and `tensors.h1_device()` raise `NotImplementedError("compute-accounting/architecture-on-branch")`
  until it hands in its tensors and H1's device (the device adapter does, at import). Without them the host paths run:
  numpy buffers, H1's Python reference tables, and no credit-gate trace (`credit_gate.outside` is a no-op).
