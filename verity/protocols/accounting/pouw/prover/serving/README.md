# verity.protocols.accounting.pouw.prover.serving

PoUW's serving side: what runs beside a served model to commit its matrix multiplies. There is one directory a scheme:
`h1/` (H1, a package with its own README), `lead/` (the lead) and `pearl_c/` (Pearl-C on the device), with `ncp.py`
for ``ncp-v1`` and ``ncp-v2``. Beside them are the calls' operands (`operands`), the device libraries (`device`,
`native`, `row_table_kernels`), a call's circuit records (`circuit`), H1's credit gate (`credit_gate`) and the int8 arm
a workload hands in (`int8_arm`). The package's docstring states the two calls, and each module's docstring says what it
holds.

## Interface

- **Provides** a workload's adapter the names in the package's `__all__`, and only those; an adapter imports no deeper
  module. There are two calls per matrix multiply:
  - `run_matmul(run, site, x, weight, bias=None)` runs one under PoUW, served and committed by ``run``, a `MatmulRun`:
    `NcpRun`, `NcpCircuitRun`, `PearlCDeviceRun` or `LeadRun` (whose ``x`` is a `QuantizedRows`).
  - `observe_matmul(run, site, x, weight, y, *, x_scale=None, weight_scale=None, z=None)` records one the workload
    served itself, by ``run``, a `MatmulObserver` (`H1Run`), after the workload has handed H1 its int8 arm
    (`use_int8_arm`).

  ``site`` is the linear's ordinal in the forward pass. Beside the calls are the names a serve loop needs to manage a
  window (its coins, salt session, gate, draw and outcome), and the schemes' names, settings and refusals. The package
  defines the calls, their types and four wrappers that look up their targets at each call (`in_process_coins`,
  `verifier_coins`, `lead_in_process_certifier`, `ready_verifier`). Every other exported name is read from its module
  at each lookup, never cached, so importing the package loads no torch, and a function that imports a name sees its
  module's current attribute.
- **Loaded by:** the vLLM adapter (`verity_vllm.protocols.pouw` and its `pipeline`, `acquire`, `program` and `commit`
  packages); PoUW's prover tests (`verity/protocols/accounting/pouw/prover/tests/`), SPOT's prove path, and PoUW's
  `kernels/served_zk/`; and the benchmarks under `benchmarks/pouw/`, `benchmarks/one_stage/`, `benchmarks/harness/` and
  `benchmarks/hash/`, which import its modules directly.
- **Loads:** `verity.circuits`, `verity.commitments`, `verity.errors`, the rest of `verity.protocols.accounting.pouw`
  and SPOT (`verity.protocols.proofs.spot`); numpy; torch, at import in `h1.receipts` and inside functions elsewhere;
  Triton, in `pearl_c.screen_kernels`; and `verity_flock`, inside the functions that use it. It loads nothing under
  `adapters/` and no vLLM. Where it needs the workload, it asks: `int8_arm.arm()` raises
  `NotImplementedError("compute-accounting/pouw-serving-out-of-adapter")` until the workload hands in its arm.
