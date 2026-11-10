# The network certifier's prover side (`verity.ml.protocols.certifier`)

## Interface

The network certifier's Boolean Definitions that its verifier doesn't read: advice's validity check, the block-pool
cache step and the compile slices built on it, the whole window's charge, the served rows' statics and tie, the
ingress-ids tie, and the scan's ingress and egress. The Definitions the verifier's service, tie service and served
rows read are in `verity/core/protocols/certifier`, with the ones they build on.

- **Python**: `verity.ml.protocols.certifier`. Its importers, each with the modules it imports:
  - `integrations/vllm`: `ingress_scan`, `window_charge`;
  - `benchmarks/network_traces`: `ingress_scan`, `egress`;
  - `benchmarks/prove_path`: `ingress_scan`, `served_statics`, `served_tie`;
  - `benchmarks/network_e2e`: `window_charge`, `cache_step_v2`, `cache_step_v3`, `compile_slice_v2`,
    `compile_slice_v4`;
  - `benchmarks/mempool`: `cache_step_v3`, `compile_slice_v6`;
  - `verity/core/protocols/certifier`: its `lean/scripts/window_vectors.py`, `window_charge`; and its tests, every
    module, and they alone `advice_check`, `io_ids`, `compile_slice_v3` and `compile_slice_v5`. No module of that
    package imports this one.
- **Commands**: none. No console script, no `__main__`, no Lean.
- **Circuit-check**: every module has a sidecar, `<module>.circuit-check.py`, with its pins in
  `<module>.circuit-check.json`, which `circuit-check` (`tools/verity/circuit_check`) reads.
- **Documents**: none. No module writes a file or defines a schema.
- **Loads from**: `verity.core.protocols.certifier` (the schedule and records the Definitions are stated over, and
  the Definitions they build on), `verity.core.protocols.spot`, `verity.ml.circuits` and `verity.ml.commitments`.
- **Tests**: not here. They are in `verity/core/protocols/certifier/tests`, with the verifier's.
