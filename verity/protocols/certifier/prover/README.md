# The network certifier's prover side (`verity.protocols.certifier.prover`)

## Interface

The network certifier's Boolean Definitions that its verifier doesn't read: advice's validity check, the block-pool
cache step and the compile slices built on it, the whole window's charge, the served rows' statics and tie, the
ingress-ids tie, the scan's ingress and egress, the lengths rows' proof block and the response tie's. The
Definitions the verifier's service, tie service and served rows read are in `verity/protocols/certifier/verifier`,
with the ones they build on.

- **Python**: `verity.protocols.certifier.prover`. Its importers, each with the modules it imports:
  - `adapters/vllm`: `ingress_scan`, `window_charge`;
  - `benchmarks/network_traces`: `ingress_scan`, `egress`;
  - `benchmarks/prove_path`: `ingress_scan`, `served_statics`, `served_tie`;
  - `benchmarks/network_e2e`: `window_charge`, `cache_step_v2`, `cache_step_v3`, `compile_slice_v2`,
    `compile_slice_v4`, `compile_slice_v7`;
  - `benchmarks/mempool`: `cache_step_v3`, `compile_slice_v6`;
  - `verity/protocols/certifier/device`: its `program/lean/scripts/window_vectors.py`, `window_charge`;
  - `verity/protocols/certifier/verifier`: its tests, every module, and they alone `advice_check`, `io_ids`,
    `compile_slice_v3` and `compile_slice_v5`. No module of that package imports this one.
- **Commands**: none. No console script, no `__main__`, no Lean.
- **Circuit-check**: every module has a sidecar, `<module>.circuit-check.py`, with its pins in
  `<module>.circuit-check.json`, which `circuit-check` (`tools/verity/circuit_check`) reads.
- **Documents**: none. No module writes a file or defines a schema.
- **Loads from**: `verity.protocols.certifier.verifier` (the schedule and records the Definitions are stated over, and
  the Definitions they build on), `verity.protocols.spot.verifier`, `verity.circuits`, `verity.commitments` and, for
  `memcheck` and `block_key`, `verity.definitions`.
- **Tests**: not here. They are in `verity/protocols/certifier/verifier/tests`, with the verifier's.
