# The network certifier's prover side (`verity.protocols.accounting.certifier.prover`)

## Interface

The network certifier's Boolean Definitions that its verifier doesn't read: the whole window's charge, the ingress-ids
tie, the scan's ingress and egress, the lengths rows' proof block and the response tie's. The Definitions the
verifier's service and tie service read are in `verity/protocols/accounting/certifier/verifier`, with the ones they
build on. Advice's validity check, the block-pool cache step and the compile slices built on it, and the served rows'
statics and tie are vLLM's compiler's (`adapters/vllm/verity_vllm/compiler`).

- **Python**: `verity.protocols.accounting.certifier.prover`. Its importers, each with the modules it imports:
  - `adapters/vllm`: `ingress_scan`, `scan_zk`, `window_charge`, `workload`;
  - `benchmarks/network_traces`: `block_certifier`, `egress_zk`, `ingress_scan`, `leak_scan`, `lengths_row`, `n2_zk`,
    `network_certifier_zk`, `proof_drivers`, `scan_period`, `scan_zk`, `stop_verdicts`, `window_chain`,
    `window_charge`;
  - `benchmarks/network_e2e`: `window_chain`, `window_charge`;
  - `benchmarks/one_stage`: `leak_scan`, `n2_zk`, `network_certifier_zk`, `scan_zk`;
  - `benchmarks/prove_path`: `leak_scan`, `n2_zk`;
  - `benchmarks/sampling_plan`: `response_tie`;
  - `tools/verity`: `ingress_scan`;
  - `verity/protocols/accounting/certifier/device`: `scan_zk`, `stop_verdicts`, `window_charge` (and its
    `program/lean/scripts/window_vectors.py`, `window_charge`);
  - `verity/protocols/compose/device/live_challenger`: `scan_zk`;
  - `verity/protocols/accounting/certifier/verifier`: its tests, `egress`, `ingress_scan`, `io_ids`, `lengths_row`,
    `response_tie`, `window_charge` and `workload`. No module of that package imports this one.
- **Commands**: none. No console script, no `__main__`, no Lean.
- **Period evidence**: `period.verified_answer`, each answer the period check reads with its one-stage inputs; an
  interface, its body not written.
- **Circuit-check**: every module but `period` has a sidecar, `<module>.circuit-check.py`, with its pins in
  `<module>.circuit-check.json`, which `circuit-check` (`tools/verity/circuit_check`) reads.
- **Documents**: none. No module writes a file or defines a schema.
- **Loads from**: `verity.protocols.accounting.certifier.verifier` (the schedule and records the Definitions are stated over, and
  the Definitions they build on), `verity.protocols.proofs.spot.verifier`, `verity.circuits` and `verity.commitments`.
- **Tests**: not here. They are in `tests/verity/protocols/accounting/certifier/verifier`, with the verifier's.
