# The network attester's prover side (`verity.protocols.resource_accounting.network.prover`)

## Interface

The network attester's Boolean Definitions that its verifier doesn't read: the whole window's charge, the ingress-ids
tie, the scan's ingress and egress, the lengths rows' proof block and the response tie's. The Definitions the
verifier's service and tie service read are in `verity/protocols/resource_accounting/network/verifier`, with the ones they
build on. Advice's validity check, the block-pool cache step and the compile slices built on it, and the served rows'
statics and tie are vLLM's compiler's (`adapters/vllm/verity_vllm/compiler`).

- **Python**: `verity.protocols.resource_accounting.network.prover`. Its importers, each with the modules it imports:
  - `adapters/vllm`: `ingress_scan`, `window_charge`, `workload`;
  - `benchmarks/network_traces`: `block_certifier`, `egress_zk`, `ingress_scan`, `leak_scan`, `lengths_row`, `n2_zk`,
    `network_attester_zk`, `proof_drivers`, `released_keys`, `scan_period`, `scan_zk`, `stop_verdicts`, `window_chain`,
    `window_charge`;
  - `benchmarks/network_e2e`: `window_chain`, `window_charge`;
  - `benchmarks/one_stage`: `leak_scan`, `n2_zk`, `network_attester_zk`, `scan_zk`;
  - `benchmarks/prove_path`: `leak_scan`, `n2_zk`;
  - `benchmarks/sampling_plan`: `response_tie`;
  - `tools/verity`: `ingress_scan`;
  - `verity/protocols/resource_accounting/network/attester`: its `program/lean/scripts/window_vectors.py`,
    `window_charge`;
  - `verity/protocols/compose/verifier/live_challenger`: its tests, `released_keys` and `scan_zk`;
  - `verity/protocols/resource_accounting/network/verifier`: its tests, `egress`, `ingress_scan`, `io_ids`, `lengths_row`,
    `response_tie`, `window_charge` and `workload`. No module of that package imports this one.
- **Commands**: no console script and no Lean. The drivers `egress_zk`, `n2_zk`, `n2_tie_zk`, `network_attester_zk`,
  `released_keys`, `scan_period` and `scan_zk` each have a `__main__`, which `benchmarks/network_traces`'s jobs run.
- **The period check's evidence**: `period.verified_answer`, each answer the period check reads with its one-stage
  inputs; an interface, its body not written.
- **Circuit-check**: each module holding Boolean Definitions (`egress`, `ingress_scan`, `io_ids`, `lengths_row`,
  `response_tie`, `window_charge`) has a sidecar, `<module>.circuit-check.py`, with its pins in
  `<module>.circuit-check.json`, which `circuit-check` (`tools/verity/circuit_check`) reads.
- **Documents**: none. No module writes a file or defines a schema.
- **Loads from**: `verity.protocols.resource_accounting.network.verifier` (the schedule and records the Definitions are stated over, and
  the Definitions they build on), `verity.protocols.proofs.spot.verifier`, `verity.circuits` and `verity.commitments`.
- **Tests**: the drivers' are in `tests/`, run with no proof drivers installed; the Definitions' are in
  `verity/protocols/resource_accounting/network/verifier/tests`, with the verifier's.
