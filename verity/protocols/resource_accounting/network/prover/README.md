# The network attester's prover side (`verity.protocols.resource_accounting.network.prover`)

## Interface

The network attester's Boolean Definitions that its verifier doesn't read: the whole window's charge, the ingress-ids
tie, the scan's ingress and egress, the lengths rows' proof block and the response tie's. The Definitions the
verifier's service and tie service read are in `verity/protocols/resource_accounting/network/verifier`, with the ones they
build on. Advice's validity check, the block-pool cache step and the compile slices built on it, and the served rows'
statics and tie are vLLM's compiler's (`adapters/vllm/verity_vllm/compiler`).

- **Python**: `verity.protocols.resource_accounting.network.prover`. Its importers, each with the modules it imports:
  - `adapters/vllm`: `ingress_scan`, `scan_zk`, `window_charge`, `workload`;
  - `benchmarks/network_traces`: `block_certifier`, `egress_zk`, `ingress_scan`, `leak_scan`, `lengths_row`, `n2_zk`,
    `network_attester_chain`, `network_attester_zk`, `proof_drivers`, `scan_period`, `scan_zk`, `stop_verdicts`,
    `window_chain`, `window_charge`, `wipe_egress`;
  - `benchmarks/network_e2e`: `window_chain`, `window_charge`;
  - `benchmarks/one_stage`: `leak_scan`, `n2_zk`, `network_attester_chain`, `network_attester_zk`, `scan_zk`;
  - `benchmarks/prove_path`: `leak_scan`, `n2_zk`;
  - `benchmarks/sampling_plan`: `response_tie`;
  - `tools/verity`: `ingress_scan`;
  - `verity/protocols/resource_accounting/network/attester`: `scan_zk`, `stop_verdicts`, `window_charge` (and its
    `program/lean/scripts/window_vectors.py`, `window_charge`);
  - `verity/protocols/compose/verifier/live_challenger`: `scan_zk`;
  - `verity/protocols/resource_accounting/network/verifier`: its tests, `egress`, `ingress_scan`, `io_ids`, `lengths_row`,
    `response_tie`, `window_charge` and `workload`. No module of that package imports this one.
- **Commands**: no console script and no Lean. Run by path: `scan_period.py`, `released_keys.py` and `wipe_egress.py`
  (the count of what can leave a node over a memory wipe's session, a `wipe-egress/v1` record); the zero-knowledge
  drivers (`scan_zk`, `egress_zk`, `network_attester_zk`, `n2_zk`, `n2_tie_zk`) through
  `benchmarks/network_traces/zk_drivers.py`.
- **Period evidence**: `period.verified_answer`, each answer the period check reads with its one-stage inputs; an
  interface, its body not written.
- **Circuit-check**: every module but `period` has a sidecar, `<module>.circuit-check.py`, with its pins in
  `<module>.circuit-check.json`, which `circuit-check` (`tools/verity/circuit_check`) reads.
- **Documents**: `wipe_egress.py` writes the `wipe-egress/v1` record, and `network_attester_chain.py` reads the
  auditor's record of a network attester's deployment (`certifier-key/v1`, its `RECORD`).
- **Loads from**: `verity.protocols.resource_accounting.network.verifier` (the schedule and records the Definitions are stated over, and
  the Definitions they build on), `verity.protocols.proofs.spot.verifier`, `verity.circuits` and `verity.commitments`;
  `wipe_egress` reads PoUS's unit of a wipe's egress (`verity.protocols.resource_accounting.memory.pous.verifier`).
- **Tests**: `tests/` here, with `attester_runs.py`, the gate's synthetic network attester run that the chain's tests
  and those of its readers under `benchmarks/network_traces/tests/` build on; the Definitions' are in
  `verity/protocols/resource_accounting/network/verifier/tests`, with the verifier's.
