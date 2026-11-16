# verity.circuits.ops: design

What someone working on these Definitions needs beyond the [README](README.md) and each module's docstring.

## Beside the modules

Beside the modules are each Definition module's binding file and pins (`<module>.circuit-check.py`,
`<module>.circuit-check.json`), the store ids a module's conformance texts cite (`<module>_evidence.json`), and
`metadata.json`, the names and formats of these Definitions' parts, outside every digest (`verity.circuits.metadata`).

## Interface

- **Provides:** the primitive circuits' Definitions, one family a module, their Boolean versions (`ops.boolean`), and each
  module's sidecar and pins, which `circuit-check` reads.
- **Loaded by:** the served workloads' Definitions (`verity_vllm.definitions`); `verity.commitments.periods`, whose
  `interface` imports every module; PoUW's prover (`verity.protocols.resource_accounting.compute.pouw.prover`); Flock's prover (`verity_flock`), pod
  scripts and agreement with upstream's verifier (`verity/protocols/proofs/flock`); the script that writes the difftest
  vectors of the inference-only policy's check
  (`verity/protocols/workload_classification/training_detection/inference_only/verifier/lean/scripts/difftest_vectors.py`); `circuit-check`, which imports each
  module that has a sidecar; the vLLM integration (`verity_vllm`); and benchmarks. Tests across the tree import it too,
  verifiers' tests among them. No verifier loads it.
- **Loads:** `verity.circuits` (its types, its Boolean forms, its census and its library), `verity.circuits.silicon`,
  and numpy, and `verity.circuits.kernels.tc`, whose batch kernels `gemm`, `moe` and
  `prims` register.
