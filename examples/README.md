# Examples

One directory per example (Daniel, 25 Oct). An example defines only its frontend: the workload it serves, where each party
runs, and the commands a person runs to serve and audit it. Everything else is the integration it attaches through
(`integrations/`) and the protocols, as core defines them (`verity/core/`). An example is not a package: its scripts run by
path from the tree root, and nothing imports `examples/`.

- `vllm/`: Llama-3.2-1B served by vLLM on one GPU through `verity_vllm`'s committing endpoint, and each Commit window audited
  on the prove path.
