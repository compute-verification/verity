"""PoUB on vLLM: the node's store (`verity.protocols.resource_accounting.memory.poub.prover.node_store`) attached to a vLLM engine.

- `binder`: installs the store's hooks into a vLLM engine (`verity_vllm.engine.hooks`), all of them undone by
  `uninstall`.
- `remote`: `CommitSession`, the Commit's PoUB session on its engine (``--remote-verifier HOST:PORT``).
"""
