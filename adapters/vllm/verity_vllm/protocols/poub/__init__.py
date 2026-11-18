"""PoUB on vLLM: the node's store (`verity.protocols.resource_accounting.memory.poub.prover.node_store`) attached to a vLLM engine.

- The store's binder is PoUB's (`node_store.binder`): it subscribes to the engine's ``kv`` hook
  (`verity_vllm.engine.kv_events`), which names no scheme.
- `remote`: `CommitSession`, the Commit's PoUB session on its engine (``--remote-verifier HOST:PORT``).
"""
