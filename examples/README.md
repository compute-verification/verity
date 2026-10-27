# Examples

One directory per example (Daniel, 25 Oct). An example holds what is its own: the workload it serves, its placement (where
each party runs), and the commands a person runs to serve and audit it. What attaches Verity to the serving stack is the
integration's (`integrations/`), and the protocols it runs stay in core (`verity/core/`). `verity` imports nothing from
`examples/` (`tests/test_boundaries.py`), and `verity example NAME` runs `examples/<NAME>/`'s `run.py` or `run.sh`.

- `vllm/`: Llama-3.2-1B served by vLLM on one GPU through `verity_vllm`'s committing endpoint, a served window audited
  through every subprotocol. Its row is in its own `workloads/`, its placement is `placement.toml`, which `run.sh` reads, and
  its scripts reach the integration by `PYTHONPATH`; it is not a package.
- `toy_network/`: the toy (`integrations/torch_loop`) audited by sampled proofs, every role from `config.toml` through
  `verity run`, on CPU; the network certifier joins its `protocols` once its role exists.
