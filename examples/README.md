# Examples

One directory per example (Daniel, 25 Oct). An example holds what is its own: the workload it serves, its placement (where
each party runs), and the commands a person runs to serve and audit it. What attaches Verity to the serving stack is the
integration's (`integrations/`), and the protocols it runs stay in core (`verity/core/`). `verity` imports nothing from
`examples/` (`tests/test_boundaries.py`), and `verity example NAME` runs `examples/<NAME>/`'s `run.py` or `run.sh`.

- `network_sim/`: a mock datacenter of Linux network namespaces on one host, each block behind the real network
  certifier, its records audited, with N3 and N2 on the workload it serves (verity_vllm's serving Program, or an MLP).
  network-accounting's.
- `network_vllm/`: network accounting end to end on a model vLLM serves: a certifier-serve rehearsal's runs through N2,
  the head checks proved behind the Lean firewall, and the negatives, in one report. network-accounting's.
- `vllm/`: Llama-3.2-1B served by vLLM on one GPU through `verity_vllm`'s committing endpoint, each closed Commit window
  audited on the prove path. Its scripts reach the integration by `PYTHONPATH`; it is not a package.
