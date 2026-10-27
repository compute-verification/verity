# Examples

One directory per example (Daniel, 25 Oct). An example holds what is its own: the workload it serves, its placement (where
each party runs), and the commands a person runs to serve and audit it. What attaches Verity to the serving stack is the
integration's (`integrations/`), and the protocols it runs stay in core (`verity/core/`). `verity` imports nothing from
`examples/` (`tests/test_boundaries.py`), and `verity example NAME` runs `examples/<NAME>/`'s `run.py` or `run.sh`.

- `vllm/`: Llama-3.2-1B served by vLLM on one GPU through `verity_vllm`'s committing endpoint, each closed Commit window
  audited on the prove path. Its scripts reach the integration by `PYTHONPATH`; it is not a package.
- `wipe/`: PoUS's streamed-label wipe of one GPU's memory, the memory challenger on a host of its own streaming the fill
  and timing its spot reads, the Lean and the Python judging the signed session. Its scripts run by path from the tree
  root; the wipe's GPU timing measurement stays in `benchmarks/pous/streamed_wipe/`.
