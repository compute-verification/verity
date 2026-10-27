# Examples

One directory per example (Daniel, 25 Oct). An example holds what is its own: the workload it serves, its placement (where
each party runs), and the commands a person runs to serve and audit it. What attaches Verity to the serving stack is the
integration's (`integrations/`), and the protocols it runs stay in core (`verity/core/`). `verity` imports nothing from
`examples/` (`tests/test_boundaries.py`), and `verity example NAME` runs `examples/<NAME>/`'s `run.py` or `run.sh`.

- `h200/`: the one-H200 walking skeleton (CMP-96), the frozen MQA shape served on one GPU node with every op's output
  committed a step, and onsite's legs in skeleton form: drawn units replayed and drawn slots opened, PoUS's spot reads and
  wipe, PoUW's per-call tiles. `interfaces/` is its subprotocols' replay bundle and block record, a suite of its own.
- `pouw_window/`: one PoUW window on stand-in int8 values through today's spot check, the honest window and every attack,
  on one CPU machine.
- `toy_onsite/`: onsite on the torch loop's toy (`integrations/torch_loop`), every window audited by every protocol's leg
  and composed, on one CPU machine. Its README has S1's run of record.
- `vllm/`: Llama-3.2-1B served by vLLM on one GPU through `verity_vllm`'s committing endpoint, each closed Commit window
  audited on the prove path. Its scripts reach the integration by `PYTHONPATH`; it is not a package.
