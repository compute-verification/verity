# Examples

One directory per example (Daniel, 25 Oct). An example holds what is its own: the workload it serves, its placement (where
each party runs), and the commands a person runs to serve and audit it. What attaches Verity to the serving stack is the
integration's (`integrations/`), and the protocols it runs stay in core (`verity/core/`). `verity` imports nothing from
`examples/` (`tests/test_boundaries.py`), and `verity example NAME` runs `examples/<NAME>/`'s `run.py` or `run.sh`.

- `vllm/`: Llama-3.2-1B served by vLLM on one GPU through `verity_vllm`'s committing endpoint, a served window audited
  through every subprotocol. Its row is in its own `workloads/`, its placement is `placement.toml`, which `run.sh` reads, and
  its scripts reach the integration by `PYTHONPATH`; it is not a package.
- `one_stage/`: sampled proofs' one-stage audit end to end, run by run (A0 to A11), on one CPU pod (`pod.sh`). It takes
  vLLM's committed values through registration, the verifier's draw, C-Flock's circuit prover and the Lean verifier, to
  the audit record and its negatives. Its cost jobs (`zk_cost*`, `a8_cost.py`, `a8_gpu.sh`) are measurement. They are
  here while they import or run its drivers.
- `private_circuit/`: protocol 2, a random hidden netlist run as a value of the public universal unit
  (`verity.ml.private_circuits.universal`), proved by C-Flock under `--zk` and verified by the Lean verifier on one CPU node
  (`pod.sh`).
