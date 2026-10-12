# One-stage audit runs

The one-stage audit (`verity.core.protocols.sampled_proofs.one_stage`) end to end: serving's committed values through registration, the verifier's
draw, C-Flock's circuit prover (M0) and the Lean verifier, to the audit record with its integrity profile, then the
negatives.

- `a<N>.py`: the driver of run A<N>; its docstring says what the run adds and how to call it.
- `compose_circuit.py`, `stage_subset.py`, `stage_tampered.py`: helpers that run inside M0's tree (its `verity_flock`).
- `pod.sh`: the entry point on a CPU pod.
- `a5.sh`: run A5's entry point on a CPU machine. A5 is a served Commit's own window, not a stand-in: the CPU RoPE d64 Commit
  fixture, its window from `verity-vllm serving-rows`, and M0 and Lean built from the run's tree.

C-Flock's circuit prover (M0, `verity_flock.circuit`) is on `main`, but a run does not take it from its checkout:
`pod.sh` takes `git archive`s of M0's tree and the Lean verifier's tree at pinned commits, and a captured input set, as
inputs sent with the run; it builds `flock-circuit` and `flock-verify` once per pod, then runs the driver, whose
`--m0-pythonpath` names that M0 tree. The drivers' checks go through the verifier of record
(`one_stage.verifier`), which builds the sampled-proofs Lean in the run's tree on its first check, hence `--cwd clone`:

~~~sh
research run --on POD --source . --cwd clone --send m0.tar --send lean.tar --send set.tar -- bash benchmarks/one_stage/pod.sh
~~~

`check` runs nothing here.
