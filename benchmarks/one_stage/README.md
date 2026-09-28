# One-stage audit runs

The one-stage audit (`protocols/one_stage`) end to end: serving's committed values through registration, the verifier's
draw, C-Flock's circuit prover (M0) and the Lean verifier, to the audit record with its integrity profile, then the
negatives.

- `a<N>.py`: the driver of run A<N>; its docstring says what the run adds and how to call it.
- `compose_circuit.py`, `stage_subset.py`, `stage_tampered.py`: helpers that run inside M0's tree (its `verity_flock`).
- `pod.sh`: the entry point on a CPU pod.

The runs need more than `main`: C-Flock's circuit prover (M0, `verity_flock.circuit`) is on draft PR #83. `pod.sh` takes
`git archive`s of M0's tree and the Lean verifier's tree at pinned commits, and a captured input set, as inputs sent with
the run; it builds `flock-circuit` and `flock-verify` once per pod, then runs the driver:

~~~sh
research run --on POD --source . --cwd source --send m0.tar --send lean.tar --send set.tar -- bash benchmarks/one_stage/pod.sh
~~~

`check` runs nothing here.
