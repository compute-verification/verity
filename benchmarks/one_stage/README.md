# One-stage audit runs

The one-stage audit (`verity.core.protocols.sampled_proofs.one_stage`) end to end: serving's committed values through registration, the verifier's
draw, C-Flock's circuit prover (M0) and the Lean verifier, to the audit record with its integrity profile, then the
negatives.

- `a<N>.py`: the driver of run A<N>; its docstring says what the run adds and how to call it.
- `compose_circuit.py`, `stage_subset.py`, `stage_tampered.py`: helpers that run inside M0's tree (its `verity_flock`).
- `pod.sh`: the entry point on a CPU pod.
- `a5.sh`: run A5's entry point on a CPU machine. A5 is a deferred Commit's window, a stand-in linked to its vllm-v1 run root
  and replay bundle: the CPU RoPE d64 Commit fixture, its window from `verity-vllm serving-rows`, and M0 and Lean built from the
  run's tree.
- `a5_tp.sh`: A5 over a TP Commit's windows, one `served` window per rank (`a5.py --tp-commit`): GLM-5.3's TP8 Commit with
  `SERVING_ROWS=benchmarks/one_stage/glm53_silu_mul_v2_tp8.json` (its SiluMul_v2 partition: every unit of `SiluMul_v2{I=1536}` and
  `SiluMul_v2{I=256}`, the two widths a TP8 rank serves, and its law), or by default the CPU TP stand-in
  (`integrations/vllm/tests/pipeline/served_tp.py`).

C-Flock's circuit prover (M0, `verity_flock.circuit`) is on `main`, but a run does not take it from its checkout:
`pod.sh` takes `git archive`s of M0's tree and the Lean verifier's tree at pinned commits, and a captured input set, as
inputs sent with the run; it builds `flock-circuit` and `flock-verify` once per pod, then runs the driver, whose
`--m0-pythonpath` names that M0 tree. The drivers' checks go through the verifier of record
(`one_stage.verifier`), which builds the sampled-proofs Lean in the run's tree on its first check, hence `--cwd clone`:

~~~sh
research run --on POD --source . --cwd clone --send m0.tar --send lean.tar --send set.tar -- bash benchmarks/one_stage/pod.sh
~~~

`check` runs nothing here.
