# torch_loop

A minimal serving stack that is not vLLM: a two-layer W8A8 MLP in plain PyTorch on CPU (`verity_torch_loop/loop.py`),
serving a few requests for a few decode steps and recording every linear's output coordinates into a window of Verity's
stack-neutral serving interface (`verity.ml.serving.interface`). Its one template is
`ScaledMmS8Coordinate_v1{K=64, DOT=BlackwellS8ImmaDot32_v1}` from `verity.ml.circuits.ops.gemm`: the loop builds nothing of
its own into the circuit, and imports nothing from `integrations/vllm`.

- `tests/`: the loop's outputs are the Definition's, unit by unit; the written window's registration passes core's
  `registration.check`, and refuses a changed root; a tampering loop's outputs are not.
- `benchmarks/one_stage/a5_torch_loop.py`: the loop end to end through the proof service, `a5.audit` (live coins, C-Flock
  `--zk` proofs behind the firewall, the Lean verifier of record), honest and tampered. `a5_tp.sh` runs it as
  `STANDIN=torch-loop` and `STANDIN=torch-loop:tamper`.
