# torch_loop

A minimal serving loop that is not vLLM: a two-layer W8A8 MLP in plain PyTorch on CPU (`verity_torch_loop/loop.py`),
serving a few requests for a few decode steps and recording every linear's output coordinates into a window of Verity's
interface (`verity.ml.windows.interface`). Its one template is `ScaledMmS8Coordinate_v1{K=64, DOT=BlackwellS8ImmaDot32_v1}`
from `verity.ml.circuits.ops.gemm`: the loop builds nothing of its own into the circuit, and imports nothing from
`integrations/vllm`. Its circuit is fixed by its shape, so it registers no compiler or advice.

- `verity_torch_loop/serve.py`: the window written for the audit (`--out DIR`, `--tamper` for a loop whose outputs are not
  its model's), or the circuit's SHA-512 from the loop's shape alone (`--circuit-only`), the one the verifier registers.
- `tests/`: the loop's outputs are the Definition's, unit by unit; core's registration check passes the written window and
  refuses a port under another domain; the verifier's window from the records (`interface.of_records`) is the loop's; a
  tampering loop's outputs are not its Definition's; the loop loads no vLLM.
- End to end: `benchmarks/prove_path/run.sh` with `PP_TORCH_LOOP=1` audits the honest and the tampered window from their
  records alone (`records_audit.py audit --circuit SHA512`): live coins, C-Flock `--zk` behind the Lean firewall, the Lean
  verifier of record, the tampered window refused.
