# benchmarks/remote_session

The session harness of PoUB (proof of useful bandwidth, `verity/core/protocols/remote_bandwidth`), not the remote
protocol's demo, which is `verity example vllm`. It runs PoUB end to end: a verifier's process and a prover's, over TCP,
and the memory line a gate run publishes.

**Status: active.** The vLLM integration imports it (`integrations/vllm/verity_vllm/protocols/remote_bandwidth/remote.py`
and `in_place.py`, which `tests/test_boundaries.py` lists), the audit's Lean verifier names `verifier.py` as its driver and
`auditor_line.py` as its line of record, and its suite runs in `check`. Decided by `rg remote_session` over `verity/`,
`benchmarks/`, `examples/`, `tests/` and `integrations/`.

**Run:** `uv run tools/verity/check/suites.py benchmarks/remote_session` runs both processes over a loopback, the driver's
reference (`CpuBackend`) as the prover's backend, and the Lean line test where `lake` is. `run.sh` is the GPU run, one
side a machine: `ROLE=verifier` on a machine the prover doesn't control and `ROLE=prover` on the prover's, with no
default (`ROLE=both` runs the two on one machine, a test of the scripts that audits nothing).

- `verifier.py`: the verifier's process. It runs the Lean verifier of record, `remote-audit-verifier` (`lean_verifier.py`),
  beside `audit.Session`, and refuses a run where the two differ.
- `prover.py`: the prover's process; its GPU backend is `device.py` with `session.cu` (which includes
  `benchmarks/codec_tensor/reader.cu`), `inplace.py` reads vLLM's tensors in place, and `serve.py` serves vLLM as the prover.
- `auditor_line.py`: the memory line on the auditor's clock, computed by the Lean (`line_record`) and held equal to `line.py`'s.

**Reads and writes:** a store (synthetic, or a pinned checkpoint `store_layout.py` writes) and the prover's plan (`PLAN`:
`honest`, or `skip:F`, a fraction F of the chunks dropped); it writes `verifier.json`, `prover.json` and their logs, which
`summary.py`, `sampling_inputs.py` and `gate_whatif.py` read. **Backs:** its line reads the conclusions of
`Accounting.Guarantees.Remote.HeldOverTime` and `HeldOverTimeLinks`, and `Properties.RemoteAudit.Line`'s `LineHeld` and
`LineHeldIdentity` state that line holds, with the named assumption `NoOutsideCompute` among their premises.
