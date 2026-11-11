# benchmarks/remote_session

The session harness of PoUB (`verity/protocols/poub/verifier`), not the remote
protocol's demo, which is `verity demo vllm`. It runs PoUB end to end: a verifier's process and a prover's, over TCP,
and the memory line a gate run publishes.

## Interface

What code outside this directory may use. Everything else here is internal.

| Entry point | What it is |
|---|---|
| `uv run tools/verity/check/suites.py benchmarks/remote_session` | The suite: both processes over a loopback on the CPU, and the Lean line test where `lake` is. It writes no result. |
| `bash benchmarks/remote_session/run.sh`, with `ROLE=verifier` or `ROLE=prover` (`ROLE=both` only tests the scripts on one machine), and `PLAN`, `GIB`, `GPUS` and the sessions' parameters as its header lists them | PoUB sessions on GPUs, the verifier's process and the prover's over TCP. It writes `verifier.json`, `prover.json` and their logs to `$RESEARCH_RUN_DIR/out`. |
| `benchmarks/remote_session/gate_run.sh [--dry-run] PERIOD` | A gate run in one command: the developer's whole node serving a model, the verifier on the verifier's host through its gate, the coins from the challenger's host, then the line of record (`auditor.sh`, which runs `auditor_line.py`) and the what-if beside it (`gate_whatif.sh`). Each step is a recorded job, and their ids go to `~/.research/gate-runs/PERIOD.env`. |
| `verifier_process.py`, with `verifier.py`'s arguments | The verifier's process. It writes `verifier.json` under `--out`. |

Read from outside `benchmarks/`: `verifier_process.py`, run by its path; `prover.py`, loaded from its file; and
`session.cu`'s build.

**Status: active.** The vLLM integration imports it (`adapters/vllm/verity_vllm/protocols/remote_bandwidth/remote.py`,
which `tests/test_boundaries.py` lists), the audit's Lean verifier names `auditor_line.py` as its line of record, and
its suite runs in `check`. Decided by `rg remote_session` over `verity/`, `benchmarks/`, `demos/`, `tests/` and
`adapters/`.

**Run:** `uv run tools/verity/check/suites.py benchmarks/remote_session` runs both processes over a loopback, the driver's
reference (`CpuBackend`) as the prover's backend, and the Lean line test where `lake` is. `run.sh` is the GPU run, one
side a machine: `ROLE=verifier` on a machine the developer doesn't control and `ROLE=prover` on the developer's, with no
default (`ROLE=both` runs the two on one machine, a test of the scripts that audits nothing).

- `verifier_process.py`: the verifier's process, which these scripts and live_verifier's host run:
  `verity/protocols/poub/verifier/session/verifier.py`'s `main`, handed what a verifier imports none of, the LMS
  verify a served or gate session's signatures are checked with and the live challenger's `serve.py`. That verifier
  runs the Lean verifier, `remote-audit-verifier` (`lean_verifier.py` beside it), beside `audit.AuditRun`, and refuses
  a run where the two differ; `wire.py` beside it frames the messages between the two processes.
- `prover.py`: the prover's process; its GPU backend is `device.py` with `session.cu` (which includes
  `benchmarks/codec_tensor/reader.cu`), and `serve.py` serves vLLM as the prover.
- `auditor_line.py`: the memory line on the auditor's clock, computed by the Lean (`line_record`) and held equal to `line.py`'s.

**Reads and writes:** a store (synthetic, or a pinned checkpoint `store_layout.py` writes) and the prover's plan (`PLAN`:
`honest`, or `skip:F`, a fraction F of the chunks dropped); it writes `verifier.json`, `prover.json` and their logs, which
`summary.py`, `sampling_inputs.py` and `gate_whatif.py` read. **Backs:** its line reads the conclusions of
`Accounting.Guarantees.Remote.HeldOverTime` and `HeldOverTimeLinks`, and `Properties.RemoteAudit.Line`'s `LineHeld` and
`LineHeldIdentity` state that line holds, with the named assumption `NoOutsideCompute` among their premises.
