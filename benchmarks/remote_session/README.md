# PoUB's session harness

The session harness of PoUB (`verity/protocols/resource_accounting/memory/poub/verifier`), not the remote
protocol's application, which is `applications/remote.toml`. It runs PoUB end to end: a verifier's process and a prover's, over TCP,
and the memory line a gate run publishes.

**Backs:** its line reads the conclusions of
`ResourceAccounting.Guarantees.Remote.HeldOverTime` and `HeldOverTimeLinks`, and `Properties.Poub.Line`'s `LineHeld` and
`LineHeldIdentity` state that line holds, with the named assumption `NoOutsideCompute` among their premises. The
line's error reads δ + Σ_t adv_t (SHA-512 stream as PRF: not priced, not assumed): the table prints it in those words
(`ledger.DELTA_FORM`) with δ's number after them, and every error of the document names the term (`NAMED_TERMS`).
A line's δ is its credited sessions' summed; where that is 1 or more the line holds no share and says "no bound at
this period" with the sum (`verity/protocol.py`'s rule for every row's error).

## Try it

~~~sh
uv run tools/check/suites.py benchmarks/remote_session
~~~

It runs both processes over a loopback on the CPU, the driver's reference (`CpuBackend`) as the prover's backend, and
the Lean line test where `lake` is. It writes no result.

| Command | Where | Writes to `$RESEARCH_RUN_DIR/out` |
| --- | --- | --- |
| `bash benchmarks/remote_session/run.sh` | GPUs, one side a machine: `ROLE=verifier` on a machine the developer doesn't control and `ROLE=prover` on the developer's, with no default (`ROLE=both` runs the two on one machine, a test of the scripts that audits nothing); its header lists `PLAN`, `GIB`, `GPUS` and the sessions' parameters | `verifier.json`, `prover.json` and their logs |
| `bash benchmarks/remote_session/rows.sh` | under `research run --cwd clone`; `BUILD_ONLY=1` needs no card | `rows.json`, `same_sass.json` and `sass_gate.json` |
| `benchmarks/remote_session/gate_run.sh [--dry-run] PERIOD` | an agent's VM: a gate run in one command, each step a recorded job | the jobs' ids, to `~/.research/gate-runs/PERIOD.env` |
