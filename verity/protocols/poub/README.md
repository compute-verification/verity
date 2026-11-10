# PoUB (`verity.protocols.poub`)

PoUB: how many reads of a node's registered store the node made, window by window, certified by a verifier with no
hardware at the site. This directory is PoUB's parts, one directory each. It holds no code of its own.

## Interface

Nothing is imported from this directory itself: its `__init__.py` is empty. Each part's README states that part's
interface.

| Part | What it is |
|---|---|
| `verifier/` | The verifier and the reference in Python: the store's codec, receipts, the session's lifecycle and record, its declared outputs, and a session over PoUS's P2 store. `PROTOCOL.md` there is the specification. |
| `verifier/lean/` | The verifier in Lean, of record (the Lake package `remote_audit`): the verifier's process and a session as a core protocol, as executables and as modules other packages import. |
| `verifier/remote_session/` | The verifier of live sessions as a process a machine runs, `verifier_process.py`. |
| `device/` | The auditor's side of a deployment: `device/live_verifier/`, the host that runs the verifier and the memory challenger's parties. |

The prover's process is `benchmarks/remote_session/prover.py`, in the session harness, which runs a session end to
end.
