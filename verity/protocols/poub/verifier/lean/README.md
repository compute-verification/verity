# PoUB's Lean verifier (`remote_audit`)

PoUB's verifier as executable Lean, its definition: a window's receipt checks, a session's books, the verifier's
registrations, the store binding, the memory line, and the same over PoUS's P2 store. The Python verifier
(`verity/protocols/poub/verifier/`) is held to it by this directory's suite. It is code: Lean core and the packages its
`lakefile.toml` requires, with no Mathlib and nothing of the security analysis.

## Interface

What code outside this directory may use. Everything else here is internal.

| Entry point | What it is |
|---|---|
| `remote-audit-verifier` (a Lean executable) | The verifier's process, of record (`RemoteAudit.Verifier.serve`). It reads one request a line on stdin, a JSON object with its `op`, and writes one line back, `{"answer": JSON}` or `{"error": MESSAGE}`. It holds the registry and every session's state, and decides each registration, event and window, the store binding (`bind`, `hold`), a session over PoUS's P2 store (the `p2_*` requests) and the memory line of record (`line_record`). The Python verifier starts it: `verity/protocols/poub/verifier/session/lean_verifier.py` for a live session and `verity/protocols/poub/verifier/p2_lean.py` for a session over the P2 store. |
| `remote-audit-p2-protocol COMMAND` (a Lean executable) | PoUB over PoUS's P2 store as a core protocol, `verity/remote-bandwidth/p2/v0` (`RemoteAudit.P2Protocol.protocol`), the bandwidth component of onsite Verity, on the core protocols' command line: `id`, `check PARAMS COMMITMENT EVIDENCE`, or `serve`, one case a line in and one answer a line out. It replays the verifier's log of a session against the auditor's registration and publishes `eps`, `q_bytes` and `span_ns`. `verity/protocols/poub/verifier/p2_case.py` writes its cases. |
| `remote-audit-protocol COMMAND` (a Lean executable) | A session over a registered store as a core protocol, `verity/remote-bandwidth/v0` (`RemoteAudit.Protocol.protocol`), on the same command line. It replays the case that `run` of `verity/protocols/poub/verifier/period.py` makes and publishes `q_bytes` and `eps` (`certify`). |
| `RemoteAudit`, `RemoteAudit.Session`, `RemoteAudit.Verifier`, `RemoteAudit.Protocol`, `RemoteAudit.P2Protocol`, `RemoteAudit.Schedule`, `RemoteAudit.Shared`, `RemoteAudit.Line`, `RemoteAudit.Basic` (Lean modules of the Lake package `remote_audit`) | What another Lake package imports: the session's state machine (`Session.State`, `Session.Event`, `Session.step`, `Session.run`), the two protocols, the schedule check (`Schedule.check`), a coin shared by several windows (`Shared.run`), the memory line (`Line.lineOfRecordAt`, `Line.ofSummaryAt`) and the pooled bound (`Pooled.pooled`, which `Line` imports). The properties' package and onsite Verity's composite require the package. |
| `uv run tools/verity/check/suites.py verity/protocols/poub/verifier/lean` | The suite, where `lake` is: the Python verifier against the Lean through the difftest harness (the units, the session, the registry, the store binding, the schedule, the memory line and its pooled bound, a replayed session), the P2 store's sessions and party, and the two protocols against the Python's answers. |

`lake build` here builds the executables. `remote-audit-difftest`, which the suite drives, is internal.

Read from outside past this table: the proofs of the session's properties read four private definitions through
`open private`, `live` and `merge` of `RemoteAudit.Session` and `endsOf` and `lexLt` of `RemoteAudit.Plan`.

The package requires PoUS's Lean check (`verity/protocols/pous/verifier/check/`): a P2 store's openings are checked by
PoUS's own leaf and path check, `PousCheck.Merkle`.
