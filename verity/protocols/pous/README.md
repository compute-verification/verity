# PoUS (`verity.protocols.pous`)

PoUS, proof of useful space: a node holds its registered weights encoded as labels, and the memory challenger's timed
rounds check that it still holds them. This directory is PoUS's parts, one directory each. It holds no code of its own.

## Interface

Nothing is imported from this directory itself: its `__init__.py` is empty. Each part's README states that part's
interface.

| Part | What it is |
|---|---|
| `verifier/` | The verifier and the reference in Python: the schemes, setup, one run of PoUS on one node, the timed audit and the streamed wipe. `PROTOCOL.md` there is the specification. |
| `verifier/check/` | The verifier in Lean, of record (the Lake package `pous_check`): the setup and run protocols and the timed audit's verdict, as executables and as modules other packages import. |
| `prover/` | PoUS's Boolean circuits: P2's modular squaring, the decode of one label, a round's answer and the session's verdict. |
| `device/` | The memory challenger, the auditor-trusted device that runs the timed rounds and signs them: its Python, its party and the timed audit's process. |
| `device/lean/` | The verdicts on what the memory challenger signed, in Lean (the Lake package `core_memory_challenger`). |
| `lean/` | The grader of a submitted proof against one of PoUS's frozen statements (the Lake package `pous_grader`). |

The harness, P2's store and the pieces of a served run are in `benchmarks/pous/`. PoUS's definitions, guarantees and
proofs are in `security/` (`Definitions/Pous`, `Proofs/Pous`).
