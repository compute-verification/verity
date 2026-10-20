# Remote bandwidth audit (`verity.core.protocols.remote_bandwidth`)

The remote bandwidth audit v0: a verifier with no hardware at the site certifies, window by window, how many reads of a
node's registered store the node made. `PROTOCOL.md` has the semantics, the byte formats and the record.

## Which side is the definition

The verifier's definition is the Lean package `lean/` (`RemoteAudit`; Daniel, 18 Oct: Lean is the definition and the
verifier of record, Python mirrors it, and the remote audit switches first). `audit.py` and what it calls here
(`receipts.py`, `codec.py`, `coins.py`) mirror it and are held to it by `lean/tests/`' difftest: every event's result,
the session's outcome and its record, the registry (`Registry`: one live registration a node, each session's index
`k`, the outcomes `close` counts, and `registry/v0`), and the pieces (registration and its codec seed, codec, plans,
receipts, openings, draws).
Where the two differ, the Python is what is wrong. The properties of the Lean's verdicts are
`Security/Properties/RemoteAudit.lean`.

`benchmarks/remote_session/verifier.py` runs both, by its one switch line `VERDICT` (`--verdict`). Under `lean`, the
default, the Lean's process `remote-audit-verifier` (`RemoteAudit.Verifier`) is the verifier of record: it decides
each registration, event and window, an accepted session's record is the one it returns, and the ledger reads that
record. `audit.Session` keeps the books beside it as the reference, and a registration, event, window verdict or record
field on which they differ refuses the session at that window as `disagreement`. Under `both` `audit.Session` checks
every window's openings too; under `python` there is no Lean. The default went to `lean` under the 8 Oct ruling (the
friction skill), on these grounds:

- the gap list is empty. A registration that binds PoUW's tiles is refused by both registries. A paced window's
  (k + 1)-th root is refused as `order` by both sessions. A transcript carries its session's pace. The Lean's record
  holds every field the ledger reads, except four. The coins' commitment root is the books'. The block size is a
  constant. The registration's receipt, bind, codec, layout and rows are bound by its digest, which the Lean's record
  names. The integrity profile is a function of the Lean's lengths and draws at δ (`audit.window_profile`).
- every recorded session in the store replays through both to the same verdict (`lean/tests/replay.py`), and to the
  one it recorded wherever its `verifier.json` keeps enough to rebuild its events in turn: three sessions that failed
  `few` don't keep their appends, and both verifiers refuse the same call of their replay.
- the Python keeps running beside the Lean as the reference.
- the protocol's properties are in `Security/Properties/`.

The driver's scheduling (when a boundary leaves, `Session.next_boundary_ns`), the plans the `memory` diagnostic
reads, and the window and draw counts `verifier.json` keeps are still read from the books. The Lean checks the same
plans event by event.

## Replay

`lean/tests/replay.py` replays a recorded session through both verifiers and compares their verdicts and reasons: a
transcript (the difftest's case: every event, its clock, the coins and each window's openings, which
`verifier.py --transcript DIR` writes for each session it runs) replays in full, a `receipt`, `codec` or path rejection
included, and a remote session's `verifier.json`, which keeps no roots, plans, coins or openings, replays as the case
its arrivals, steps, draws and parameters make (the module's docstring says how it is held to the recorded verdict).

~~~text
uv run python verity/core/protocols/remote_bandwidth/lean/tests/replay.py record VERIFIER.json ... --out ROWS.json
uv run python verity/core/protocols/remote_bandwidth/lean/tests/replay.py transcript CASE.json ... [--record VERIFIER.json ...]
~~~

A transcript written before it named its session's pace (20 Oct) replays only with the pace its `verifier.json` gives
(`--record`), matched by the transcript's sha256.

## Code

- `audit.py`: the lifecycle (`register`, `Registry`, `Session`), pacing (`Pace`: a live coin every k passes), the
  record and its ledger.
- `ledger.py`: the declared outputs from an accepted session's record (`declared`): γ at both R, per window and over
  the session, and the memory line's three forms (snapshot, time-held and link-charged) on the verifier's node
  (`Node`).
- `coins.py`: the challenger's committed live coins.
- `receipts.py`: seeds, plans, receipts and their trees.
- `codec.py`: the store's encoding (kv/v0).
- `memory_map.py`: memory-map/v1, serving's record of the KV blocks it wrote and freed, as the audit's appends,
  releases and plan changes.
- `timing.py`: the honest schedule, the stated claim read through the ledger, and the sketch's rank bound.

Tests: `uv run tools/verity/check/suites.py verity/core/protocols/remote_bandwidth` (the Python), and
`verity/core/protocols/remote_bandwidth/lean` (the difftest and the replay, which build the Lean).
