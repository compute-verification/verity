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

The verifier of record is still `audit.py`: `benchmarks/remote_session/verifier.py` runs `audit.Session`. It switches
to the Lean under the 8 Oct ruling (the friction skill): the gap list is empty for the record versions in use, every
recorded run in the store replayed through the Lean gets the verdict the Python gave, the Python keeps running beside it
as the reference, the switch is one reversible line, and the protocol's properties move from `Models/` to
`Properties/`. What the Lean doesn't check yet that `audit.py` does:

- a registration that binds PoUW's tiles, which the Lean refuses;
- the record past the summary `RemoteAudit.Session` returns: the coins' commitment, the integrity profile and the
  ledger's terms.

## Replay

`lean/tests/replay.py` replays a recorded session through both verifiers and compares their verdicts and reasons: a
transcript (the difftest's case: every event, its clock and the coins) replays in full, and a remote session's
`verifier.json`, which keeps no roots, plans, coins or openings, replays as the case its arrivals, steps, draws and
parameters make (the module's docstring says how it is held to the recorded verdict).

~~~text
uv run python verity/core/protocols/remote_bandwidth/lean/tests/replay.py record VERIFIER.json ... --out ROWS.json
uv run python verity/core/protocols/remote_bandwidth/lean/tests/replay.py transcript CASE.json ...
~~~

## Code

- `audit.py`: the lifecycle (`register`, `Registry`, `Session`), the record and its ledger.
- `coins.py`: the challenger's committed live coins.
- `receipts.py`: seeds, plans, receipts and their trees.
- `codec.py`: the store's encoding (kv/v0).
- `memory_map.py`: memory-map/v1, serving's record of the KV blocks it wrote and freed, as the audit's appends,
  releases and plan changes.
- `timing.py`: the honest schedule, the stated claim read through the ledger, and the sketch's rank bound.

Tests: `uv run tools/verity/check/suites.py verity/core/protocols/remote_bandwidth` (the Python), and
`verity/core/protocols/remote_bandwidth/lean` (the difftest and the replay, which build the Lean).
