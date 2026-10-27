# Remote bandwidth audit (`verity.core.protocols.remote_bandwidth`)

The remote bandwidth audit v0 (`verity/remote-bandwidth/v0`): a verifier with no hardware at the site certifies, window
by window, how many reads of a node's registered store the node made. The node (its GPUs' HBM, its host's DRAM, its
CPUs and NICs) is the prover, and none of its hardware is trusted. The store is 4 KiB chunks encoded under a codec seed
drawn after the payload root (kv/v0: Sloth over 2^1024 − 105, masked, k = 3). Each window's challenge and openings
coin are live coins from the proof service's one challenger. Inside a window the passes are chained: each pass's seed
is derived from the window's challenge and the prover's previous root, so a read serves at most one receipt of the
session. Each pass commits a receipt of every chunk in its plan. The verifier draws `samples` receipts a window after
its roots are final and checks their openings. An accepted session certifies its correct receipts' reads, at least
(1 − γ) of the node's read capacity R over the session's span, and a memory line: the registered bytes the session
shows held in node memory, at least (1 − ε)·N·b except with δ. The same audit runs over PoUS's P2-encoded weights
store (`p2.py`), with units of four 2,056-byte chunks.

## The Lean verifier

The verifier is the Lean package `lean/` (`RemoteAudit`). Python mirrors it, and `lean/tests/`' difftest holds the two
together: every event's result, the session's outcome and its record, the registry and the pieces (registration, codec
seed, codec, plans, receipts, openings, draws). Where they differ, the Python is wrong.

| Lean | what it decides | Python held to it |
| --- | --- | --- |
| `RemoteAudit.Protocol` | the session behind the core interface (`Core.Protocol`); `certify` publishes `q_bytes` and `eps` | `period.py` |
| `RemoteAudit.Verifier` | the process `remote-audit-verifier`: every registration, event and window, and the session's record | `benchmarks/remote_session/verifier.py` runs it |
| `RemoteAudit.Session` | a session's bookkeeping, event by event, on the verifier's clock | `audit.Session` |
| `RemoteAudit.Window` | `verifyWindow`: a window's draw and its openings' checks | `audit.Session.close`, `verify` |
| `RemoteAudit.Registry` | one live registration a node, `registry/v0` | `audit.Registry` |
| `RemoteAudit.Receipts`, `Plan`, `Codec`, `Wire` | seeds, the chain over appends, plans, receipts, trees, decoding and the openings frame | `receipts.py`, `codec.py`, `wire.py` |
| `RemoteAudit.Shared`, `Flight` | one openings coin over several windows; a paced session at two windows in flight | `audit.Params.groups_per_coin`, `audit.Pace` |
| `RemoteAudit.Pooled`, `Line` | the openings' pooled bound in exact rationals; the memory line of record, refused unless the record is of the registration given with it (`lineOfRegistered`) | `pooled.py`, `line.py` |
| `RemoteAudit.Schedule` | the schedule check: each pass's plan is a function of the published schedule | `schedule.py` |
| `RemoteAudit.Store`, `Unboxed` | whether a registered store is the one a pinned model fixes | `store.py` |
| `RemoteAudit.P2`, `P2Session` | a P2 store's registration, receipts and openings, and its session | `p2.py`, `p2_party.py` |

The verifier's process is the verifier: an accepted session's record is the one it returns, and the ledger reads that
record. The Python (`audit.Session`, `p2.Session`, `store.Binding`) runs beside it as the reference, and a verdict or
record field on which the two differ refuses the session as `disagreement`.

What the Lean accepts is stated in `Security/Properties/RemoteAudit.lean` and `Security/Properties/RemoteAudit/`
(`Properties.RemoteAudit`): `Coins` (draws openable, openings unread until drawn), `Live` (the live process is the
session's run), `Shared`, `Pooled` (`PooledCovers`), `Line` (`LineChecked`, `LineRegistered`, `StretchBound`, and
`LineHeld`, which reads that no device outside the node computed for it from the network certifiers' observed ingress),
`Schedule` (`ScheduleChecked`, `ScheduleFixesLayout`; `steered_of_checked` under `ScheduleLayoutGap`) and `Component`
(`AcceptIsVerified`, `LineHeldAtComponent`), with the chain, the plans' distinctness, the steps, the lifetime cap and
released chunks staying out of later plans in `Properties.RemoteAudit` itself.

## Guarantees

The guarantees are in `Security/Definitions/Accounting/Guarantees/`, each proved as
`Accounting.SecurityProofs.G : Accounting.Guarantees.G` and listed in `Security/lean-audit.json`:

- **Conservation:** `RemoteSessionRunConservation` and `RemoteWindowRunConservation`, each with a `Relay` form
  (`RateAuditDeployed.lean`), on `Bandwidth.Guarantees.AuditRO` applied per pass.
- **The draws:** `RateWindowDraw`, `RateSessionDraw` and `HorizonSampling` (`RateAuditHorizon.lean`), and
  `Remote.PooledDraw` on the deployed draw, with no hiding term (`RemoteDraw.lean`).
- **The receipts:** `Remote.ReceiptsBindingOfCR` (`RemoteBinding.lean`); `RateAuditSketch`, `RateAuditTensorSketch`
  and `RateSketchTable` (`RateAuditSketch.lean`); `Remote.SketchBoundSession` and `Remote.SketchBoundDeployed`
  (`RemoteSketch.lean`), `Remote.SketchBoundLine` (`RemoteSketchLine.lean`) and `Remote.SketchBoundP2`
  (`RemoteSketchP2.lean`).
- **The memory line:** `Remote.Held`, `Remote.HeldOverTime` and `Remote.HeldOverTimeLinks`, each with a `Drawn` form,
  and `Remote.LiveCoinsDrawn` (`RemoteHeld.lean`).
- **Credit after the codec seed:** `DeclaredCircuitRequirement` (`DeclaredCircuit.lean`) and `RequirementMoved`
  (`Requirement.lean`).

Their named assumptions are `Accounting.Remote.Held.Assumptions` (`NoOutsideCompute`, `Identity`, `Links`, `Covers`,
`Paced`, `SketchBoundAt`, `ChipCeiling`, `SquaringCeiling`, `Disks`), `Accounting.Remote.Draw.Assumptions` (`Live`,
`Refused`, `Counted`) and, for the sketch, `Accounting.Assumptions.SketchRank` and `SketchHashBudget`. P2's
`WorkHardnessP2` and `SquaringCeilingP2` are in `Security/Definitions/Accounting/Assumptions/RemoteP2.lean`.

## Code

- `audit.py`: the lifecycle. `register` and `Registry` (`register`, `session`, `close`, `live`, `drawn`), `Params`
  and `Pace`, `Session` (`start`, `boundary`, `next_boundary_ns`, `append`, `release`, `receive`, `close`, `verify`,
  `outcome`, `record`, `disclose`), the prover's `Prover` and `Store`, `Record` and `WindowRecord`, `window_profile`,
  `payload_tree` and `append_tree`, and the ledger (`ledger`, `Ledger`, `pooled`, `floors`, `Rebuild`,
  `RequirementLink`, `PostSeedVerdict`, `Scheduled`).
- `receipts.py`: `seed`, `Plan` (`of`, `from_array`, `changed`, `digest`), the receipt kinds (`KINDS`: `sketch-sha512`,
  the default, and `block-sha512`; `SketchReceipter`, `ROWS`), the chunk layout (`LAYOUTS`, `plan_of`) and the receipt
  tree (`tree`, `leaf_receipt`, `verify_leaf`).
- `codec.py`: `Codec` (`DEFAULT` `p1024/k3/masked`, `FALLBACK` `p1024/k2/masked`, and `p512/k2`), `key`, `encode`,
  `decode`, and `P2Codec`.
- `coins.py`: a session's live coins, one epoch of `sampled_proofs.challenger` (`Coins`, `Released`), released after
  the root each follows, with the commitment and every opening disclosed only at the session's end
  (`audit.Session.disclose`), so the prover holds none of it while the session runs, and no guarantee takes a hiding
  premise on it; `released_reason` holds the challenger's signed log to the session's record, and
  `trusted` reads the auditor's session through `challenger.pinned_session`.
- `ledger.py`: the declared outputs of an accepted session (`declared`): γ at the specified and measured R, per window
  and over the session (`gamma`), and the memory line's snapshot, time-held and link-charged forms (`memory`) on the
  verifier's `Node`, with their δ parts (`DELTA_PARTS`: `draw`, `binding`, `hiding`, `rejection`, `sketch`).
- `monitoring.py`: the monitoring-period table (`table`, format `verity/remote-bandwidth-monitoring-period/v1`): one
  accepted session's `q_bytes` and `eps` over a week, a month and a year of a deployment's sessions, at δ = 1% and
  10⁻⁶, pooled by `prCoin_count_le_choose`'s count bound (`pooled_step`; `short_sessions` in
  `verity/core/protocols/pous/wipe_monitoring.py`, shared with the wipe's) and per window at δ/T (`per_window`).
- `wire.py`: the openings frame (`encode_openings`, `openings_of`).
- `memory_map.py`: memory-map/v1, serving's record of the KV pages it wrote, freed and withdrew, as the audit's
  appends, releases and plan changes (`Replay`), and as the schedule the prover publishes (`schedule`).
- `schedule.py`: the schedule check, without which `ledger` credits nothing after the codec seed.
- `store.py`: a registered store bound to a pinned model (`read_pin`, `check`, `arena`, `Store`, `expected_root`,
  `Binding.hold`). `benchmarks/remote_session/store_layout.py` writes pins and synthetic checkpoints.
- `p2.py`, `p2_party.py`, `p2_lean.py`: the P2 store, a P2 session's verifier and challenger as a party of their own,
  and the Lean verifier's process as that party runs it.
- `period.py`, `session_parties.py`, `verifier_party.py`: one session a period behind the core interface, and the
  session's verifier with its challenger, and its prover, as parties on separate machines.
- `timing.py`: the honest schedule, the stated claim read through the ledger (`stated`), and where the sketch's rank
  bound is read (`sketch_reading`, `sketch_failure`). `python -m verity.core.protocols.remote_bandwidth.timing`
  prints the grid.
- `line.py`, `pooled.py`: the memory line of record and the openings' pooled bound, beside their Lean.

## Records and formats

Every byte format is pinned by `tests/vectors.json`, and the Lean's by the difftest's vectors.

- **Registration:** `Registration.digest` over the node, N, the payload root, the codec seed, the codec, the receipt
  kind, `bind`, the chunk layout (`store-order/v0`) and the sketch's rows. Every session's coin commitment binds it,
  and every session's chain starts from it.
- **Coins:** `verity/remote-bandwidth/coins/v1`, epoch k a session's index over its registration, with the families
  challenge, openings and append; a released coin is `(epoch, label, index, after, coin)` (`coins.Released`).
- **The session's record:** `Record.public()` (the registration digest, the coin commitment's root, N, b, the receipt
  kind, `bind`, the codec and its parameters, the layout, the rows, each window's passes, receipts and post-seed
  receipts, the appends' chain, the lifetime cap, the draws' `samples` and the span), and each `WindowRecord`
  (passes' lengths, steps, heads, the challenge's release, the deadline and the `IntegrityProfile`). `stale` stays in
  the verifier's own record and is never published.
- **The registry:** `registry/v0` (`Registry.to_json`), digests and outcomes only, never the verifier's secret.
- **The memory map:** memory-map/v1 (`memory_map.py`), one line a step.
- **The monitoring period:** `verity/remote-bandwidth-monitoring-period/v1` (`monitoring.table`).

## Decided in Python only

- γ and its terms (`ledger.py`), the timing grid and the stated claim (`timing.py`), and the sketch's rank bound at r*
  (`timing.sketch_reading`), which `SketchRank` takes as a parameter.
- The coin commitment's layout and the challenger's release log (`coins.py`): the Lean takes the released coins as
  given.
- Reading a pinned model's files (`store.py`'s `check`, `tensors`, `arena`, `weight_digests`): the Lean decides the
  binding from the weights chunks' digests.
- Encoding: the verifier only decodes.

## Replay

`lean/tests/replay.py` replays a recorded session through both verifiers and compares their verdicts and reasons. A
transcript (every event, its clock, the coins and each window's openings, which `verifier.py --transcript DIR` writes)
replays in full. A session's `verifier.json`, which keeps no roots, plans, coins or openings, replays as the case its
arrivals, steps, draws and parameters make.

~~~text
uv run python verity/core/protocols/remote_bandwidth/lean/tests/replay.py record VERIFIER.json ... --out ROWS.json
uv run python verity/core/protocols/remote_bandwidth/lean/tests/replay.py transcript CASE.json ... [--record VERIFIER.json ...]
~~~

Tests: `uv run tools/verity/check/suites.py verity/core/protocols/remote_bandwidth` (the Python), and
`verity/core/protocols/remote_bandwidth/lean` (the difftest, the replay and the P2 party's sessions on the Lean
verifier, which build the Lean).
