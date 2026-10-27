# Remote bandwidth audit (`verity.core.protocols.remote_bandwidth`)

The remote bandwidth audit v0 (`verity/remote-bandwidth/v0`): a verifier with no hardware at the site certifies, window
by window, how many reads of a node's registered store the node made. Its definition is the Lean package `lean/`
(`RemoteAudit`); this README is its outline and signposts, the Python API, and what Lean doesn't cover yet. The spec
this replaces, `PROTOCOL.md`, is in the history at commit `08da379cb780`.

## The protocol, in outline

- **Parties.** The prover is the node: its GPUs' HBM, its host's DRAM, its CPUs and NICs, none of them trusted. The
  challenger is the proof service's one challenger (`sampled_proofs.challenger`), whose live coins for the session are
  drawn and committed before it starts (`coins.py`, `verity/remote-bandwidth/coins/v1`). The verifier is the Lean
  process `remote-audit-verifier`, run as a party of its own on a host the prover doesn't control (`verifier_party.py`,
  `p2_party.py`; `trusted-host`).
- **Registration.** The node registers its store before any coin: `N` blocks of 4,096 bytes, a payload root over
  hm96-salted payload leaves, a codec seed and its codec (kv/v0: Sloth over 2^1024 − 1, `p1024/k3/masked`), the
  receipt kind and the block layout. A node has one live registration. The store binding decides whether the store is
  the one a pinned model fixes.
- **Windows and passes.** Each window opens on a live challenge coin. Inside a window the node reads its store in
  passes, each a plan of distinct blocks and one receipt a block in a frame-v3-sha512 tree. A pass's seed is derived
  from the window's challenge and the previous pass's root, so the passes are chained (Fiat–Shamir inside a window,
  which Daniel allowed for this protocol on 12 Oct because a live coin a pass is impractical at its rate). Each root is
  due by the window's deadline, so every read serves at most one receipt. KV pages written during the session join the
  store as appends, each under its own append coin drawn after its root, and leave it by releases; a block appended
  `L` or more windows before is in no plan (the lifetime cap).
- **Openings and verdict.** Once a window's roots are final, its openings coin draws receipts, and the prover opens
  each: the block decodes to its payload, the payload leaf is on the payload root (or its append's root), and the
  receipt recomputes. The verifier accepts the session once every window verifies, and publishes the reads it
  certifies and the error that holds at, `{"eps", "q_bytes"}`, from the pooled bound over the session's windows at
  δ = 2^-40.
- **What follows from it.** `ledger.py` turns an accepted record into the declared outputs: γ, the share of the node's
  read capacity `R` over the session's span that the credited reads don't fill, as the sum of the terms timing,
  sampling, on-chip storage, AuditRO's slack, rebuilds, the host tier, links and carriers. The memory line held over
  time is the line of record a gate run publishes (`HeldOverTime`, `HeldOverTimeLinks`).
- **The P2 store.** Over PoUS's P2-encoded weights store (`p2.py`), the registration comes from PoUS's setup, a
  receipt is a unit of four 2,056-byte blocks that the decode kernel sketches as it decodes, and the openings are
  checked on PoUS's `vk`.

## Where each part is defined

The Lean is the verifier of record (Daniel, 18 Oct). Python mirrors it and is held to it by `lean/tests/`' difftest,
every refusal compared.

| Part | Lean (`lean/RemoteAudit/`) | Python |
|---|---|---|
| A session's events (`start`, `boundary`, `append`, `release`, `receive`, `close`, `verify`), its failures, deadlines, steps, the lifetime cap and `RECEIPT_LIMIT` | `Session` | `audit.Session` |
| One openings coin over `g` windows; one or two windows in flight (`Pace`) | `Shared` (`Shared.run g`), `Flight` | `audit.Params.groups_per_coin`, `audit.Pace` |
| A window's draw and the checks on each opening (shape, receipt path, decode, payload path, receipt) | `Window` (`verifyWindow`, `checkWindow`; `REGISTRATION_DOMAIN`, `PAYLOAD_DOMAIN`, `APPEND_DOMAIN`, `OPENINGS_DOMAIN`, `CODEC_DOMAIN`) | `audit.Session.verify` |
| Seeds, the chain over appends, receipts (block-sha512, sketch-sha512), the receipt tree | `Receipts` | `receipts.py` |
| Plans, whole or as a change of the last | `Plan` (`PLAN_DOMAIN`) | `receipts.Plan` |
| The codec's decode, the bound key `b_j(c_a)` | `Codec` | `codec.py` |
| The openings frame | `Wire` | `wire.py` |
| The registry: one live registration a node, `registry/v0` | `Registry` | `audit.Registry` |
| The schedule check | `Schedule` | `schedule.py` |
| The store binding (`bind`, `hold`, `expectedRoot`) | `Store`, `Unboxed` | `store.py` |
| The pooled bound | `Pooled` | `pooled.py` |
| The memory line held over time (`lineOfRecordAt`) | `Line` | `line.py` |
| The P2 store's registration digest, sketch receipts and openings | `P2` | `p2.py` |
| The core protocol `verity/remote-bandwidth/v0` and what it publishes (`certify`) | `Protocol` | `period.py` |
| The verifier's process: ops `init`, `bind`, `hold`, `register`, `session`, `event`, `openings`, `verify`, `settle`, `abort`, `outcome`, `close`, `registry`, `line_record`, `line` | `Verifier` | `benchmarks/remote_session/verifier.py` drives it |
| The difftest's port | `Difftest` | `lean/tests/` |

Executables: `remote-audit-verifier`, `remote-audit-protocol` and `remote-audit-difftest` (`lean/lakefile.toml`). The
byte formats are also pinned by `tests/vectors.json`.

The guarantees are in `Security/`:

- The Lean verdicts' properties: `Security/Properties/RemoteAudit/` (`AcceptIsVerified`; `PooledCovers` and
  `LineCoversSession`; `LineHeld`, that the memory line is true of the trace that produced the record; the coins'
  order; the schedule's).
- Conservation over a session's passes: `Accounting.Guarantees.RemoteSessionRunConservation` and
  `RemoteWindowRunConservation`.
- The draws: `RateWindowDraw` and `RateSessionDraw` (from `HorizonSampling`), and `Remote.PooledDraw` on the deployed
  draw (`Accounting.Remote.Draw`).
- The receipts: `ReceiptsBindingOfCR`; the sketch's bounds `RateAuditSketch`, `SketchBoundSession` and
  `SketchBoundP2`; AuditRO's slack, `Bandwidth.Guarantees.AuditRO` applied per pass.
- The memory line: `Remote.HeldOverTime` and `HeldOverTimeLinks`.
- The circuit's requirement: `RequirementMoved`, `DeclaredCircuitRequirement`, `StoreMatches`, `PlanMatches`,
  `ScheduleDeterministic`, `SteeredLayout`.
- Named assumptions: `WorkHardness` (and `WorkHardnessP2`, `SquaringCeilingP2`), `NoOutsideCompute`,
  `ProviderLockdown`.

## Which side decides

`benchmarks/remote_session/verifier.py` runs both verifiers, by its one switch line `VERDICT` (`--verdict`). Under
`lean`, the default since the 8 Oct ruling (the friction skill), the Lean process decides each registration, event and
window, an accepted session's record is the one it returns, and the ledger reads that record. `audit.Session` keeps the
books beside it as the reference, and a registration, event, window verdict or record field on which they differ
refuses the session at that window as `disagreement`. Under `both` `audit.Session` checks every window's openings too;
under `python` there is no Lean. The driver's scheduling (`Session.next_boundary_ns`), the plans the `memory`
diagnostic reads, and the window and draw counts `verifier.json` keeps are still read from the books; the Lean checks
the same plans event by event.

The store binding's decision is the Lean's: the verifier's process takes the pin's bytes and the weights blocks'
digests at startup (`bind`) and decides each hello (`hold`), and the gate line (`auditor_line.py`) credits `bound` only
where the Lean held the hello's root. Reading the files stays Python only: the pin's bytes, `check`, the safetensors
headers and config.json, the arena's layout (`tensors`, `arena`) and the weights blocks' digests (`weight_digests`).
The binding lives in this tree because a period's circuit is this tree's git hash (`verity/remote-bandwidth/v0
verifier tree <tree>`, `sampled_proofs.challenger.credited_session`), so a change here gives later periods a new
circuit.

The memory line of record (`auditor_line.py`, from `verifier.py`'s `recorded_inputs`) is the Lean's `line_record`,
computed off the Lean verifier's record of the session and refused unless the record is of the node's registration
given with it (`lineOfRegistered`); `line.py` computes it beside it from the same inputs, and `auditor_line.py` refuses
the line where they differ.

## Replay

`lean/tests/replay.py` replays a recorded session through both verifiers and compares their verdicts and reasons: a
transcript (every event, its clock, the coins and each window's openings, which `verifier.py --transcript DIR` writes)
replays in full, and a remote session's `verifier.json`, which keeps no roots, plans, coins or openings, replays as the
case its arrivals, steps, draws and parameters make.

~~~text
uv run python verity/core/protocols/remote_bandwidth/lean/tests/replay.py record VERIFIER.json ... --out ROWS.json
uv run python verity/core/protocols/remote_bandwidth/lean/tests/replay.py transcript CASE.json ... [--record VERIFIER.json ...]
~~~

A transcript written before it named its session's pace (20 Oct) replays only with the pace its `verifier.json` gives
(`--record`), matched by the transcript's sha256.

## Code

- `audit.py`: the lifecycle (`register`, `Registry`, `Session`), pacing (`Pace`), the record (`WindowRecord`) and its
  ledger.
- `ledger.py`: the declared outputs from an accepted session's record (`declared`): γ at both readings of `R`, per
  window and over the session, and the memory line's three forms (snapshot, time-held and link-charged) on the
  verifier's node (`Node`).
- `coins.py`: a session's live coins, an epoch of the proof service's challenger (in process or through
  `sampled_proofs.remote`), each opened after the root it follows.
- `receipts.py`: seeds, plans, receipts and their trees. `codec.py`: the store's encoding (kv/v0).
- `p2.py`: the audit over PoUS's P2 store (`Sketcher`, the decode kernel's CPU reference). `p2_party.py`: a P2
  session's verifier and challenger as a party of their own; `prove` is the prover's side.
- `verifier_party.py`: a period's session's verifier and challenger as a party of their own (`period.session`'s
  case, played over a socket); `prove` is the prover's side.
- `wire.py`: the openings frame (`openings_of`).
- `memory_map.py`: memory-map/v1, serving's record of the KV pages it wrote and freed, as the audit's appends,
  releases and plan changes, and as the schedule the prover publishes (`schedule`).
- `schedule.py`: the schedule check, without which `ledger` credits nothing after the seed.
- `store.py`: a registered store bound to a pinned model (`read_pin`, `pin_of`, `check`, `arena`, `Store`,
  `expected_root`, `Binding`). `benchmarks/remote_session/store_layout.py` writes pins and synthetic checkpoints.
- `pooled.py`, `line.py`: the pooled bound and the memory line in exact rationals, beside their Lean.
- `opening.py`: a drawn receipt's checks as one proof block (`verity.ml.commitments.gates.block_check`) in place of
  its opening in the clear.
- `period.py`: one session a period on one node, the bandwidth component of onsite Verity.
- `timing.py`: the honest schedule, the stated claim read through the ledger, and the sketch's rank bound
  (`sketch_reading`, read at `r* = 12`).
- `monitoring.py`: one accepted session's `q_bytes` and `eps` over a week, a month and a year of sessions, pooled by
  #4413's count bound.

Tests: `uv run tools/verity/check/suites.py verity/core/protocols/remote_bandwidth` (the Python), and
`verity/core/protocols/remote_bandwidth/lean` (the difftest and the replay, which build the Lean).

## Not in Lean yet

Each item is memory-accounting's unless it names another owner.

- **Zero-knowledge openings.** The openings are checked in the clear here. Deployed, each drawn receipt is one proof
  block (`opening.py`: `verifyWindow`'s checks on one opening, plus the link from an appended block's payload to the
  circuit's committed KV values), and the plans and the schedule are committed, not sent in the clear. The units'
  circuits are open.
- **The declared outputs.** `RemoteAudit.Protocol` publishes `q_bytes` and `eps` from the pooled bound; γ's terms
  (`ledger.py`) and the stated claim (`timing.stated`) are Python only, read with the node's measured `R` and on-chip
  storage. `μ_host` and `μ_link` are models.
- **The requirement's link.** Counting the audited reads as the served circuit's needs the memory-map and plan units
  (I4) to prove the session's appends, releases and plans are the circuit's memory map: every append a write of the
  map, in its order, no dummy appends, a re-key only at the lifetime cap, every other free at its values' last
  structural reader, and each plan the resident set at its step (`PlanMatches`). `ledger` takes their verdicts as
  given (`RequirementLink`). Owner: proofs; the audit's side (the chain, the heads, the stale bounds) is here.
- **The schedule check's deployment.** The schedule is witness, so deployed it is committed and its check is a proof
  block; the cap travels with the session's parameters, and `benchmarks/remote_session`'s hello carries none.
- **The timing of a pass.** Nothing yet shows that the pass at step `s` ran while serving executed `s`; a certifier
  that stamps each step's record and each pass's root on one clock would. Owner: memory-accounting with
  network-accounting (the certifier).
- **KV pages registered mid-session.** Credit waits on the property's verdict on their writers (`PostSeedVerdict`),
  whose binding to this session's appends is I2's and I4's to check. Appends end with their session. The device's KV
  receipt (`kv_receipt.py`, `kv.cu`) isn't the registration's kind, so no KV page is credited until it is, and serving
  doesn't yet pass the bound key to `kv.cu` and `paged.cu`.
- **The gap `binding-split`.** `ledger.binding` takes the chance that two runs from a window's cut root it differently
  as 0, and every line says it is conditional on that (`auditor_line.py`'s `CONDITIONAL_ON`). #4968 sends the coin
  commitment only after the prover's last root; the statement that the play then fixes each window's receipts by its
  cut is open.
- **The P2 session.** `RemoteAudit.P2` holds the P2 store's registration, receipts and openings; the session over a
  P2 store is the Python's until #4977 makes it the Lean's.
- **A registration that binds `z`.** Both verifiers refuse a registration that binds PoUW's tiles; where `z` comes
  from (Pearl-C's k-slice accumulators) and the schedule of tiles are design only.
- **The challenger's state.** Persisting a session's unreleased coins across challenger restarts is out of scope.
  Owner: proofs (the proof service's challenger).
- **Multi-GPU trees.** Each GPU's blocks should be an aligned power-of-two range of leaves, so a pass's root combines
  its GPUs' subtree roots.

## History

The chained passes and their Fiat–Shamir seeds: Daniel, 12 Oct (the live-coin design they replaced is in the old
spec's Appendix A). Lean as the definition and verifier of record: Daniel, 18 Oct. The append coin, the lifetime cap
and the memory map's choices: the red team's KB-2 (note:20261017T1859Z-report-property-redteam) and
note:20261017T1855Z-report-stacking-lean. The requirement's link: note:20261017T1847Z-draft-a1-requirement and
note:20261019T0021Z-report-format-r6-r8. The pooled count bound: #4413. The schedule's cap: red team round 5, RD-18.
