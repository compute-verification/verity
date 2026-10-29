# PoUB (`verity.core.protocols.poub`)

PoUB v0: a verifier with no hardware at the site certifies, window by window, how many reads
of a node's registered store the node made. `PROTOCOL.md` has the semantics, the byte formats and the record.

## Which side is the definition

The verifier's definition is the Lean package `lean/` (`PoubCheck`: Lean is the definition and the
verifier of record, Python mirrors it, and PoUB switches first). `audit.py` and what it calls here
(`receipts.py`, `codec.py`, `coins.py`) mirror it and are held to it by `lean/tests/`' difftest: every event's result,
the session's outcome and its record, the registry (`Registry`: one live registration a node, each session's index
`k`, the outcomes `close` counts, and `registry/v0`), and the pieces (registration and its codec seed, codec, plans,
receipts, openings, draws).
Where the two differ, the Python is what is wrong. The properties of the Lean's verdicts are
`Security/Properties/PoubCheck.lean`.

`benchmarks/poub_session/verifier.py` runs both. The Lean's process `poub-check-verifier` (`PoubCheck.Verifier`)
is the verifier of record: it decides each registration, event and window, an accepted session's record is the one
it returns, and the ledger reads that record. `audit.Session` keeps the books beside it as the reference, and a
registration, event, window verdict or record field on which they differ refuses the session at that window as
`disagreement`. The books check no openings: a window's verdict is the Lean's, and the Python check of its openings
runs beside the Lean's in a transcript's replay (`lean/tests/replay.py`). The Lean became the verifier of record under
the 8 Oct ruling (the friction skill), on these grounds:

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

The store binding (`store.py`) decides whether a registered store is the one a pinned model fixes (`verifier.py
--model`, and the gate line's `bound` check, `auditor_line.py`). Its decision from what the files hold is Lean's,
`lean/PoubCheck/Store.lean`: the verifier's Lean process takes the pin's bytes and the weights blocks' digests at
startup (`bind`) and decides each hello (`hold`), its payload root's subtrees of 2^16 leaves on `--layout-workers`
threads, and `Binding` beside it is the reference: a hello the two decide differently (the regions or reason, or the
expected root) is refused as `disagreement`, and the gate line credits `bound` only where the Lean held the hello's
root. `lean/tests/test_binding.py`'s difftest holds it to the Python, every refusal's reason compared: the pin from its
file's bytes (`pinOf`, `modelPin`; the pin, a registry line, a registry's lines of a role and core's `models.json`), the
canonical JSON and `checkpoint_sha512`, the payload salts (`storeSalt`), the payload root from the weights blocks'
digests (`expectedRoot`, also against `audit.payload_tree`), the hello's decision (`hold`) and the two requests
(`binding`). What stays Python only is reading the files: the pin's bytes, `check` (each pinned file's SHA-512), the
safetensors headers and config.json, the arena's layout from them (`tensors`, `arena`) and the weights blocks' digests
(`weight_digests`); the Lean takes those digests as given. It lives in this tree because a period's circuit is this
tree's git hash (`verity/poub/v0 verifier tree
<tree>`, `sampled_proofs.challenger.credited_session`), so the circuit a period signs commits to the code that decided
`bound`. A change to it, like any change here, gives later periods a new circuit.

The memory line held over time is the line of record a gate run publishes (`auditor_line.py`, from `verifier.py`'s
`recorded_inputs`; `paced_line` is the verifier's diagnostic): `HeldOverTime`'s and `HeldOverTimeLinks`' conclusions
read over each stretch of a paced session's groups, from their timings, the store, the on-chip storage, the node's
rates and bursts and the wrong receipts, in exact rationals. The Lean computes it (`lean/PoubCheck/Line.lean`, its
W from `Pooled.lean`), as the verifier process's `line_record` request, off the Lean verifier's record of the session
(`lineOfRecordAt`: its groups' passes, its windows' receipts and draws and its registered blocks the record's, each
window's draws refused unless `min(s, U_t)`; drawn g > 1 windows a coin, its draws' receipts and draws, each `g`
windows' summed and refused unless `min(s, U)` and below 2^64, and at one window in flight no line), refused unless
the record is of the node's registration given with it (`lineOfRegistered`), with the auditor's times and each
group's reads by kind beside it; and
`line.py` computes it beside it from the same inputs (`line_of_record`): `auditor_line.py` holds the two exactly equal
and refuses the line where they differ, and `lean/tests/test_line.py`'s difftest holds them together, every refusal
compared word for word. The Lean statement that the line is true of the trace that produced the record is
`Security/Properties/PoubCheck/Line.lean`'s `LineHeld`.

## Replay

`lean/tests/replay.py` replays a recorded session through both verifiers and compares their verdicts and reasons: a
transcript (the difftest's case: every event, its clock, the coins and each window's openings, which
`verifier.py --transcript DIR` writes for each session it runs) replays in full, a `receipt`, `codec` or path rejection
included, and a remote session's `verifier.json`, which keeps no roots, plans, coins or openings, replays as the case
its arrivals, steps, draws and parameters make (the module's docstring says how it is held to the recorded verdict).

~~~text
uv run python verity/core/protocols/poub/lean/tests/replay.py record VERIFIER.json ... --out ROWS.json
uv run python verity/core/protocols/poub/lean/tests/replay.py transcript CASE.json ... [--record VERIFIER.json ...]
~~~

A transcript written before it named its session's pace (20 Oct) replays only with the pace its `verifier.json` gives
(`--record`), matched by the transcript's sha256.

## Code

- `audit.py`: the lifecycle (`register`, `Registry`, `Session`), pacing (`Pace`: a live coin every k passes, two
  windows in flight or, at `in_flight` 1, each coin after the root before it), the record and its ledger.
- `ledger.py`: the declared outputs from an accepted session's record (`declared`): γ at both R, per window and over
  the session, and the memory line's three forms (snapshot, time-held and link-charged) on the verifier's node
  (`Node`), the paced ones `line.py`'s stretch held over time with the whole session one stretch, in floats.
- `coins.py`: a session's live coins, an epoch of the proof service's one challenger (`sampled_proofs.challenger`, in
  process or through `sampled_proofs.remote`), each opened after the root it follows, and what the auditor's ledger
  opens for the session's window inside the audit's period, read through `challenger.pinned_session`: the host's pins
  log records the session's pin, on the verifier's own first-parent list of main (`trusted`).
- `receipts.py`: seeds, plans, receipts and their trees.
- `codec.py`: the store's encoding (kv/v0).
- `p2.py`: the audit over PoUS's P2-encoded weights store (`PROTOCOL.md` §11): the registration from PoUS's setup,
  receipts of units of four 2,056-byte blocks that the decode kernel sketches as it decodes (`Sketcher`, its CPU
  reference), and openings checked on PoUS's `vk`. Its Lean is `lean/PoubCheck/P2.lean`, held to it by
  `lean/tests/test_p2.py`; the Lean session doesn't take a P2 store yet.
- `p2_party.py`: a P2 session's verifier and its challenger as a party of their own, on a host the prover doesn't
  control: started over ssh with the channel on its stdin and stdout (`--stdio`, as an auditor gate's verb starts it)
  or on a TCP port, it holds the auditor's registration and its plan (the units holding a tensor's block, not the
  store's padding past them, which is PoUS's: MEM-72), drives the session on its own clock and coins, and answers
  the verdict with the credited reads (`q_bytes`). `prove` is the prover's side of the channel.
- `wire.py`: the openings frame a prover sends a window's openings in, and a window's openings from its frames
  (`openings_of`), beside `lean/PoubCheck/Wire.lean`'s.
- `memory_map.py`: memory-map/v1, serving's record of the KV pages it wrote and freed, as the audit's appends,
  releases and plan changes, and as the schedule the prover publishes (`schedule`).
- `schedule.py`: the schedule check, without which `ledger` credits nothing after the seed: each pass's plan is the
  verifier's plan of the published schedule at its step, the schedule is within the cap on its bits the session
  committed before its coins (`audit.Params.schedule_cap`), and that cap is the layout's steering
  (`audit.Session.scheduled`, `audit.Ledger.steering`). Of record in `lean/PoubCheck/Schedule.lean`.
- `store.py`: a registered store bound to a pinned model. It reads and checks the pin (`read_pin`, `pin_of`, `check`),
  lays out the arena of a BF16 checkpoint's weights in vLLM's module order and the store of arenas and host filler
  (`arena`, `Store`), and computes the payload root that store has under the verifier's salt key (`expected_root`).
  `Binding.hold` decides a hello against it, the reference beside `lean/PoubCheck/Store.lean`'s `bind` and `hold`,
  which decide it in the verifier.
  `benchmarks/poub_session/store_layout.py` writes pins and synthetic checkpoints.
- `timing.py`: the honest schedule, the stated claim read through the ledger, and the sketch's rank bound.
- `monitoring.py`: one accepted session's `q_bytes` and `eps` over a week, a month and a year of a deployment's
  sessions, at δ = 1% and 10⁻⁶: pooled by the count bound (all but M − 1 sessions certify), with each session at
  δ/T beside it. The deployment's machines and window are parameters.

Tests: `uv run tools/verity/check/suites.py verity/core/protocols/poub` (the Python), and
`verity/core/protocols/poub/lean` (the difftest and the replay, which build the Lean).
