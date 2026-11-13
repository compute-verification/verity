# PoUB (`verity.protocols.poub.verifier`)

Proof of useful bandwidth (PoUB) v0: a verifier with no hardware at the site certifies, window by window, how many reads
of a node's registered store the node made. `PROTOCOL.md` has the semantics, the byte formats and the record; this file
is the outline and the interface.

## What runs today

A session runs end to end on live coins: the verifier and its challenger as a party of their own, on a host the developer
doesn't control (`verifier_party.py`), or every party in one process (`period.py`), against the prover's passes over
its registered store. A period's case replays through the Lean (`RemoteAudit.Protocol`), and the session harness runs
the Lean verifier beside the books (below). Not all of it is zero knowledge yet:

- **Openings are in the clear.** Each window's openings go to the verifier party as they are (`wire.encode_openings`),
  and it checks them itself (`audit.AuditRun.verify`). Those checks are the specification of what each zero-knowledge
  proof block proves (`PROTOCOL.md` §8), but no session proves them: `opening.py` evaluates the opening unit on the
  honest witness, and only `benchmarks/one_stage/a8.py` proves one window's units in zero knowledge.
- **Plans are in the clear.** The verifier sees which blocks each pass covers (`PROTOCOL.md` §8, "What the plan
  shows").
- **The requirement's units and the stacking property are inputs, not proofs.** The ledger credits the served
  circuit's reads only on the plan and memory-map units' verdict (`audit.RequirementLink`), and blocks appended after
  the seed only on the stacking property's (`audit.PostSeedVerdict`); without them it credits the node's reads of its
  registered store. Only `benchmarks/one_stage/a7.py` proves the units.

## Interface

What code outside this directory may use. Everything else here is internal.

| Entry point | What it is |
|---|---|
| `remote-audit-verifier` (the Lean executable, `lean/`) | The verifier's process, of record (`RemoteAudit.Verifier.serve`). It reads one request a line on stdin, a JSON object with its `op`, and writes one line back, `{"answer": JSON}` or `{"error": MESSAGE}`. It holds the registry and every session's state, and decides each registration, event and window, the store binding (`bind`, `hold`), a session over PoUS's P2 store (the `p2_*` requests) and the memory line of record (`line_record`). `session/lean_verifier.py` and `p2_lean.py` start it. |
| `remote-audit-p2-protocol COMMAND` (the Lean executable, `lean/`) | PoUB over PoUS's P2 store as a core protocol, `verity/remote-bandwidth/p2/v0` (`RemoteAudit.P2Protocol.protocol`), the bandwidth subprotocol of onsite Verity, on the core protocols' command line: `id`, `check PARAMS COMMITMENT EVIDENCE`, or `serve`, one case a line in and one answer a line out. It replays the verifier's log of a session against the auditor's registration and publishes `eps`, `q_bytes` and `span_ns`. `p2_case.py` writes its cases. |
| `remote-audit-protocol COMMAND` (the Lean executable, `lean/`) | A session over a registered store as a core protocol, `verity/remote-bandwidth/v0` (`RemoteAudit.Protocol.protocol`), on the same command line. It replays the case `period.run` makes and publishes `q_bytes` and `eps` (`certify`). |
| `RemoteAudit`, `RemoteAudit.Session`, `RemoteAudit.Verifier`, `RemoteAudit.Protocol`, `RemoteAudit.P2Protocol`, `RemoteAudit.Schedule`, `RemoteAudit.Shared`, `RemoteAudit.Line`, `RemoteAudit.Basic` (Lean modules of the package `remote_audit`, `lean/`) | What another Lake package imports: the session's state machine (`Session.State`, `Session.Event`, `Session.live`), the two protocols, the schedule check (`Schedule.check`), a coin shared by several windows (`Shared.run`), the memory line (`Line.lineOfRecordAt`, `Line.ofSummaryAt`) and the pooled bound (`Pooled.pooled`, which `Line` imports). |
| `session.verifier.main(argv: Sequence[str] \| None = None, *, verify: Verify \| None = None, serve: Sequence[str] \| None = None) -> int` (`session/verifier.py`), with `session/wire.py` | The verifier's process of live sessions. It serves one prover over TCP with a secret of its own, the Lean process deciding and `audit.AuditRun` keeping the books beside it, and writes `verifier.json` under `--out`. `session/wire.py` is the frames the two sides exchange (`Conn`, `listen`, `connect`) and their JSON forms (`registration_json`, `registration_of`, `coin_of`, `openings_of`). |
| `audit` (`audit.py`) | The lifecycle the parties drive. The verifier's side is `register(source: bytes, node: bytes, blocks: int, payload_root: bytes, codec: Codec = codec.DEFAULT, receipt: str = receipts.DEFAULT, *, bind: bool = False, layout: str = receipts.LAYOUT, rows: int \| None = None, nonce: bytes \| None = None) -> Registration`, `Registry` (one live registration a node) and `AuditRun` (one session, at its `Params`). The prover's side is `payload_tree` and `append_tree` (its commitments), `Store` (its encoded store) and `Prover` (the honest prover). An accepted session leaves a `Record`. |
| `codec` (`codec.py`) and `receipts` (`receipts.py`) | The bytes a prover's device must reproduce. `codec` is the store's encoding, kv/v0: `key(seed: bytes, codec: Codec = DEFAULT) -> Key`, `encode(payload: bytes, j: int, k: Key, *, bound: bool = False, coin: bytes \| None = None) -> bytes`, `decode`, `digest(payload: bytes) -> bytes` and `BLOCK_BYTES`. `receipts` is a pass's seed (`seed(challenge: bytes, i: int, previous: bytes) -> bytes`), its plan (`Plan`), its receipts (`receipter(kind: str, seed: bytes, rows: int \| None = None) -> Receipter \| SketchReceipter`) and their tree (`tree`, `verify_leaf`, `hidden`). |
| `memory_map` (`memory_map.py`) | memory-map/v1, serving's record of the KV pages it wrote and freed: `parse(lines: Iterable[str]) -> tuple[Header, Iterator[Step]]`, `dumps(head: Header, steps: Iterable[Step]) -> str`, `Replay` (the map as the audit's appends, releases and plan changes) and `schedule` (the schedule the prover publishes). |
| `ledger.declared(record: audit.Record, node: Node, *, wrong: Callable[[IntegrityProfile], float] = IntegrityProfile.bound) -> Declared` (`ledger.py`), with `timing.py` | An accepted session's declared outputs on the verifier's node (`ledger.Node`): γ at both R, for each window and over the session, and the memory line's three forms. `timing.py` holds the device constants a node is built from (a GPU's and a host's on-chip storage and their read, write, SHA-512 and squaring rates) and the least allowance an honest prover meets (`timing.deadline`). `monitoring.table` scales one accepted session up to a deployment's monitoring period. |
| `python -m verity.protocols.poub.verifier.p2_party (--registration FILE \| --pous-record DIR --pous-session ID) (--stdio \| --listen HOST:PORT) [OPTIONS]` | The verifier of a session over PoUS's P2 store and its challenger as a process of their own, for a host the developer doesn't control. It holds the auditor's registration and plan, holds each request to the auditor's parameters (`p2_party.SESSION`), drives the session on its own clock and coins with the Lean deciding, and answers with the verdict and the credited reads (`q_bytes`). |
| `p2` (`p2.py`), `p2_party.prove`, `p2_party.core_answer` and `p2_case` (`p2_case.py`) | The same session in Python. `p2` is the registration from PoUS's setup (`register`, `Registration`), the prover's store and passes (`Store`, `Prover`), and the receipts of units of four labels that a decode kernel must reproduce (`Sketcher`, `UNIT_BLOCKS`, `UNIT_BYTES`). `p2_party.prove(f_in: BinaryIO, f_out: BinaryIO, prover: Prover, request: Mapping[str, Any], log: Callable[[str], None] = lambda _m: None) -> dict[str, Any]` is the prover's side of the channel, and `p2_party.core_answer(a: Mapping[str, Any], r_bytes_per_s: int \| None = None) -> dict[str, Any]` is the verifier's answer in the core protocols' form. `p2_case` writes the session's case for `remote-audit-p2-protocol` (`case_of`, `judged`, `write`) and checks that it names the store PoUS's run is committed to (`same_store`). |
| `period.run(node, store, coins)` (`period.py`) and `verifier_party [--listen HOST:PORT]` (`verifier_party.py`, run with `python -m` as `p2_party` is) | One session over a registered store with every party in one process, with its answer and its case; `period.session` is the verifier's side. `verifier_party` is that verifier and its challenger as a process of their own, serving one prover (`verifier_party.period_with` drives a period through it). |

`remote-audit-difftest`, which holds `audit.py` to the Lean, is internal. `benchmarks/remote_session/` runs a session
over TCP on GPUs, the verifier's side and the prover's each on its own machine.

## Which side is the definition

The verifier's definition is the Lean package `lean/` (`RemoteAudit`), the verifier; the Python mirrors it.
`audit.py` and what it calls here (`receipts.py`, `codec.py`, `coins.py`) are held to it by `lean/tests/`' difftest:
every event's result, the session's outcome and its record, the registry (`Registry`: one live registration a node,
each session's index `k`, the outcomes `close` counts, and `registry/v0`), and the pieces (registration and its codec
seed, codec, plans, receipts, openings, samples). Where the two differ, the Python is what is wrong. The properties of
the Lean's verdicts are `security/Properties/Accounting/Remote.lean`.

`verity/protocols/poub/verifier/session/verifier.py` runs both. The Lean's process `remote-audit-verifier` (`RemoteAudit.Verifier`)
decides each registration, event and window, an accepted session's record is the one it returns, and the ledger reads
that record. `audit.AuditRun` keeps the books beside it as the reference, and a registration, event, window verdict or
record field on which they differ refuses the session at that window as `disagreement`. The books check no openings: a
window's verdict is the Lean's, and the Python check of its openings runs beside the Lean's in a transcript's replay
(`lean/tests/replay.py`). The Lean decides alone because:

- the two refuse the same forms: a registration that binds PoUW's tiles is refused by both registries, a paced
  window's (k + 1)-th root is refused as `order` by both sessions, and a transcript carries its session's pace.
- the Lean's record holds every field the ledger reads but four: the coins' commitment root, which is the books'; the
  block size, a constant; the registration's receipt, bind, codec, layout and rows, bound by its digest, which the
  Lean's record names; and the integrity profile, a function of the Lean's lengths and samples at δ
  (`audit.window_profile`).
- every recorded session in the store replays through both to the same verdict (`lean/tests/replay.py`), and to the
  one it recorded wherever its `verifier.json` keeps enough to rebuild its events in turn: three sessions that failed
  `few` don't keep their appends, and both verifiers refuse the same call of their replay.

The driver's scheduling (when a boundary leaves, `AuditRun.next_boundary_ns`), the plans the `memory` diagnostic
reads, and the window and sample counts `verifier.json` keeps are still read from the books. The Lean checks the same
plans event by event.

The store binding (`store.py`) decides whether a registered store is the one a pinned model fixes (`verifier.py
--model`, and the gate line's `bound` check, `auditor_line.py`). Its decision from what the files hold is Lean's,
`lean/RemoteAudit/Store.lean`: the verifier's Lean process takes the pin's bytes and the weights blocks' digests at
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
tree's git hash (`verity/remote-bandwidth/v0 verifier tree
<tree>`, `spot.challenger.credited_session`), so the circuit a period signs commits to the code that decided
`bound`. A change to it, like any change here, gives later periods a new circuit.

A session over PoUS's P2 store (`p2_party.py`) is decided by the Lean too, since the trusted base is the compiled Lean
verifier, which parses bytes of untrusted provenance, and Python is never trusted: the verifier's process holds the
registration file's bytes, recomputes the registration's digest from its fields, reads the prover's request, refused
unless it restates the auditor's session parameters (its pinned `P2Session.SESSION`, `p2_party.SESSION`, unless the
auditor's party names a toy's), and every root and openings line as sent, and checks each opened unit's four blocks
on the registered vk root
(`lean/RemoteAudit/P2Session.lean`); C* is too large to read whole, and an opening is what is checked of it.
`audit.AuditRun` over the P2 codec keeps the books beside it and can only veto: an event, a held store or an answer on
which the two differ rejects the session `disagreement`. What the Lean takes as given is the clock's readings and the
coins the challenger releases. `lean/tests/test_p2.py`'s `p2_session` difftest holds the books to it on recorded sessions.

The memory line held over time is the line of record a gate run publishes (`auditor_line.py`, from `verifier.py`'s
`recorded_inputs`; `paced_line` is the verifier's diagnostic): `HeldOverTime`'s and `HeldOverTimeLinks`' conclusions
read over each stretch of a paced session's groups, from their timings, the store, the on-chip storage, the node's
rates and bursts and the wrong receipts, in exact rationals. The Lean computes it (`lean/RemoteAudit/Line.lean`, its
W from `Pooled.lean`), as the verifier process's `line_record` request, off the Lean verifier's record of the session
(`lineOfRecordAt`: its groups' passes, its windows' receipts and samples and its registered blocks the record's, each
window's samples refused unless `min(s, U_t)`; sampled g > 1 windows a coin, its samples' receipts and samples, each `g`
windows' summed and refused unless `min(s, U)` and below 2^64, and at one window in flight no line), refused unless
the record is of the node's registration given with it (`lineOfRegistered`), with the auditor's times and each
group's reads by kind beside it; and
`line.py` computes it beside it from the same inputs (`line_of_record`): `auditor_line.py` holds the two exactly equal
and refuses the line where they differ, and `lean/tests/test_line.py`'s difftest holds them together, every refusal
compared word for word. The Lean statement that the line is true of the trace that produced the record is
`security/Properties/Accounting/Remote/Line.lean`'s `LineHeld`.

## Replay

`lean/tests/replay.py` replays a recorded session through both verifiers and compares their verdicts and reasons: a
transcript (the difftest's case: every event, its clock, the coins and each window's openings, which
`verifier.py --transcript DIR` writes for each session it runs) replays in full, a `receipt`, `codec` or path rejection
included, and a remote session's `verifier.json`, which keeps no roots, plans, coins or openings, replays as the case
its arrivals, steps, samples and parameters make (the module's docstring says how it is held to the recorded verdict).

~~~text
uv run python verity/protocols/poub/verifier/lean/tests/replay.py record VERIFIER.json ... --out ROWS.json
uv run python verity/protocols/poub/verifier/lean/tests/replay.py transcript CASE.json ... [--record VERIFIER.json ...]
~~~

A transcript that doesn't name its session's pace replays only with the pace its `verifier.json` gives
(`--record`), matched by the transcript's sha256.

## Code

- `audit.py`: the lifecycle (`register`, `Registry`, `AuditRun`), pacing (`Pace`: a live coin every k passes, two
  windows in flight or, at `in_flight` 1, each coin after the root before it), the record and its ledger.
- `ledger.py`: the declared outputs from an accepted session's record (`declared`): γ at both R, per window and over
  the session, and the memory line's three forms (snapshot, time-held and link-charged) on the verifier's node
  (`Node`), the paced ones `line.py`'s stretch held over time with the whole session one stretch, in floats. Every δ
  and error it gives reads δ + Σ_t adv_t (SHA-512 stream as PRF: not priced, not assumed): `DELTA_FORM`, the term
  named in the object that carries the δ (`NAMED_TERMS`).
- `coins.py`: a session's live coins, an epoch of the proof system's one challenger (`spot.challenger`, in
  process or through `spot.remote`), each opened after the root it follows, and what the auditor's ledger
  opens for the session's window inside the audit's period, read through `challenger.pinned_session`: the host's pins
  log records the session's pin, on the verifier's own first-parent list of main (`trusted`). The prover gets the
  coins in this order: each window's challenge as it is released, its value and the root it follows (`Released`); the
  window's samples after its roots; and, once the session has its outcome and no root can follow, the commitment and
  every released coin's opening (`audit.AuditRun.disclose`; the remote session's `coins` after `ended`), which it
  checks each coin it was sent against (`received_reason`). The commitment's root is a function of every coin, so the
  prover holds none of it while the session runs, and no guarantee takes a hiding premise on it (`PROTOCOL.md` §4).
- `receipts.py`: seeds, plans, receipts and their trees.
- `codec.py`: the store's encoding (kv/v0).
- `p2.py`: the audit over PoUS's P2-encoded weights store (`PROTOCOL.md` §11): the registration from PoUS's setup,
  receipts of units of four 2,056-byte blocks that the decode kernel sketches as it decodes (`Sketcher`, its CPU
  reference), and openings checked on PoUS's `vk`. Its Lean is `lean/RemoteAudit/P2.lean`, held to it by
  `lean/tests/test_p2.py`, and its session's is `lean/RemoteAudit/P2Session.lean`, of record.
- `p2_party.py`: a P2 session's verifier and its challenger as a party of their own, on a host the developer doesn't
  control: started over ssh with the channel on its stdin and stdout (`--stdio`, as an auditor gate's verb starts it)
  or on a TCP port, it holds the auditor's registration and its plan (the units holding a tensor's block, not the
  store's padding past them, which is PoUS's), holds each session to the auditor's parameters (`SESSION`,
  the served bandwidth piece's: a request carrying other values is refused by name before any coin), drives the
  session on its own clock and coins, and answers
  the verdict with the credited reads (`q_bytes`), the Lean's (`Twin`), with `audit.AuditRun`'s books (`Books`) as its
  veto. `prove` is the prover's side of the channel.
- `p2_lean.py`: the Lean verifier of a P2 session, the process `remote-audit-verifier`, as `p2_party.py` runs
  it: built in `lean/` unless `--lean-exe` names a built one (without lake on the PATH, the one built there), a fresh
  process each session. The standard library only, so the party runs as an auditor gate starts it (`python3 -s -E`).
- `session_parties.py`: `period.session` as two parties of the deployment layer (`verity.configuration.deploy`): the verifier
  and its challenger, on their own clock and coins, which time each root by its arrival (`Received.at`), disclose the
  session's coins at its end (`audit.AuditRun.disclose`) and finish with its answer and case; and the prover, on its
  node's store (`capabilities["store"]`), which checks each challenge it was sent against them (`period.Prover.end`).
- `wire.py`: the openings frame a prover sends a window's openings in, and a window's openings from its frames
  (`openings_of`), beside `lean/RemoteAudit/Wire.lean`'s.
- `memory_map.py`: memory-map/v1, serving's record of the KV pages it wrote and freed, as the audit's appends,
  releases and plan changes, and as the schedule the prover publishes (`schedule`).
- `schedule.py`: the schedule check, without which `ledger` credits nothing after the seed: each pass's plan is the
  verifier's plan of the published schedule at its step, the schedule is within the cap on its bits the session
  committed before its coins (`audit.Params.schedule_cap`), and that cap is the layout's steering
  (`audit.AuditRun.scheduled`, `audit.GammaTerms.steering`). Of record in `lean/RemoteAudit/Schedule.lean`.
- `store.py`: a registered store bound to a pinned model. It reads and checks the pin (`read_pin`, `pin_of`, `check`),
  lays out the arena of a BF16 checkpoint's weights in vLLM's module order and the store of arenas and host filler
  (`arena`, `Store`), and computes the payload root that store has under the verifier's salt key (`expected_root`).
  `Binding.hold` decides a hello against it, the reference beside `lean/RemoteAudit/Store.lean`'s `bind` and `hold`,
  which decide it in the verifier.
  `benchmarks/remote_session/store_layout.py` writes pins and synthetic checkpoints.
- `timing.py`: the honest schedule, the stated claim read through the ledger, and the sketch's rank bound.
- `monitoring.py`: one accepted session's `q_bytes` and `eps` over a week, a month and a year of a deployment's
  sessions, at δ = 1% and 10⁻⁶: pooled by the count bound (all but M − 1 sessions certify), with each session at
  δ/T beside it. The deployment's machines and window are parameters. Its `eps` and δ leave out each session's
  Σ_t adv_t (SHA-512 stream as PRF: not priced, not assumed), which the table says (`EXCEPT`).
- `period.py`: one session a period on one node (`run`, `session`), its core case and its answer.
- `verifier_party.py`: the verifier and its challenger as a party of their own, on a host the developer doesn't control,
  one line of JSON a message and each window's openings in the openings frame (`main`, `period_with`).
- `opening.py`: check 3 through the opening unit (`verity.protocols.poub.verifier.commitments.gates.block_check`) on the honest witness,
  the stand-in for the unit's proofs.
- `pooled.py`: the openings' pooled bound in exact rationals, beside `lean/RemoteAudit/Pooled.lean`.
- `line.py`: the memory line a paced session shows held over time, beside `lean/RemoteAudit/Line.lean`.

Tests: `uv run tools/verity/check/suites.py verity/protocols/poub/verifier` (the Python), and
`verity/protocols/poub/verifier/lean` (the difftest, the replay and the P2 party's sessions on the Lean of
record, which build the Lean).
