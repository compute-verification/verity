# The memory attester: design

## Interface

What code outside this directory may use. Everything else here is internal.

| Entry point | What it is |
|---|---|
| `python -m verity.protocols.resource_accounting.memory.pous.attester.challenger_party --device DIR [--listen HOST:PORT]` | The timed rounds' memory attester as a process of its own, for the memory attester's host. It reads one JSON line on stdin. A salt request `{"salt": {"audit", "commitment"}}` it answers with `{"source": HEX}`, logged in its salt log. A run's request `{"binding", "block_bytes", "blocks", "k", "limit_ns", "node"}` it answers by printing `{"listening": PORT}`, serving one prover and printing its report `{"host", "public_key", "session", "session_id", "warmups"}`. A refusal prints `{"refused": REASON}` and exits 1. Its key, session log and salt log are in `DIR`, and it keeps each run it completes in `DIR/setups/`. `challenger_party.PROTOCOL` is the same two parties as a `ProtocolDef` (`verity.protocol`), which the deployment layer (`verity.runtime`) runs. |
| `python verity/protocols/resource_accounting/memory/pous/attester/timed_audit.py --ready READY.json --leaves LEAVES.bin --out AUDIT.json [OPTIONS]` | The verifier process of the harness's timed audit: a client of a responder that runs `protocol.TimedAudit`'s audits against it (PoUS's `protocol.py`) and writes each audit's record and verdicts to `AUDIT.json`. With `--strict` it refuses a `k` or a limit the certificate doesn't cover. Every P2 v4 document is judged by the Lean, `pous-timed-audit` (at `--lean-exe PATH`, else `pous_check`'s), whose verdict is the record's: where it can't run, every audit is undecided and no profile is issued. A scheme no Lean judges (the band's) is measured, its verdict `unjudged`, with no profile. It exits 0 when the Lean accepted every audit and 1 otherwise. `--calibrate-only` measures the round trip and writes the limit instead. |
| `python -m verity.protocols.resource_accounting.memory.pous.attester.wipe_challenger --device DIR [--listen HOST:PORT] [--reads K] [--streamer build\|PATH [--threads T]]` | The streamed wipe's memory attester as a process of its own, for the memory attester's host. It reads the request `{"labels", "limit_ns", "node", "tiers"}` as one JSON line on stdin, prints `{"listening": PORT}`, serves one prover and prints its report. `wipe_challenger.PROTOCOL` is the same two parties for the deployment layer. |
| The Lean package `lean/` (`core_memory_challenger`) | The verdicts on what the device signed, as executables, Lean modules and two Python callers. `lean/README.md` states its interface. |

From Python there are eight modules, each imported by its path.
`memory_challenger` is the timed rounds' device and its records: the device and its coin tree (`MemoryChallenger`,
`CoinTree`, `CoinOpening`, `verify_coin`), what it publishes and signs (`SessionCommitment`, `SignedSession`, `Round`,
`records_digest`, `message`), the check of a signed run (`verdict`, `Verdict`), its two logs (`SessionLog`, `SaltLog`), its
signature schemes (`Signer`, `Verify`, `LmsSigner`, `SCHEMES`, `LMS`, `TEST_PREFIX`) and a block's digest
(`block_digest`), with the formats' names and constants. `challenger_session` is a run of timed rounds between a node and
the memory attester's host: the two sides and their frames, the request and its ids (`request_of`, `session_id`, `placement`),
the side it is handed (`Side`, `command`), the loop that steps its parties (`run`), the node's salt request (`salt`)
and the setups the host keeps (`setup_of`, `SETUPS`); the prover's side is `pous/prover/challenger_prover.py`.
`challenger_party` is the memory attester's side of that run: `serve`, `give_salt`, `designate`, `keep`, `report`, `Refusal`,
`PROTOCOL` and `SOURCE_BYTES`, the byte length of a salt's source. `timed_audit` is the timed audit's client and its
record's verdicts: the wire (`OP_CHALLENGE`, `OP_PING`, `OP_CLOSE`, `HEADER`), `audit`, `calibrate`, `judge`, `accepted`,
`verdict`, `lean_verdict`, `run_lean` and `lean_judges`, and the clocks, counters and parameters a harness reads beside them.
`continuous` is the continuous audit's memory attester: `ContinuousMemoryChallenger`, `Reply` and `run_continuous`.
`onsite` is the memory attester's key as a deployment starts it: `device_key`, `signer` and `pin`. `wipe_challenger` is the
streamed wipe's memory attester: `WipeMemoryChallenger`, one session over a socket (`serve`), its report (`report`),
its party (`MemoryChallengerParty`, `start`, `PROTOCOL`) and the command (`main`). `wipe_stream` is the wipe's native
fill on the memory attester's CPUs (`Streamer`, over `wipe_stream.c`).

What leaves the memory attester's host goes into the node that started its party. `challenger_party` answers on two
channels only: its stdout, which goes to the node that started it (the salt's source, `{"listening": PORT}`, the report
with the release, or a refusal), and the one connection its prover makes (the indices and the end frame). A refusal
names the field and the rule the request breaks and quotes none of the request's bytes (`Refusal`, `setup_of`). Nothing
else of a party leaves its host: its key, session log, salt log and kept setups stay in `DIR`; the auditor device's gate
keeps each party's log for root on the host and serves it to no ssh caller; and it starts a party only for a caller
whose address is a registered node's (`verity/protocols/resource_accounting/memory/poub/attester/live_verifier/README.md`).

Two files are named `timed_audit.py`, and they are two things. This directory's is the timed audit as the memory
attester runs it: the client of a responder's wire, the record it writes and `verdict`, the verdict on that record in
Python, which is the reference. `verity/protocols/resource_accounting/memory/pous/verifier/check/timed_audit.py` is the Python beside PoUS's Lean
check: its `lean_verdict` gives the same verdict by the Lean executable `pous-timed-audit`, and its two commands hold
the Lean to the reference, `--generated` on records it builds and `--replay RUN_DIR` on a run's. This directory's
`lean_verdict` runs an executable it is given (`run_lean`), else calls that one and loads its file by path (`pous_check`), the one load of a file by path in the device's
modules. An import by name cannot reach it: the Lean check is a tree with an import root of its own (`distribution` in
`tests/import_order.toml`) that reads `verity_difftest`, and `verity` imports no other distribution. The check's other
Python is `verity/protocols/resource_accounting/memory/pous/verifier/check/interop.py --all`, which holds the Lean to the P2 wire's Python
reference on every vector, and `verity/protocols/resource_accounting/memory/pous/verifier/check/scripts/difftest_vectors.py`, which writes the
difftest's instances with the Python's answers on them and which the Lean check runs.

The streamed-label wipe's records and verdict (`streamed_wipe.py`) and the messages both its sides speak
(`wipe_party.py`) are PoUS's verifier's, an entry point of `verity/protocols/resource_accounting/memory/pous/verifier/DESIGN.md`.

## Where to look

- `memory_challenger.py`: the timed rounds' device (`MemoryChallenger`, `CoinTree`, `SessionLog`, `LmsSigner`,
  `verdict`); its module docstring is the record formats' spec.
- `challenger_session.py` and `challenger_party.py`: a run of timed rounds, from the node's side and as the memory attester's
  party on its own host.
- `timed_audit.py`: the timed audit's client, its record and the two verdicts on it.
- `continuous.py`: the continuous audit's memory attester, which enforces the three rules above.
- `wipe_challenger.py`: the streamed wipe's memory attester as a party of the party runtime (`verity/runtime`), over
  `verity/protocols/resource_accounting/memory/pous/verifier/wipe_parties.py`'s messages; the prover's party is
  `verity/protocols/resource_accounting/memory/pous/prover/wipe_prover.py`.
- `onsite.py`: the memory attester's key as a deployment starts it (`ProtocolDef.starts`, `challenger_party.start`,
  `wipe_challenger.start`), refused unless the configuration pins it for the protocol it signs for.
- `verity/protocols/resource_accounting/memory/pous/verifier/streamed_wipe.py`: the streamed-label wipe's device (`Challenger`) and `verdict`;
  `wipe_party.py` beside it, the wipe as a party of its own on the memory attester's host.
- `lean/tests/`: the difftests of both verdicts and of the record formats against the Python.
