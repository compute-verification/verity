# The memory challenger (`verity/protocols/pous/device`)

The memory challenger is the auditor-trusted device that runs PoUS's timed rounds and the streamed-label wipe's timed
reads, and signs what it saw. It is the auditor's hardware, on a machine the developer doesn't control at the developer's site
and reached in-region, never a process on the developer's node. Trust in it and its host is the named assumption
`trusted-host`. For the timed rounds it commits its coins in a coin tree before the run; for a wipe it fills the
server's memory with labels it derives itself and flips its coins once the fill is complete. It reads each answer's
receipt time on its own clock and signs the run's records once, under LMS over SHA-512. The verifier checks a signed run
against the commitment published before it and publishes the verdict, and for a wipe the survivor bound it certifies.

This directory holds the device's Python, which is the reference, and its verdicts as executable Lean, the package
`lean/` (its README lists the API), difftested against that Python.

## Interface

What code outside this directory may use. Everything else here is internal.

| Entry point | What it is |
|---|---|
| `python -m verity.protocols.pous.device.challenger_party --device DIR [--listen HOST:PORT]` | The timed rounds' memory challenger as a process of its own, for the challenger host. It reads one JSON line on stdin. A salt request `{"salt": {"audit", "commitment"}}` it answers with `{"source": HEX}`, logged in its salt log. A run's request `{"binding", "block_bytes", "blocks", "k", "limit_ns", "node"}` it answers by printing `{"listening": PORT}`, serving one prover and printing its report `{"host", "public_key", "session", "session_id", "warmups"}`. A refusal prints `{"refused": REASON}` and exits 1. Its key, session log and salt log are in `DIR`, and it keeps each run it completes in `DIR/setups/`. `challenger_party.PROTOCOL` is the same two parties as a `ProtocolDef` of the deployment layer (`verity.configuration.deploy`). |
| `python verity/protocols/pous/device/timed_audit.py --ready READY.json --leaves LEAVES.bin --out AUDIT.json [OPTIONS]` | The verifier process of the harness's timed audit: a client of a responder that runs `protocol.TimedAudit`'s audits against it (PoUS's `protocol.py`) and writes each audit's record and verdicts to `AUDIT.json`. With `--strict` it refuses a `k` or a limit the certificate doesn't cover. It exits 0 when every audit accepted and 1 otherwise. `--calibrate-only` measures the round trip and writes the limit instead. |
| The Lean package `lean/` (`core_memory_challenger`) | The verdicts on what the device signed, as executables, Lean modules and two Python callers. `lean/README.md` states its interface. |

From Python there are four modules, each imported by its path.
`memory_challenger` is the timed rounds' device and its records: the device and its coin tree (`MemoryChallenger`,
`CoinTree`, `CoinOpening`, `verify_coin`), what it publishes and signs (`SessionCommitment`, `SignedSession`, `Round`,
`records_root`, `message`), the check of a signed run (`verdict`, `Verdict`), its two logs (`SessionLog`, `SaltLog`), its
signature schemes (`Signer`, `Verify`, `LmsSigner`, `SCHEMES`, `LMS`, `TEST_PREFIX`) and a block's digest
(`block_digest`), with the formats' names and constants. `challenger_session` is a run of timed rounds between a node and
the challenger's host: the two sides and their frames, the request and its ids (`request_of`, `session_id`, `placement`,
`side_of`), the node's calls (`salt`, `session`, `answer`) and the setups the host keeps (`setup_of`, `SETUPS`).
`challenger_party` is the challenger's side of that run: `serve`, `give_salt`, `designate`, `keep`, `report`, `Refusal`,
`PROTOCOL` and `SOURCE_BYTES`, the byte length of a salt's source. `timed_audit` is the timed audit's client and its
record's verdicts: the wire (`OP_CHALLENGE`, `OP_PING`, `OP_CLOSE`, `HEADER`), `audit`, `calibrate`, `judge`, `verdict`
and `lean_verdict`, and the clocks, counters and parameters a harness reads beside them.

Two files are named `timed_audit.py`, and they are two things. This directory's is the timed audit as the memory
challenger runs it: the client of a responder's wire, the record it writes and `verdict`, the verdict on that record in
Python, which is the reference. `verity/protocols/pous/verifier/check/timed_audit.py` is the Python beside PoUS's Lean
check: its `lean_verdict` gives the same verdict by the Lean executable `pous-timed-audit`, and its two commands hold
the Lean to the reference, `--generated` on records it builds and `--replay RUN_DIR` on a run's. This directory's
`lean_verdict` calls that one and loads its file by path (`pous_check`), the one load of a file by path in the device's
modules. An import by name cannot reach it: the Lean check is a tree with an import root of its own (`SEPARATE` in
`tests/test_boundaries.py`) that reads `verity_difftest`, and `verity` imports no other distribution. The check's other
Python is `verity/protocols/pous/verifier/check/interop.py --all`, which holds the Lean to the P2 wire's Python
reference on every vector, and `verity/protocols/pous/verifier/check/scripts/difftest_vectors.py`, which writes the
difftest's instances with the Python's answers on them and which the Lean check runs.

The streamed-label wipe's Python (`streamed_wipe.py`, and the wipe's process `wipe_party.py`) is PoUS's verifier's, an
entry point of `verity/protocols/pous/verifier/README.md`.

## Top statements

- `TrustedHost` (`security/Definitions/Core/Assumptions.lean`): the device runs its stated code on its host, which
  changes none of what it emits or when, and gives it a monotonic clock.
- `MemoryChallenger.Properties.VerdictAccepts` (`security/Properties/Pous/MemoryChallenger.lean`): an accepted timed-rounds
  verdict's run is signed under a trusted key and an accepted scheme, its header is the commitment, each round's
  index opens to its committed coin, and each answer has the expected digest and is on time within `Δ + RTT`.
- `MemoryChallenger.Properties.VerdictSigned` (same file): that run is one the device holding the key ran, recorded
  and timed itself, or the run's own data carries an LMS forgery or a SHA-512 collision.
- `MemoryChallenger.Properties.WipeSurvivorBound` (`security/Properties/Pous/Wipe.lean`): a wipe `wipe-protocol` accepts
  and certifies meets the survivor bound at the `foreign_bits` it publishes, except with `eps` plus the labels' PRF
  advantage plus the network certifiers' `δ_net`.
- `Pous.Component.WipeAccepts` (`security/Properties/Pous/Component.lean`): what `wipe-protocol` accepts is a signed
  run, of the parameters' server and run id, that passes every check of `StreamedWipe.verdict`.
- The continuous audit's memory challenger (`security/Definitions/Pous/Model/Continuous.lean`) enforces its three rules,
  no pipelining, a hard deadline and every late answer counted (`security/Proofs/Pous/Continuous/Verifier.lean`).

The proofs of the properties are in `security/Proofs/Properties/Pous/` (`MemoryChallenger.lean`, `Wipe.lean`,
`Component.lean`), and the PoUS guarantees the verdicts serve are in `verity/protocols/pous/verifier/README.md`.

## Where to look

- `memory_challenger.py`: the timed rounds' device (`MemoryChallenger`, `CoinTree`, `SessionLog`, `LmsSigner`,
  `verdict`); its module docstring is the record formats' spec.
- `challenger_session.py` and `challenger_party.py`: a run of timed rounds, from the node's side and as the challenger's
  party on its own host.
- `timed_audit.py`: the timed audit's client, its record and the two verdicts on it.
- `verity/protocols/pous/verifier/streamed_wipe.py`: the streamed-label wipe's device (`Challenger`) and `verdict`;
  `wipe_party.py` beside it, the wipe as a party of its own on the challenger's host, and `wipe_parties.py`, the wipe's
  parties on the party runtime (`verity/configuration/deploy`).
- `lean/tests/`: the difftests of both verdicts and of the record formats against the Python.
