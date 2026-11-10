# The memory challenger (`verity/protocols/pous/device`)

The memory challenger is the auditor-trusted device that runs PoUS's timed rounds and the streamed-label wipe's timed
reads, and signs what it saw. It is the auditor's hardware, on a machine the developer doesn't control at the developer's site
and reached in-region, never a process on the developer's node. Trust in it and its host is the named assumption
`trusted-host`. For the timed rounds it commits its coins in a coin tree before the run; for a wipe it fills the
server's memory with labels it derives itself and flips its coins once the fill is complete. It reads each answer's
receipt time on its own clock and signs the run's records once, under LMS over SHA-512. The verifier checks a signed run
against the commitment published before it and publishes the verdict, and for a wipe the survivor bound it certifies.

This directory holds the timed rounds' device in Python (`memory_challenger.py`, the reference the Lean is difftested
against), its party and the messages of a run, the timed audit's process, and the device's verdicts as executable
Lean, the package `lean/` (its README lists the API). The streamed wipe's device is in
`verity/protocols/pous/verifier/` (below).

## Interface

What code outside this directory may use. Everything else here is internal.

| Entry point | What it is |
|---|---|
| `python -m verity.protocols.pous.device.challenger_party --device DIR [--listen HOST:PORT]` | The timed rounds' memory challenger as a process of its own, for the challenger host. It reads one JSON line on stdin. A salt request `{"salt": {"audit", "commitment"}}` it answers with `{"source": HEX}`, logged in its salt log. A run's request `{"binding", "block_bytes", "blocks", "k", "limit_ns", "node"}` it answers by printing `{"listening": PORT}`, serving one prover and printing its report `{"host", "public_key", "session", "session_id", "warmups"}`. A refusal prints `{"refused": REASON}` and exits 1. Its key, session log and salt log are in `DIR`, and it keeps each run it completes in `DIR/setups/`. `challenger_party.PROTOCOL` is the same two parties as a `ProtocolDef` of the deployment layer (`verity.configuration.deploy`). |
| `python verity/protocols/pous/device/timed_audit.py --ready READY.json --leaves LEAVES.bin --out AUDIT.json [OPTIONS]` | The verifier process of the harness's timed audit: a client of a responder that runs `protocol.TimedAudit`'s audits against it (`verity/protocols/pous/verifier/protocol.py`) and writes each audit's record and verdicts to `AUDIT.json`. With `--strict` it refuses a `k` or a limit the certificate doesn't cover. It exits 0 when every audit accepted and 1 otherwise. `--calibrate-only` measures the round trip and writes the limit instead. |
| `memory_challenger` (`memory_challenger.py`) | The memory challenger's timed rounds in Python, the reference its Lean verdict is held to: the device (`MemoryChallenger`, `CoinTree`, `SessionLog`, `LmsSigner`), the records it publishes and signs (`SessionCommitment`, `CoinOpening`, `Round`, `SignedSession`; the module docstring is their formats' spec), `block_digest(block: bytes) -> bytes`, and the check of a signed run, `verdict(signed: SignedSession, commitment: SessionCommitment, openings: Sequence[CoinOpening], expected: Callable[[int], bytes], *, trusted: frozenset[bytes] \| set[bytes], schemes: Mapping[str, Verify] = SCHEMES) -> Verdict`. |
| `challenger_session.session(side: Side, request: Mapping[str, Any], respond: Callable[[int], bytes]) -> dict[str, Any]`, with `side_of`, `request_of`, `session_id` and `placement` (`challenger_session.py`) | The timed rounds from the node's side: it starts the memory challenger's side where `side_of(period: Mapping[str, Any], node: str, device: str \| Path) -> Side` puts it, sends the request, answers each index with `respond(index)` and returns the side's report. The module also holds the messages both sides speak (`MANIFEST`, `Frames`), which `challenger_party` imports. |
| The Lean package `lean/` (`core_memory_challenger`) | The verdicts on what the device signed, as executables, Lean modules and two Python callers. `lean/README.md` states its interface. |

The streamed wipe's Python is not here: its device and verdict (`streamed_wipe.py`) and its process (`wipe_party.py`)
are in `verity/protocols/pous/verifier/`, entry points of that directory's README. The two directories import each
other: the modules here read the verifier's parameters, schemes, setup and run records, and the verifier's run and
wipe read `memory_challenger` and `challenger_session`.

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

- `verity/protocols/pous/device/memory_challenger.py`: the timed rounds' device (`MemoryChallenger`, `CoinTree`,
  `SessionLog`, `LmsSigner`, `verdict`); its module docstring is the record formats' spec.
- `verity/protocols/pous/verifier/streamed_wipe.py`: the streamed-label wipe's device (`Challenger`) and `verdict`.
- `verity/protocols/pous/device/challenger_party.py` and `verity/protocols/pous/verifier/wipe_party.py`: each as a party
  of its own on the challenger's host; `verity/protocols/pous/verifier/wipe_parties.py`, the wipe's parties on the party
  runtime (`verity/configuration/deploy`).
- `lean/tests/`: the difftests of both verdicts and of the record formats against the Python.
