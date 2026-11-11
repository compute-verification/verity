# The memory challenger (`verity/protocols/pous/device`)

The memory challenger is the auditor-trusted device that runs PoUS's timed rounds and the streamed-label wipe's timed
reads, and signs what it saw. It is the auditor's hardware, on a machine the developer doesn't control at the developer's site
and reached in-region, never a process on the developer's node. Trust in it and its host is the named assumption
`trusted-host`. For the timed rounds it commits its coins in a coin tree before the run; for a wipe it fills the
server's memory with labels it derives itself and flips its coins once the fill is complete. It reads each answer's
receipt time on its own clock and signs the run's records once, under LMS over SHA-512. The verifier checks a signed run
against the commitment published before it and publishes the verdict, and for a wipe the survivor bound it certifies.

This directory holds the device's verdicts as executable Lean, the package `lean/` (its README lists the API). The
device's Python is in `verity/protocols/pous` (below), the reference the Lean is difftested against.

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
- `verity/protocols/pous/device/challenger_party.py` and `wipe_party.py`: each as a party of its own on the challenger's
  host; `wipe_parties.py`, the wipe's parties on the party runtime (`verity/configuration/deploy`).
- `lean/tests/`: the difftests of both verdicts and of the record formats against the Python.
