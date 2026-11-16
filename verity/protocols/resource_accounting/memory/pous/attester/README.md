# The memory attester

The memory attester is the auditor-trusted device that runs PoUS's timed rounds and the streamed-label wipe's timed
reads, and signs what it saw. It is the auditor's hardware, on a machine the developer doesn't control at the developer's site
and reached in-region, never a process on the developer's node. Trust in it and its host is the named assumption
`trusted-device`. For the timed rounds it commits its coins in a coin tree before the run; for a wipe it fills the
server's memory with labels it derives itself and flips its coins once the fill is complete. It reads each answer's
receipt time on its own clock and signs the run's records once, under LMS over SHA-512. The verifier checks a signed run
against the commitment released with it and publishes the verdict, and for a wipe the survivor bound it certifies.

This directory holds the device's Python, which is the reference, and its verdicts as executable Lean in
[`lean/`](lean/README.md), difftested against that Python. A deployment starts the device's processes on the auditor's
host; there is nothing here to run by hand.

## Top statements

- `TrustedDevice` (`security/Definitions/Core/Assumptions.lean`): the device runs its stated code on its host, which
  changes none of what it emits or when, and gives it a monotonic clock.
- `MemoryChallenger.Properties.VerdictAccepts` (`security/Properties/Pous/MemoryChallenger.lean`): an accepted timed-rounds
  verdict's run is signed under a trusted key and an accepted scheme, its header is the commitment, each round's
  index opens to its committed coin, and each answer has the expected digest and is on time within `Δ + RTT`.
- `MemoryChallenger.Properties.VerdictSigned` (same file): that run is one the device holding the key ran, recorded
  and timed itself, or the run's own data carries an LMS forgery or a SHA-512 collision.
- `MemoryChallenger.Properties.WipeSurvivorBound` (`security/Properties/Pous/Wipe.lean`): a wipe `wipe-protocol` accepts
  and certifies meets the survivor bound at the `foreign_bits` it publishes, except with `eps` plus the labels' PRF
  advantage plus the network attesters' `δ_net`.
- `Pous.Component.WipeAccepts` (`security/Properties/Pous/Component.lean`): what `wipe-protocol` accepts is a signed
  run, of the parameters' server and run id, that passes every check of `StreamedWipe.verdict`.
- The continuous audit's memory attester (`security/Definitions/Pous/Model/Continuous.lean`) enforces its three rules,
  no pipelining, a hard deadline and every late answer counted (`security/Proofs/Pous/Continuous/Verifier.lean`).

The proofs of the properties are in `security/Proofs/Properties/Pous/` (`MemoryChallenger.lean`, `Wipe.lean`,
`Component.lean`), and the PoUS guarantees the verdicts serve are in `verity/protocols/resource_accounting/memory/pous/verifier/DESIGN.md`.
