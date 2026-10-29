# Security

The Lean statement and proof of what an accepted run of Verity guarantees: for each subprotocol, and then for the composed
protocol, a bound on the probability that the verifier accepts while a claim about the run fails. The terms are the
[Glossary](../GLOSSARY.md)'s.

## Three trees

- `Definitions/` (package `security`, with `Models.lean`): the spec, the part a reader reads. Each area (`Core`, `Pouw`,
  `Pous`, `NetworkCertifier`, `Accounting`, ...) has its vocabulary, its named assumptions (`Assumptions`) and its
  guarantee statements (`Guarantees`), each a `def G : Prop`.
- `Properties/` (package `security_properties`): statements about the code's Lean functions, the checks each component
  runs, one `Properties.<Protocol>` module per component.
- `Proofs/` (package `security_proofs`): every proof, `<Area>.SecurityProofs.G : <Area>.Guarantees.G` for each
  guarantee, and the lemmas behind them.

What may import what (`tests/test_lean_packages.py`): `security` requires only Mathlib; `security_properties` requires
`security` and the code it states properties of, never `security_proofs`; `security_proofs` requires both, the code and
ArkLib; and no code package requires anything here. The lock's `layers` say which spec modules each module prefix may
import (the spec imports only itself and Mathlib), and its `trees` forbid a statement from reading, or a definitions or
properties module from importing, anything in `Proofs/`. C-Flock is the exception until its spec is extracted: its
guarantees and assumptions are still in `Proofs/Flock`, which `trees` lists by module.

## The lock

Each package's `lean-audit.json` is its lock. For each guarantee it records the signature, the named assumptions among
its hypotheses and a SHA-256 hash of its type, and, per module, every definition the statements read with a digest. The
Lean check (`tools/verity/lean/check.py`) builds every file, replays every declaration through Lean's kernel, and fails
when:

- a declaration uses an axiom other than `propext`, `Classical.choice` and `Quot.sound` (so no `sorry`, `axiom` or
  `native_decide`), or an option that skips the kernel;
- a guarantee has a `Prop` hypothesis that is not a named assumption: a `Prop` listed by name in the lock, with an owner
  and its definition's digest;
- a statement reads a definition outside the spec, or breaks `layers` or `trees`;
- a claim definition lacks its witnesses: a theorem that it fails at a toy instance, and one per deployed record that its
  checkable antecedents hold there;
- a record no longer matches the build.

A changed record means a guarantee now says something else. When one lands, the change is shown, the statement before
and after and each changed definition with its source; nothing waits on it, and a guarantee that only moved shows none.

## The guarantees to read first

**Protocol soundness, `Verity.Soundness`: not yet in Lean on `main`.** It will be one theorem over all the subprotocols:
the verifier accepts while a claim of the period fails with probability at most their errors summed, each subprotocol's
guarantee entering as a target until that subprotocol's own proof discharges it. The subprotocols are the table's first
six rows, memory wipes among them; PoUB's row is outside it. Neither composite runs the bandwidth audit over PoUS's store
or memory wipes yet, and onsite composes PoUB instead (below). Its building block on `main` is `Verity.Sound`
(`Definitions/Core/Game.lean`).

The table lists what it rests on, then one guarantee outside it; each is proved under `Proofs/`. A status names the statement's named assumptions; the
conditions it places on its inputs and setup, which the lock also lists, are left out.

| Subprotocol | Guarantee | Status |
|---|---|---|
| Sampled proofs | `Flock.Guarantees.SampledKSLiveV3`, stated in `Proofs/Flock/Bridge/Refused.lean`, proved in `Proofs/Flock/Bridge/KSLive.lean` | Proved under named assumptions: `UniformRandomBytes`, `RecordCustodyZKJ`, and SHA-512's collision resistance for its finders, `TableCRZAt` and `LinkCRZC` |
| PoUW | `Properties.Pouw.Period.H1CheckedCreditPeriod`, the instance at the deployed H1 tile unit of the general `CheckedCreditPeriod` (both in `Properties/lean-audit.json`), `Properties/Pouw/Period.lean` | Proved under named assumptions: `ServiceSound`, `DrawIsCoin`, `SaltIsLiveCoin`, `RegisteredBeforeDraw`, `StagedPins`, `UnitFixed`, `OperandHMu`, `ReceiptLuckAt`, `GuessLuckAt`. The deployed path doesn't meet `SaltIsLiveCoin` yet, and no served path meets `AcceptedLive` (credit only after `flock-verify --zk --registered` accepts) |
| PoUS | `Pous.Period.PeriodHeld`, `Properties/Pous/Period.lean` | Proved under named assumptions: `CoinCommitHides`, `RingDevice`, and `P2RingAuditSecure`, an open conjecture |
| Bandwidth audit, PoUB's per-pass lemma | `Accounting.Guarantees.Remote.AuditRO`, `Definitions/Accounting/Guarantees/RemotePass.lean` | Proved, in the random-oracle model |
| Network certification | `NetTiming.Guarantees.PeriodLeakAccepted`, `Definitions/NetworkCertifier/Guarantees/Period.lean` | Proved under named assumptions: `TrustedHost`, `Claims.Flock.ListWithin`, and sampled proofs' `Claims.Sampling.KS` and `Claims.Sampling.Commitment.Binds`; each window is bounded alone, with no union over the period yet |
| Memory wipes | `Pous.Guarantees.StreamedWipe`, `Definitions/Pous/Guarantees/StreamedWipe.lean` | Proved; the PRF enters as its advantage term |
| PoUB | `Properties.RemoteAudit.Line.LineHeld`, `Properties/RemoteAudit/Line.lean` | Outside protocol soundness for now, that audit's own guarantee. Proved under named assumptions: `Accounting.Remote.Held.Assumptions`, `Accounting.Remote.Draw.Assumptions` |

PoUW has other guarantees, outside protocol soundness.

## The remote protocol in Lean

Remote (`verity/core/protocols/remote`, `Core.Remote.protocol`) is the composite for an auditor with no devices at the
site. It composes two subprotocols, sampled proofs' one-stage audit and PoUW's window check, and accepts only when both
accept and the commitment binds both (`Properties.Composition`, proved). Onsite adds PoUS, PoUB and network certification; memory wipes are a component of neither yet.

- **Guarantees:** the table's sampled-proofs and PoUW rows, both proved. The composed statement, `Verity.Soundness`, is
  not in Lean yet (above), so the two compose by hand, and nothing ties the deployed verdicts to the theorems: on
  `main`, remote's sampled-proofs component is one-stage v1, which takes the sessions' verdicts as evidence; the
  verifier's run of `flock-verify --zk` produces them, and the link from remote's acceptance to that run's comes with
  v2.
- **Assumptions:** those rows' named assumptions, `UniformRandomBytes`, `RecordCustodyZKJ`, `TableCRZAt`, `LinkCRZC`,
  `ServiceSound`, `DrawIsCoin`, `SaltIsLiveCoin`, `RegisteredBeforeDraw`, `StagedPins`, `UnitFixed`, `OperandHMu`,
  `ReceiptLuckAt` and `GuessLuckAt`. Trust model: nothing the prover runs or hosts is trusted, and every coin is live,
  with no Fiat–Shamir and no retries. The challenger and the verifier both run on auditor hosts. The challenger, which
  draws each coin after the commitment it challenges, is the one auditor-trusted device remote relies on
  (`trusted-host`); the verifier is the auditor's own check. Two premises about where the challenger runs,
  `CoinsConfidential` and `OneClock`, are prose only (`verity/core/protocols/sampled_proofs/README.md`): no `.lean`
  file defines either. Network certifiers and memory challengers are onsite's.
- **Not PoUB.** `Accounting.Remote` and `Properties.RemoteAudit` are PoUB, the proof of useful bandwidth
  (`verity/core/protocols/remote_bandwidth`), which proves the bits a node read from its registered store. It is a
  different protocol from the remote composite, and a component of onsite only.

## Building and checking

With the toolchain `lean-toolchain` pins:

~~~text
lake build                                    # in Security/: the spec
lake build                                    # in Security/Properties/, then Security/Proofs/ (fetches ArkLib)
python3 tools/verity/lean/check.py Security   # from the repository root
~~~

The check audits `Proofs/` with the spec, from one build, and refuses a machine below its memory floor.
