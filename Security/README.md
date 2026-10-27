# Security

The Lean statement and proof of what an accepted run of Verity guarantees: for each subprotocol, and then for the composed
protocol, a bound on the probability that the verifier accepts while a claim about the run fails. The terms are the
[Glossary](../ARCHITECTURE.md#glossary)'s (Vocabulary, below).

## Three trees

- `Definitions/` (package `security`, with `Models.lean`, on Mathlib alone): the spec, the part a reader trusts. Each
  area has its vocabulary, its named assumptions (`Assumptions`) and its guarantee statements (`Guarantees`), each a
  `def G : Prop`. The areas: `Core` (circuits, partitions, sound games, server-time blocks and the wipes between them),
  `Hash` (SHA-512, frame-v3 trees, hm96 rows), `Claims` (the definitions every subprotocol's claims read, and C-Flock's
  claims), `Population` (what sampled proofs' guarantee counts), `Accounting` (the device model the resource audits
  share, their joint guarantee, PoUB and the bandwidth audit), `Pouw`, `Pous`, `NetworkCertifier`, `Nci`, and `Flock`
  (C-Flock, the proof system).
- `Properties/` (package `security_properties`): statements about the code's Lean functions as they run, the check each
  subprotocol runs among them, in `Properties.<Protocol>` modules.
- `Proofs/` (package `security_proofs`): every proof, lemma and proof-only definition, with
  `<Area>.SecurityProofs.G : <Area>.Guarantees.G` for each proved guarantee and `Proofs.Properties.<Protocol>` for
  `Properties/`.

What may require what (`tests/test_lean_packages.py`): `security` only Mathlib; `security_properties` `security` and the
code it states properties of, never `security_proofs`; `security_proofs` both, the code and ArkLib; no code package
requires anything here. The lock's `layers` say which spec modules each module may import, and its `trees` refuse a
statement that reads, or a definitions or properties module that imports, anything in `Proofs/`. C-Flock is the
exception until its spec is extracted: some of its guarantees and assumptions are still in `Proofs/Flock`, which `trees`
lists.

## The lock

Each package's `lean-audit.json` is its lock: `Security/lean-audit.json` for the spec, whose guarantees are proved in
`Proofs/` (its `proved_in`), `Properties/lean-audit.json` for the properties, and `Proofs/lean-audit.json`, which
records no guarantee. For each guarantee a lock records the signature, the named assumptions among its hypotheses and a
hash of its type, and, per module, every definition the statements read with a digest. The Lean check
(`tools/verity/lean/check.py`) builds every file, replays every declaration through Lean's kernel, and fails when:

- a declaration uses an axiom other than `propext`, `Classical.choice` and `Quot.sound` (so no `sorry`, `axiom` or
  `native_decide`), or an option that skips the kernel;
- a guarantee has a `Prop` hypothesis that is not a named assumption: a `Prop` listed by name in the lock, with an owner
  and its definition's digest;
- a statement reads a definition outside the spec, or breaks `layers` or `trees`;
- a record no longer matches the build. A changed record means a guarantee now says something else, and is shown when it
  lands; a guarantee that only moved shows none.

## Proved or open

A guarantee is proved when its proof theorem exists, `X.SecurityProofs.G : X.Guarantees.G`, and the lock lists it under
`guarantees`; the check refuses a listed one whose proof holds a `sorry`, so the locks are the index of what is proved.
The lock's `four_names` lists the namespaces held to that form (`Nci`, `NetTiming`, `Accounting`, `Hash`); elsewhere a
proof may still carry another name (`Claims`, `FlockSoundness`, the `Properties.*` statements), and the lock lists it
all the same. A statement under `Guarantees` that no theorem proves is open, and nothing cites it as proved. A named
assumption is a `Prop` under `Assumptions`, listed by name under the lock's `assumptions` and taken as a hypothesis or
read in a statement; an open conjecture enters the same way. Two areas' namespaces differ from their directories:
`Core`'s statements are `Verity.Guarantees.*`, `NetworkCertifier`'s `NetTiming.Guarantees.*`. From the repository root:

~~~text
L='Security/lean-audit.json Security/Properties/lean-audit.json'
jq -r '.guarantees[].signature[0]' $L                                    # every proved guarantee: proof : statement
jq -r '.guarantees[].signature[0]' $L | grep -x '.* : Pous.Period.PeriodHeld'   # no output: the statement is open
git grep -nw 'theorem periodHeld' -- Security/Proofs                     # the proof, by its name's last part
jq -r '.assumptions | keys[]' $L                                         # every named assumption
~~~

## Where each subprotocol's guarantees live

| Subprotocol | Lean | The guarantee to read first, and its status |
|---|---|---|
| Sampled proofs | `Core` (`Verity.Sound`, `Definitions/Core/Game.lean`), `Flock`, `Claims.Sampling`, `Properties.SampledProofs` | `Flock.Guarantees.SampledKSLiveV3`: proved under `UniformRandomBytes`, `RecordCustodyZKJ`, and SHA-512's collision resistance for its finders, `TableCRZ` and `LinkCRZC` |
| PoUW | `Pouw`, `Properties.Pouw` | `Properties.Pouw.Period.H1CheckedCreditPeriod`: proved under `ServiceSound`, `DrawIsCoin`, `SaltIsLiveCoin`, `RegisteredBeforeDraw`, `StagedPins`, `UnitFixed`, `OperandHMu`, `ReceiptLuckAt`, `GuessLuckAt` |
| PoUS | `Pous`, `Properties.Pous` | `Pous.Period.PeriodHeld`: proved under `CoinCommitHides`, `RingDevice`, and `P2RingAuditSecure`, an open conjecture |
| PoUB | `Accounting.Remote`, `Properties.RemoteAudit` (code `RemoteAudit`), being renamed | `Properties.RemoteAudit.Line.LineHeld`: proved under `Accounting.Remote.Held.Assumptions` and `Accounting.Remote.Draw.Assumptions`; outside protocol soundness for now |
| The bandwidth audit over PoUS's store | `Bandwidth.Guarantees` (`Definitions/Accounting/Guarantees/Bandwidth.lean`) | `Bandwidth.Guarantees.AuditRO`: proved, in the random-oracle model |
| Network certification | `NetworkCertifier`, `Properties.NetworkCertifier`, `Properties.CertifierDevice` | `NetTiming.Guarantees.PeriodLeakAccepted`: proved under `TrustedHost`, `Claims.Flock.ListWithin`, and sampled proofs' `Claims.Sampling.KS` and `Claims.Sampling.Commitment.Binds`; each window is bounded alone, with no union over the monitoring period yet |
| Memory wipes | `Verity.Wipe` (`Definitions/Core/Blocks.lean`), `Pous.Guarantees.StreamedWipe`, `Properties.Pous.MemoryChallenger` | `Pous.Guarantees.StreamedWipe`: proved; the PRF enters as its advantage term |
| The inference-only policy | `Nci` (`Nci.Complies`), `Properties.Nci`, `Properties.NciDesc` (code `NciCheck`), being renamed | a policy the verifier checks at registration, not a subprotocol |
| The composite verdict | `Properties.Composition` (code `Core.Composition`) | `Properties.Composition.AcceptMeans`: proved; an accepted composite is each subprotocol's own acceptance, and its commitment binds their digests |

**Protocol soundness, `Verity.Soundness`, is not yet in Lean on `main`.** It is to be one theorem over all the
subprotocols: the verifier accepts while a claim of the monitoring period fails with probability at most their errors
summed, each subprotocol's guarantee entering as a target until its own proof discharges it. Its building block is
`Verity.Sound`, and `Definitions/Core/Guarantees/Game.lean` says how sound games compose. PoUW has other guarantees,
outside protocol soundness.

## The remote protocol

Remote (`Core.Remote.protocol`, [`verity/core/protocols/remote/`](../verity/core/protocols/remote/), whose README states
its guarantee and assumptions) is the composite for an auditor with no devices at the site. It composes two
subprotocols, sampled proofs' one-stage audit and PoUW's window check, and accepts only when both accept and the
commitment binds both (`Properties.Composition`, proved). Onsite adds PoUS, PoUB and network certification.

- **Guarantees:** the table's sampled-proofs and PoUW rows, both proved. The composed statement for the remote
  deployment is not on `main` yet.
- **Assumptions:** those rows' named assumptions. Trust model: nothing the prover runs or hosts is trusted, and every
  coin is live, drawn by the auditor's challenger after the commitment it challenges, with no Fiat–Shamir and no
  retries. Auditor-trusted devices (network certifiers, memory challengers) are onsite's; remote has none at the site.
- **Not PoUB.** PoUB proves the bits a node read from its registered store
  (`verity/core/protocols/remote_bandwidth`); its Lean's "remote" names it, not the remote composite.

## Vocabulary

[GLOSSARY.md](../GLOSSARY.md) holds the notation and the older names. Some Lean names keep an older term until their
renames land: `Nci` is the inference-only policy, `Accounting.Remote` and `RemoteAudit` are PoUB, and `Warden`
(`Definitions/NetworkCertifier/Warden.lean`, `WardenSchedule`) means the network certifier.

## Building and checking

With the toolchain `lean-toolchain` pins:

~~~text
lake build                                    # in Security/: the spec
lake build                                    # in Security/Properties/, then Security/Proofs/ (fetches ArkLib)
python3 tools/verity/lean/check.py Security   # from the repository root
~~~

The check audits `Proofs/` with the spec, from one build, and refuses a machine below its memory floor.
