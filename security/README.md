# Security

The Lean statement and proof of what an accepted run of Verity guarantees: for each subprotocol, and then for the composed
protocol, a bound on the probability that the verifier accepts while a claim about the run fails. The terms are the
[Glossary](../GLOSSARY.md)'s. The first section gives the four statements to read first; the next says where the
training slowdown is stated and where each step stands in Lean.

## Start here

Four statements carry what an accepted run guarantees. Each entry gives its file (under `security/`), Lean name and
one-line statement, and its premises by kind: **domain**, what the statement is about; **real**, a named cryptographic
or trust assumption; **folds into**, a second form of another premise; and **open**, a premise about our own code or
protocol that a proof has yet to discharge.

SPOT, the Sample-and-PrOve Transform, takes a proof system Π for one proof block to SPOT[Π], which proves a partitioned
computation by committing to every proof block and proving only the sampled ones; the deployed one is SPOT[Flock]. A
set B of proof blocks is accepted with probability P_a(B), the chance that the sample misses all of it, and the
**tolerated family** is M_δ = {B : P_a(B) > δ} (`Claims.Sampling.Design.M`, the Lean's "confidence set"). F is the
incorrect set of the trace the extractor recovers.

1. **SPOT.** `Claims.SamplingGeneric.CompilerKS` (`Definitions/Claims/SamplingGeneric.lean`, proved in
   `Proofs/Flock/SamplingGeneric.lean`): for every knowledge-sound Π and every compiled verifier meeting SPOT's contract
   with a binding commitment, at every δ ≥ 0, Pr[accept ∧ F ∉ M_δ] ≤ δ + ε_ks + ε_bind, with ε_ks = κ_δ − δ + √(γ·Pr[B])
   from Π's errors at the sampled proof blocks' sessions and ε_bind = Pr[B_ext]; `CompilerZK` carries Π's zero
   knowledge to SPOT[Π]. Their premises (`Arg.KS`, the binding, the scope, the contract) are their domain. At Flock,
   `Flock.Guarantees.SampledKSLiveV3` (stated in `Proofs/Flock/Bridge/ScopeGate.lean`, proved in
   `Proofs/Flock/Bridge/KSLive.lean`) proves the bound at the deployed sessions:
   Pr[`verify --zk` accepts ∧ the run's population is the registration's ∧ F ∉ M_δ] ≤ δ + ε̄_ks + ε_link, at
   δ = C(n−K₀, k)/C(n, k) for k of n proof blocks sampled uniformly, with ε̄_ks Flock's error averaged over the sessions
   and ε_link the link's term:
   - Domain: the circuit and public files parse, with no shared rows, and k ≤ n; `verify --zk` checks the scope, the
     tags and k's split over the tables (`Flock.KsScope.check`).
   - Real: SHA-512 collision resistance (`TableCRZAt` and `LinkCRZCAt` unfold to it) and `UniformRandomBytes` (the
     verifier's bytes, exactly uniform).
   - Folds into: `RecordCustodyZKJPLaw` (the record is live) into Core's challenger premises.
2. **The recursion.** `Claims.RecGeneric.CompilerKS` (`Definitions/Claims/RecursionGeneric.lean`, proved in
   `Proofs/Flock/RecGeneric/OuterProof.lean`): for every Π_in its repetition extractor shows knowledge sound, binding
   ZK server and knowledge-sound Π_out, the compiled verifier is knowledge sound wherever the compiler's contract holds;
   its premises are its domain. Deployed, `Flock.Guarantees.RecursiveKS` (`Proofs/Flock/Recursive/KSStatement.lean`):
   for every decoder within Flock's list-decoding radius and every dense statement, Pr[the recursive verifier accepts ∧
   the extractor recovers no witness] ≤ κ + √(γ·Pr[B]) plus the inner verifier's bad-abort, forked-coin and first-coin
   terms. `Flock.Guarantees.RecursiveKSDeployed` (`Proofs/Flock/Recursive/KSCoinTreeStatement.lean`) is that bound plus
   2·k(x)·δ₁ at the coins the challenger commits at Hello, in an hm96 tree whose leaves hide within δ₁.
   - Domain: the dense statements (`RecDeploy.DenseAt`); the decoders within the radius (`Claims.Flock.ListWithin`; the
     deployed ones meet it).
   - Real: SHA-512 collision resistance bounds Pr[B], and `Claims.Hm96Hiding` bounds δ₁.
   - Open: none.
3. **Zero knowledge of the deployed system.** `Flock.Guarantees.SessionLoopZKDeployed`
   (`Proofs/Flock/SessionLoop/Statement.lean`): at the ZK server's session loop on the deployed tables, for every t > 0,
   every V* with one more round than the verifier and every event E, Pr[E(real)] ≤ 2·(Pr[E(sim)] + 8(t+1)·n_L(x)·δ₁ +
   2·3^(−t) + 4t·3^t·Pr[fork collision]), with δ₁ hm96's hiding gap at the pinned key. For the recursive audit,
   `Flock.Guarantees.RecursiveZK` (`Proofs/Flock/Recursive/Statements.lean`): if the outer session is
   ε_o-zero-knowledge, every event of the view released at a stop from a public family St is at most |St| times as
   likely as at the simulator's, plus ε_o + n_h·δ_{K*}.
   - Domain: a witness in the relation, at a statement the loop serves, in Flock's scope.
   - Real: `Claims.Hm96Hiding` bounds δ₁ and δ_{K*}, and SHA-512 collision resistance the fork term; at the operating
     system's bytes, `UniformRandomBytesTest` (`recursiveZK_os_gap`).
   - Open: `SwCausalOn`, the tables' side's one unproved premise; per worker, `LoopInnerRound0`, `LiveRoundEq` and
     `FillRoundEq`; `Loop.Src` (the loop's inputs from its host are the tape's), the prover-side ZK server's randomness,
     which no party the verifier trusts holds; and `OuterViewZK`.
4. **Composed soundness.** `Verity.SoundnessOf` (`Properties/Verity/Soundness.lean`) at any list of ids, proved
   as `Verity.soundness_at_of_targets` (`Proofs/Properties/Verity/Soundness.lean`); `Verity.Soundness` and
   `Verity.RemoteSoundness` are it at onsite's and remote's (remote's with the two statements on the committed files
   for the inference-only policy's target, `Proofs/Properties/Verity/InferenceOnly.lean`): if each subprotocol meets
   its target, then
   Pr[the composite's verifier accepts files that passed the commit phase's check ∧ some subprotocol's claim fails]
   ≤ Σ_a error_a, where error_a is a's δ plus ε_forge plus its loss at the probability that a's finder outputs a
   SHA-512 collision, and ∞ where the finder or the forger breaks its premise.
   - Real: SHA-512 collision resistance (`Vocab.FinderCR`) and LMS unforgeability (`Hash.Assumptions.LmsUnforgeable`),
     read in the error.
   - Open: the targets (`SpotSound`, `PouwSound`, `InferenceOnlySound`, `PousSound`, `BandwidthAuditSound`,
     `NetworkAttesterSound`, `WipeSound`), each open until proved (PoUW's:
     `verity/protocols/resource_accounting/compute/pouw/verifier/`). Remote: the first two, `WiresSound`, `GateARow`.

**The real assumptions.** Items 1 to 3 rest on three: `Claims.ExpectedCR` (SHA-512 collision resistance, which the
other SHA-512 shapes fold into) and `Claims.Hm96Hiding` (hm96 leaves hide at the pinned key), each under the lock's
`open` until `Definitions/Assumptions.lean` declares it, and the declared `UniformRandomBytesTest` (the operating
system's bytes are uniform up to η on each test). Beside them the statements read `trusted-device`, Core's challenger
premises (`ChallengerLogsCoins`, `FlipsAfterRecorded`, `VerifierReadsLog`) and `Hash.Assumptions.LmsUnforgeable`.

## The training slowdown

Training's statement is [`Definitions/Slowdown/README.md`](Definitions/Slowdown/README.md), in order from circuits up:
a circuit that trains, in the sense of the properties T1 and T2 there, and is also a (c, k, n)-inference circuit has
slowdown nc/|C| at least cν/μ. The inference-only policy is what proves a registered circuit a (c, k, n)-inference
circuit (`InferenceOnly.CompliesAtMost`), composed soundness (Start here, item 4) reads its target with the other
subprotocols', and `Slowdown.Cut.Guarantees.CountHE` is the theorem on that verdict.

### Where each step stands

| Step | Lean | Status |
|---|---|---|
| The actual circuit | `ResourceAccounting.Placed`, `ResourceAccounting.Placed.region` | Defined |
| Registered objects | `Verity.Circuit`, `Verity.Partition`, `Verity.Partition.IsCK`, `Verity.RegisteredCircuit`, `Verity.Committed`; the site and its registration log, `Verity.Site` and `Verity.Registration`; B, a_j and loc_j not yet in Lean | Partly defined |
| Lemma 1 | `Verity.Guarantees.BlockDescription` | Proved |
| Theorem 3 | `Verity.Guarantees.Counting` | Proved |
| Device model | `ResourceAccounting.Placed.ComputeRate`, `ResourceAccounting.Placed.Links`, `ResourceAccounting.Placed.Memory`, `ResourceAccounting.Placed.Causal`, `ResourceAccounting.Slowdown.physClass`; hosts carry no compute, `OneClock` and the inventory not yet in Lean | Defined; three premises unstated |
| The composites | `Core.Remote.protocol`, `Core.Onsite.protocol`, `Properties.Composition.AcceptMeans` | Proved for the product rule `check`, which both run |
| The utilization policy | not yet in Lean | Not a policy yet |
| Protocol soundness | `ResourceAccounting.Slowdown.restricted`; `Verity.soundness_of_subprotocols`, `Verity.remote_soundness_of_subprotocols` | Proved from each subprotocol's target, each a hypothesis until discharged (`Properties/Verity/README.md`) |
| SPOT | `Flock.Guarantees.SampledKSLiveV3`, `Claims.Sampling.RelS` | Proved; the step from F ∈ M_δ to the credit-weighted φ is open |
| PoUW | `ResourceAccounting.Guarantees.Compose.CircuitCredit` | Proved; nothing derives Φ_D from it, or shows each credited gate distinct |
| The inference-only policy | `InferenceOnlyCheck.Properties.SecurityProofs.DescWitnesses`, `InferenceOnly.CompliesRate`, `InferenceOnly.Complies` | Proved at its rate, with each block within (c, k) on the unrolled check's verdict (the lemma `InferenceOnlyCheck.Properties.SecurityProofs.refines`); the step to a (c, k)-partition of the registered circuit itself is open, and k_F (what incorrect proof blocks inject) and k_a (the advice's share) are unbounded |
| The slowdown theorem | `Slowdown.Cut.Guarantees.HE`, `Slowdown.Cut.Guarantees.HEShared`, `Slowdown.Cut.Guarantees.CountHE` | Proved under the named assumptions T1 and T2 (`Slowdown.Cut.Assumptions.Hardness`, `Slowdown.Cut.Assumptions.Exports`), for two of the three patterns of joined words; where the Lean differs from the statement is `Definitions/Slowdown/README.md`'s last table |
| Onsite's k_S | `Pous.Guarantees.StreamedWipe`, `NetTiming.Guarantees.MonitoringPeriodLeakAccepted` | Proved per period; nothing composes them into k_S |

## Three trees

Each tree's README has its import rule, which the lock's `layers` and `trees` hold, and its areas:
[`Definitions/`](Definitions/README.md) (package `security`), the spec; [`Properties/`](Properties/README.md)
(`security_properties`), statements about the code's Lean functions; and [`Proofs/`](Proofs/README.md)
(`security_proofs`), every proof.

## What outside code may do

Outside code may run the Lean check on the packages, read a file here as data (a lock, or Lean a test holds its Python
to) and build or run Lean in a package's Lake environment. No outside package requires them
(`tests/test_lean_packages.py`), and a generated file is written only by the generator its header names. Today:

- Lean files importing `Proofs.*` and the programs that build or run them: the one list is in `Proofs/README.md`.
- The NCP pin's command in `tools/check/pins.py` builds and runs Lean in `security`'s environment.
- Generators: `machine-model lean-headline` and `machine-model --device DEVICE sheet --lean`
  (the second prints `Sheets/`); `tests/verity/circuits/ops/test_lean_fp32_vectors.py`; `fp8atom_vectors.py`,
  `h1t_vectors.py` and `row_prims_vectors.py` in `verity/protocols/resource_accounting/compute/pouw/verifier/lean/scripts/`.

## The lock

Each package's `lean-audit.json` is its lock. What it records and what the Lean check (`tools/verity/lean/check.py`)
refuses are AGENTS.md's (its Lean section). Beside each guarantee's statement and premises, the lock keeps a SHA-256 hash
of its type and, per module, a digest of every definition the statements read, and each listed `Prop` with an owner and
its definition's digest. A named assumption is declared in `Definitions/Assumptions.lean` (`assumptions_module`) or
under it; `Properties`' lock also lists `Properties/Assumptions.lean`, for one stated over the code, which the spec
can't import. The check also fails when:

- a statement reads a definition outside the spec, or breaks `layers` or `trees`;
- a claim definition lacks its witnesses: a theorem that it fails at a toy instance, and one per deployed record that its
  checkable antecedents hold there;
- a record, or a deployed guarantee's `trust_gap` entry, no longer matches the build.

A changed record means a guarantee now says something else. When one lands, the statement before and after is shown
with each changed definition's source; nothing waits on it, and a guarantee that only moved shows none.

## The guarantees to read first

**Protocol soundness** is one theorem (Start here, item 4), the union bound over a composite's subprotocols at its
compiled verifier. Start at `Properties/Verity/README.md`: the statement, the subprotocols each composes, how to read it
and what it rests on.

The table lists the subprotocols' own guarantees, each in its own form; each is proved under `Proofs/`. A status names
the statement's named assumptions, not the conditions on its inputs and setup (the lock lists them).

| Subprotocol | Guarantee | Status |
|---|---|---|
| SPOT | `Flock.Guarantees.SampledKSLiveV3` (Start here, item 1) | Proved under named assumptions: `UniformRandomBytes`, `RecordCustodyZKJ`, and SHA-512's collision resistance (`SHA512CRStrict`, `SHA512CRExpected`) at its finders |
| PoUW | `Properties.Pouw.MonitoringPeriod.H1CheckedCreditMonitoringPeriod`, the instance at the deployed H1 tile unit of the general `CheckedCreditMonitoringPeriod` (both in `Properties/lean-audit.json`), `Properties/Pouw/MonitoringPeriod.lean` | Proved under named assumptions: `ServiceSound`, `SampleIsCoin`, `SaltIsLiveCoin`, `RegisteredBeforeDraw`, `StagedPins`, `UnitFixed`, `OperandHMu`, `ReceiptLuckAt`, `GuessLuckAt`. The deployed path doesn't meet `SaltIsLiveCoin` yet, and no served path meets `AcceptedLive` (credit only after `verity-verify-proof --zk --registered` accepts) |
| PoUS | `Pous.Period.PeriodHeld`, `Properties/Pous/Period.lean`; in the shared coin game, `Pous.Period.PeriodHeldSha512`, `Properties/Pous/PeriodGame.lean` | Proved under named assumptions: the device model `DeviceModel`, the instantiation `Instantiates` (`pous-ring-instantiation`), and `P2RingAuditSecure`, an open conjecture; the coin-game form adds `Hash.Assumptions.Sha512Prf` at the memory attester's two tests, its `ε_prf` in the bound, and names the certificate's fit (`Fits`) and the run's not reading the primitive (`ReadsNoPrimitive`) |
| Bandwidth audit, PoUB's per-pass lemma | `ResourceAccounting.Guarantees.Remote.AuditRO`, `Definitions/Poub/Guarantees/RemotePass.lean` | Proved, in the random-oracle model |
| Network certification | `NetTiming.Guarantees.MonitoringPeriodLeakAccepted`, `Definitions/Certifier/Guarantees/MonitoringPeriod.lean` | Proved under named assumptions: `TrustedDevice`, and SPOT's `Claims.SamplingGeneric.Compiled.KS` and `Claims.Sampling.CommitmentScheme.Binds`; each window is bounded alone, with no union over the period yet |
| Memory wipes | `Pous.Guarantees.StreamedWipe`, `Definitions/Pous/Guarantees/StreamedWipe.lean` | Proved; the PRF enters as its advantage term |
| PoUB | `Properties.Poub.Line.LineHeld`, `Properties/Poub/Remote/Line.lean` | Inside protocol soundness: its P2 audit (`Poub.P2Protocol`, on PoUS's store) is one of onsite's seven subprotocols, so `Verity.soundness_of_subprotocols` at `.onsite` covers it through its target, `BandwidthAuditSound`. In its own form it is proved under named assumptions: `ResourceAccounting.Remote.Held.Assumptions`, `ResourceAccounting.Remote.Sample.Assumptions` |

PoUW has other guarantees, outside protocol soundness.

## The remote protocol in Lean

Remote (`verity/protocols/compose/verifier/remote`, `Core.Remote.protocol`) is the composite for an auditor with no devices at the
site. It composes three subprotocols, SPOT (one-stage v2; the recursive audit refused), PoUW's period check
(`verity/pouw/period/v1`, onsite's work subprotocol too) and the inference-only policy's check (`InferenceOnlyCheck.Protocol`), and
accepts only when all three accept and the commitment binds them (`Properties.Composition`, proved), and only when the
policy and the one-stage audit commit to one record and every execution phase of the period names it as its Commit record
(`Core.Composition.tiedSame`, `Core.Remote.windows`). Onsite holds the same two ties (`Core.Onsite.ties`) and adds PoUS,
PoUB, network certification and memory wipes, with PoUS and PoUB-P2 on one store.

- **Guarantees:** the table's SPOT row and PoUW's period guarantee,
  `Properties.Pouw.MonitoringPeriodJ10R.H1J10RCheckedCreditMonitoringPeriod`, both proved. Neither is stated yet as its target in
  `Verity.remote_soundness_of_subprotocols`, so the two compose by hand. At remote, SPOT's one-stage form is
  v2, which runs `verity-verify-proof --zk` on the sessions' files and reads the challenger's signed coins (v1, which takes
  verdicts as evidence, is refused); where v2 accepts, its session is in V5's event
  (`Verity.Sampled.checkBlobs_event`), which has a coupling.
- **The inference-only policy at remote** (`Properties.Remote.SecurityProofs.acceptInferenceOnly`, proved): remote's
  accept gives `InferenceOnlyHeld`, whose SPOT case is `StageHeld`, the one-stage form's. It has no clause for the
  two-stage form, which the verifier refuses while `SecondStage.FORM` is false; switching it on needs the two-stage
  clause first.
- **Assumptions:** those guarantees' named assumptions, `UniformRandomBytes`, `RecordCustodyZKJ`, `SHA512CRStrict`,
  `SHA512CRExpected`, `TileServiceSound`, `TileSampleIsCoin`, `SaltIsLiveCoin`, `RegisteredBeforeDraw`, `UnitFixed` and
  `SharedNoiseLucks`. The challenger and the verifier both run on auditor devices. The challenger, which
  flips each coin after the commitment it challenges, is the one auditor-trusted device remote relies on
  (`trusted-device`); the verifier is the auditor's own check. Two premises about where the challenger runs are
  onsite's (`verity/protocols/proofs/spot/verifier/README.md`): `CoinsConfidential`, the named Prop
  `Pouw.Assumptions.CoinsConfidential` (`Definitions/Pouw/Assumptions/Dither.lean`, in the lock, taken by
  `Pouw.Guarantees.Dither.FleetSound`), and `OneClock`, prose only.
  Remote's coin path is three assumptions over the challenger's log (`Definitions/Core/Challenger.lean`), stated in
  `Definitions/Core/Assumptions/Challenger.lean` and listed in the lock: the challenger logs every coin it releases
  (`ChallengerLogsCoins`), flips each only after what it samples is recorded (`FlipsAfterRecorded`), and the verifier
  reads coins only from that log (`VerifierReadsLog`). Under them a sample sampled with a logged coin misses a set fixed
  before it at most as often as the sampling design allows (`Verity.Guarantees.SampleAfterRecorded`, proved); none of
  remote's guarantees reads `CoinsConfidential`, and no `.lean` file defines `OneClock`.
- **Not the remote composite.** `ResourceAccounting.Remote` and `Properties.Poub.Remote` are PoUB
  (`verity/protocols/resource_accounting/memory/poub/verifier`), which proves the bits a node read from its registered store; it is a subprotocol
  of onsite only.

## Building and checking

With the toolchain `lean-toolchain` pins:

~~~text
lake build   # in security/: the spec
lake build   # in security/Properties/, then security/Proofs/ (fetches ArkLib)
python3 tools/verity/lean/check.py security/   # from the repository root
~~~

The check audits `Proofs/` with the spec, from one build, and refuses a run that reserved less than its floor, printing
the command at its ask (`python3 tools/verity/lean/check.py --ask security/` prints it).
