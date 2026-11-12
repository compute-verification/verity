# Protocol soundness: start here

One theorem says what an accepted run of Verity guarantees, at each composite's compiled verifier: the verifier accepts
while one of its subprotocols' claims fails with probability at most the sum of their errors. A composite is the list of
its subprotocols' ids (`Verity.Registration.Composite`, `security/Definitions/Core/Site.lean`), so the theorem holds at
every list, and onsite and remote are two of them (`Composite.onsite`, `Composite.remote`). It is the union bound over
the composite's subprotocols in one game, from each subprotocol's **target**, which it takes as a hypothesis until that
subprotocol's own proof discharges it. This page says where it is, what it reads and what it rests on.

**Imports.** These modules import what the `Properties.Verity` entry of `layers` lists
(`security/Properties/lean-audit.json`): the composites' compiled code, the definitions and the composition's
properties, never a proof. Their proofs (`security/Proofs/Properties/Verity/`) import them, and so do two readers of
`Sha512.lean`: `security/Proofs/Accounting/ReadUnit/Gates.lean` and `security/Properties/Spot/HistoryTable.lean`.

## The theorem

| Name | Statement | Proof |
|---|---|---|
| `Verity.soundness_at_of_targets : Verity.SoundnessOf` | at every composite `k`, a list of subprotocols' ids, `k`'s subprotocols' targets give `Vocab.SoundnessAt k` | `security/Proofs/Properties/Verity/Soundness.lean` |
| `Verity.soundness_of_subprotocols : Verity.Soundness` | `SoundnessOf` at onsite's list, from its seven targets | the same file |
| `Verity.remote_soundness_of_subprotocols : Verity.RemoteSoundness` | `SoundnessOf` at remote's list, from its three targets | the same file |

The statements are in `Soundness.lean` (this directory). **Remote's**: for any device and any names `V` the subprotocols
define (`Vocab`), if SPOT, PoUW and the inference-only policy each meet their target at remote, then at all
parameters `θ`, committed files `F` and commitments `c` and for every prover strategy, the probability that remote's
compiled verifier (`Core.Remote.protocol`, `verity/remote/v0`) accepts files that passed the commit phase's check while
one of the three claims fails is at most the three errors at that strategy summed. Onsite's is the same statement over
its seven. Each takes its targets by name, which are the target of every subprotocol its list names, `SoundnessOf`'s
hypothesis (`Vocab.targets_remote`, `Vocab.targets_onsite`).

The lock (`security/Properties/lean-audit.json`) records each with no named assumption among its hypotheses, which are
the targets alone, and with its two witnesses at the idle deployment (`Idle.lean`): `*_honest`, every target holds at
`Idle.unbounded`, and `*_nontrivial`, the conclusion fails at `Idle.exact` for a verifier that accepts everything.

## The subprotocols each composes

`Verity.subprotocols k` (`Game.lean`) is the subprotocols composite `k` composes: those whose compiled check's id
(`Subprotocol.id`) `k` lists. A subprotocol's compiled check is the same in every composite (`Subprotocol.protocol`,
`Verity.component`), and the composite's verifier (`Verity.verifier k`) composes the listed ones under the code's one
rule, with each tie between two it lists (`Verity.ties`). At onsite's and remote's lists it is the verifier the code
builds from that list (`Verity.ofIds_onsite`, `Verity.ofIds_remote`,
`security/Proofs/Properties/Verity/Soundness/Composite.lean`). The code builds two more, from two smaller lists of
onsite's subprotocols (`Core.Composites.network`, `Core.Composites.compute`), each under `verity/composite/v0:` and its
list's ids, comma-separated; no lemma here says yet that those two are `Verity.verifier` at their lists. The code
builds no other (`Core.Composites.ofIds`).

| Subprotocol | Target | Compiled check | Remote | Onsite |
|---|---|---|---|---|
| SPOT | `SpotSound` | `VStar.Recursive.Sampled.protocol` (id `verity/sampled-proofs/v1`): one-stage v2 (`Spot.OneStageZk`); one-stage v1 and the recursive audit are refused by name | yes | yes |
| PoUW | `PouwSound` | `PouwCheck.Period.protocol` (`verity/pouw/period/v1`) | yes | yes |
| the inference-only policy | `InferenceOnlySound` | `NciCheck.Protocol.protocol` (`verity/inference-only/v0`) | yes | yes |
| PoUS | `PousSound` | `PousCheck.RunProtocol.protocol` (`verity/pous/p2/v4/run`) | | yes |
| PoUB-P2, on PoUS's store | `BandwidthAuditSound` | `RemoteAudit.P2Protocol.protocol` (`verity/remote-bandwidth/p2/v0`) | | yes |
| the network certifier | `NetworkCertifierSound` | `NetworkCheck.Period.protocol` (`verity/certifier/v0`) | | yes |
| the wipe | `WipeSound` | `MemoryChallenger.WipeProtocol.protocol` (`verity/wipe/v0`) | | yes |

## How to read it

In this order:

1. **The game**, `Verity.Game` (`security/Definitions/Core/Game.lean`): the prover `send`s, the verifier flips a live
   `coin`, the game `ret`urns; a `Strategy` picks every message from the history, and `Game.prob` is the exact
   probability over the coins.
2. **The play** (`Game.lean`): the prover sends `θ`, `F` and `c` before any coin, then `Verity.play`, in which it drives
   the device (`Verity.releases`): each frame carries the bytes a release answers, then the device flips and releases
   its coins, and a refused frame ends the releases; the evidence comes last. `Exchange.sent` is what the composite reads; `Accepts k` is its compiled verifier's verdict on it.
3. **`Vocab`**: the names each subprotocol defines, given until it defines them: v2's reading of the committed files, the
   positions, the coin families, the fixed extractor, each subprotocol's fixed collision finders and LMS forger, the gates
   computed, the world, and the published certificates, `δ`s, `ε_forge` and losses.
4. **The claims and errors**: `Vocab.Holds k θ F c σ a` is subprotocol `a`'s claim (SPOT's on the extractor's
   transcript, the five period claims as `Accounting.PeriodLive.Holds`, the policy's as `Nci.Complies`, the wipe's as
   `Verity.Wipe.OfRun.Kept`'s negation); `Vocab.Verdict` is `a` accepting its own case inside the composite's acceptance;
   `Vocab.error` is `δ + ε_forge + loss` where the finders and forger meet their premises, `⊤` elsewhere.
5. **`Target` and `SoundnessAt`**: `Vocab.Target k a` bounds `Pr[Verdict ∧ ¬ Holds]` by `a`'s error at every
   `θ F c σ`; `Vocab.SoundnessAt k` bounds `Pr[Accepts k ∧ Registered ∧ ∃ a, ¬ Holds]` by the sum. The seven named
   targets are `Target` at one subprotocol each.
6. **The proof** (`security/Proofs/Properties/Verity/Soundness.lean`): `accepts_same` and `accepts_components` (at
   every list, the composite's acceptance is a `same` composition and each subprotocol's on its own case, from
   `Properties.Composition`),
   then `prob_exists_finset_le`, the union bound. `sent_reachable` and `Vocab.accepts_reachable` say the event covers
   every input whose blobs begin with the committed files, so no input shape escapes it.

## Where each target is discharged

- **The five period subprotocols** (SPOT, PoUW, PoUS, PoUB-P2, the network certifier): a target is the
  per-period guarantee `Accounting.PeriodLive.Sound` at `Vocab.live` (`Vocab.target_audit`;
  `security/Definitions/Accounting/JointLive.lean`), so each discharges its target by proving that guarantee.
- **SPOT** at the recursive configuration: `Verity.Recursive.spotRecursive`
  (`security/Proofs/Properties/Verity/Recursive/Target.lean`) proves the target's bound at the recursive configuration's
  verdict (`RecVerdict`) and its own error, under hypotheses that include `live-verifier`; `spotRecursive_of_ks`
  (`security/Proofs/Properties/Verity/Recursive/Audit.lean`) states it at the deployment from
  `Flock.Guarantees.RecursiveKS`. It doesn't yet instantiate `SpotSound`.
- **The others** are hypotheses of the theorem. Each subprotocol's own guarantee, in its own form, is in
  [the guarantees to read first](../../README.md#the-guarantees-to-read-first).

## What it rests on

The targets are the theorem's only hypotheses. Three kinds of assumption enter beside them, by their lock names:

- **SHA-512 collision resistance**, `Claims.ExpectedCR` (`ecr/sha-512`, `security/Definitions/Core/Interaction.lean`;
  `security/lean-audit.json`). It is asked only of each subprotocol's fixed finder: `Vocab.FinderCR` is
  `Claims.ExpectedCR` there, which bounds the finder's collision probability at `ε_cr`, its budget of SHA-512 calls
  over `2^256` (`Vocab.crBound`, `Vocab.finder_calls`), and where it fails the error is `⊤`. Further finders a subprotocol's bound reads (`Vocab.finders`) are held to the bound
  `ecr/sha-512` gives at their own budgets (`Vocab.finderCR`), and a second file with a registered digest enters the
  same way (`Vocab.fileFinder`, `Vocab.publicFinder`).
- **LMS unforgeability**, `Hash.Assumptions.LmsUnforgeable` (`euf-cma/lms-sha512`,
  `security/Definitions/Hash/Assumptions.lean`; `security/Properties/lean-audit.json`). It is asked only of each
  subprotocol's fixed forger at the compiled LMS verifier (`Verity.lmsVerify`, `Vocab.Unforgeable`) and adds
  `ε_forge(t, Q)` to the error. A log a subprotocol's check reads that isn't the device's whole log is such a forgery,
  or a SHA-512 collision of the log's signed chain (`Vocab.Logged`).
- **The auditor's live-coin challenger**, `Verity.Assumptions.ChallengerLogsCoins`, `Verity.Assumptions.FlipsAfterRecorded`
  and `Verity.Assumptions.VerifierReadsLog` (`security/Definitions/Core/Assumptions/Challenger.lean`;
  `security/lean-audit.json`): the challenger logs every coin it releases, flips each only after what it samples is
  recorded, and the verifier reads coins only from that log. The game's device, each release's coins flipped uniformly after
  the bytes they answer, is the challenger they describe. They are stated over the challenger's log
  (`security/Definitions/Core/Challenger.lean`), not over `Verity.Game`, and are not hypotheses of the theorem; what they
  give is `Verity.Guarantees.SampleAfterRecorded` (`security/Proofs/Core/Challenger.lean`). The challenger is an auditor
  device (`trusted-host`, `Verity.Assumptions.TrustedHost`).

Each subprotocol's own proof of its target adds that subprotocol's named assumptions.
