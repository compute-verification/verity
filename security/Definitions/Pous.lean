module

public import Definitions.Pous.Accounting.GradedMeets
public import Definitions.Pous.Accounting.LemmaA
public import Definitions.Pous.Accounting.Params
public import Definitions.Pous.Accounting.PubMeets
public import Definitions.Pous.Accounting.Rebuild
public import Definitions.Pous.Game.Oracle
public import Definitions.Pous.Game.Prob
public import Definitions.Pous.Model.Audit
public import Definitions.Pous.Model.BandChain
public import Definitions.Pous.Model.BandGraph
public import Definitions.Pous.Model.ColumnGame
public import Definitions.Pous.Model.Continuous
public import Definitions.Pous.Model.Dense
public import Definitions.Pous.Model.FreeBlocks
public import Definitions.Pous.Model.Graded
public import Definitions.Pous.Model.HashChain
public import Definitions.Pous.Model.Inc
public import Definitions.Pous.Model.Invertible
public import Definitions.Pous.Model.Labelling
public import Definitions.Pous.Model.M1
public import Definitions.Pous.Model.M1p
public import Definitions.Pous.Model.P3Chain
public import Definitions.Pous.Model.Pebbling
public import Definitions.Pous.Model.PubEncoder
public import Definitions.Pous.Model.SampledAudit
public import Definitions.Pous.Model.Scheme
public import Definitions.Pous.Model.SegLift
public import Definitions.Pous.Model.StreamedWipe
public import Definitions.Pous.P2.Clip
public import Definitions.Pous.P2.Cost
public import Definitions.Pous.P2.Mutable
public import Definitions.Pous.P2.Rounds

/-!
# PoUS's definitions (trusted)

PoUS, proof of useful space, bounds the memory a prover held over a period. The prover stores `C*`, an encoding of the
committed weights `W` under a salt sampled after `W` is committed. The memory challenger reveals indices of `C*` one at
a time on live coins, and accepts when each answer is correct and on time.

## Read first

The lock's `deployed` guarantees for PoUS:

* `Pous.Period.PeriodHeld` (`Properties.Pous.Period`): over one period, the deployed check accepts while the capacity
  account held fewer than `S · len` bit-ticks with probability at most `δ + ε_max`. It is PoUS's row in
  `security/README.md`, under named assumptions, among them `P2RingAuditSecure`, an open conjecture.
* `Pous.Period.PeriodHeldDeployed` (`Properties.Pous.PeriodDeployed`): the first at the deployed ring budget.
* `Pous.EndToEnd.EndToEndRingDeployed` (`Properties.Pous.EndToEnd`): P2's audit in the ring model at one root's depth,
  `D = 16,446`, `Q = 2^20` and `L = 2^64`. Under named assumptions, a bounded server passes a session with probability
  at most `1% + 2^-128`.
* `Pous.Guarantees.StreamedWipe` (`Definitions.Pous.Guarantees.StreamedWipe`): the streamed-label wipe is a proof of
  secure erasure. It is the memory wipes' row in `security/README.md`.

A scheme's guarantee is the requirement `Pous.Meets` (`Definitions.Pous.Accounting.Params`). The area's other
guarantees are `Pous.Guarantees.X` in `Definitions.Pous.Guarantees` unless another module is named, proved as
`Pous.SecurityProofs.X` in `Proofs.Pous`:

* `BandMultiMeetsFamily`: the band store of 1 to 2^64 segments meets the requirement at `D = 5`, `Q = 2^20`,
  `k = 111` and `ε = 2^-128`, with no hypothesis. `BandPubMeetsFamily` is the same with an untrusted server encoding
  `W`.
* `BandGradedMeetsFamily`: the band store with up to `t` late answers, `(t, k)` from `(0, 111)` to `(3, 243)`.
* `BackgroundRebuildExcluded`: under the memory challenger's limit on an audit's duration, no graded audit lasts long
  enough to re-encode a whole segment from `W`, in the ideal-permutation model.
* `ChainDenseMeets64`: the overwrite-chain dense scheme of 512 labels of 64 KiB meets the requirement.
* `P3ConcreteMeets64` and `P3ConcreteMeets14GB`: P3 with a concrete `H` meets it at `D = 10` and `k = 106`, on one
  segment and on 971 segments jointly.
* `ContinuousAudit` and `BandFamilyContinuousTimeAverage`: the continuous audit's bound, and its time average on the
  band store.
* `StreamedWipeMeetsReads`, `StreamedWipeMeetsServer` and `StreamedWipeMeetsDisk`
  (`Definitions.Pous.Guarantees.StreamedWipeReads`, `.StreamedWipeServer` and `.StreamedWipeDisk`) bound what a
  server keeps of its prior content, at the deployed label.
* `P2SlackFamilyUncond` (`Definitions.Pous.Guarantees.P2`): P2's abstract model on `[0, p)` meets the requirement for
  2^8 to 2^23 labels with no hypothesis, at three slacks `λ`. It is not the deployed P2.

## Assumptions

PoUS's premises are stated in `Definitions.Pous.Assumptions` and `Definitions.Pous.Assumptions.P2`, and its claims in
`Definitions.Pous.Claims.StreamedWipeServer` and `Definitions.Pous.Claims.StreamedWipeDisk`. The lock's named
assumptions are in `Definitions.Assumptions`.

## Map

This module imports the area's definitions, the trusted text its guarantees read. They import no
assumption, guarantee or proof (the lock's `layers`).

* `Game`: bit strings, uniform probability and the idealized computational model.
* `Model`: encoding schemes, the timed audit game and `INC`, the model of independent ideal permutations and that model
  on P2's domain `[0, p)`, and pebbling and data-carrying labels. It also holds the band and dense schemes over the
  overwrite chain, P3's two-layer labels and column game, the sampled, graded and continuous audits, the
  public-encoder check and the streamed-label wipe's game.
* `Accounting`: the requirement at the problem statement's parameters (`Meets`), with a public encoder (`PubMeets`) and
  with late answers (`GradedMeets`); Lemma A's bound; and the background-rebuild budget.
* `P2`: P2's cost model in a generic ring over `Z/p`, its sequential audit's rounds, the mutable store, and what a
  period's fixed extractor cuts and packs.

## Where to look

* Properties of the code: `security/Properties/Pous/`. Proofs: `security/Proofs/Pous.lean` and `security/Proofs/Pous/`.
* Locks: `security/lean-audit.json` (the guarantees) and `security/Properties/lean-audit.json` (the properties).
* The Python reference: `verity/protocols/pous/verifier` (its README). PoUS's grader, `verity/protocols/pous/lean`,
  pins this module and every module under `security/Definitions/Pous` (`TRUSTED.sha256`).
-/

@[expose] public section
