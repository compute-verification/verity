module

public import Properties.Verity.Game
public import Properties.Verity.Committed
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# Soundness at the verifier, composed from its subprotocols' targets

The verdicts are not free fields: in the causal game after the parameters, the committed files and the commitment
(`Vocab.game`), subprotocol `a` accepts at composite `k` when the files pass the commit phase's check, `k` accepted what
the prover sent as a `same` composition, and `a`'s compiled check accepts its own case in it (`Vocab.Verdict`). The
event is the verifier's acceptance (`Accepts k`) while the claim of one of `k`'s subprotocols fails
(`Vocab.Holds`, sampled proofs' on the fixed extractor's transcript), with no break in it: each subprotocol's error at a
strategy is its published `δ` plus its collision term at the probability that its fixed finder outputs a SHA-512
collision (`Vocab.error`).

`SoundnessOf` is the one statement, at every composite: each of its subprotocols' targets gives `Vocab.SoundnessAt`,
with the errors summed over its subprotocols. `Soundness` is it at onsite, from its seven targets
(`Vocab.SpotSound`, `PouwSound`, `InferenceOnlySound`, `PousSound`, `BandwidthAuditSound`,
`NetworkCertifierSound`, `WipeSound`), and `RemoteSoundness` the same conclusion at remote's composite of record
(`Core.Remote.V1.protocol`, `verity/remote/v1`), from two targets (sampled proofs, and PoUW, which at remote is
Pearl+'s run verdict and its claim in MACs at the credit its check published) and the two statements that tie the
inference-only policy's clause to the committed files (`Vocab.WiresSound` and `GateARow`,
`Properties.Verity.Committed`). They are proved by `Properties.Composition` (a composite's acceptance is its
subprotocols' and its record's) and the union bound over one strategy in one game.

Each target is a hypothesis until its subprotocol's piece proves it at its compiled check, and `Vocab`'s fields are given
until each piece defines its own: v2's reading of the committed files, the positions, the coin commitment's families and
the device's log each subprotocol read, the extractor and the finders, the gates computed, the world, the wipe and the
play read at its boundary, and the published certificates, thresholds, errors and collision terms. The inference-only
policy's target reads none of them: its claim is on its own case in the play's messages, and
`Proofs.Properties.Verity.InferenceOnly` proves it of the case as sent. The link from that case to the committed files is
two checks the verifier lacks (`Vocab.Holds` names them). At remote the two statements they would establish are
hypotheses in the target's place, each stated and not proved; at onsite the target stays a hypothesis.
-/

@[expose] public section

namespace Verity

open Accounting

/-- **Soundness at every composite's verifier, from its subprotocols' targets.** On any device, for any names
`V` the subprotocols define and any composite `k`, if each of `k`'s subprotocols meets its target at `k` in the causal
game, then at all parameters, committed files and commitments and for every strategy, the probability that `k`'s
compiled verifier accepts files that passed the commit phase's check while the claim of one of `k`'s subprotocols fails
is at most their errors at that strategy summed, each its published `δ` plus its collision term at its finder's
probability of a SHA-512 collision. -/
def SoundnessOf : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α) (k : Verity.Registration.Composite),
    (∀ a ∈ subprotocols k, V.Target k a) → V.SoundnessAt k

/-- **Soundness at onsite's verifier, from its seven subprotocols' targets.** If sampled proofs, PoUW, the
inference-only policy, PoUS, PoUB-P2, the network certifier and the wipe each meet their target at onsite, then the
probability that onsite's compiled verifier accepts files that passed the commit phase's check while one of the
period's claims fails (more than an `ε` share of the units unbound in the extractor's transcript, fewer than `W` gates
computed after their coin, a case of the policy whose Program doesn't meet the policy, fewer than `S` bits held on average,
fewer than `Q` bits read, more than `K` bits through the network certifiers, or foreign content kept across the wipe past
its survivor threshold) is at most the seven errors at that strategy summed. -/
def Soundness : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α),
    V.SpotSound .onsite → V.PouwSound .onsite → V.InferenceOnlySound .onsite → V.PousSound .onsite →
      V.BandwidthAuditSound .onsite → V.NetworkCertifierSound .onsite → V.WipeSound .onsite → V.SoundnessAt .onsite

/-- **Soundness at remote's verifier, from two targets and the two statements on the committed files.** If sampled
proofs and PoUW (`Vocab.PouwSound`: at least `(1 − ρ)(1 − ε) · W` MACs computed on the run's tiles at the credit
Pearl+'s check published) each meet their target at remote, the record's wiring is the wiring the committed units are
held to (`Vocab.WiresSound`, at twice sampled proofs' share and error), and the committed circuit has one gate a row
(`GateARow`), then the probability that remote's compiled verifier accepts files that passed the commit phase's check
while the claim of sampled proofs, PoUW or the inference-only policy fails is at most their three errors at that
strategy summed.

The inference-only policy's target is no hypothesis: its claim is on the case as sent, of which it is proved
(`Vocab.inferenceOnlySound`). The two statements are what that leaves unsaid of the committed files, and neither is
proved. The conclusion's clause for the policy is still the one on the case as sent, and no clause of it reads either
statement: they stand among the hypotheses so that what remote's soundness rests on names them, and a conclusion on
the committed files waits on their proofs.

Discharging `V.PouwSound .remote` reads Pearl+'s properties in `Properties.Pouw.PearlPlus` (`AcceptMeans`, proved as
`acceptMeans`, and `WorkIsTheCredit`, `DrawIsOnItsStreams` and `CoinBytesGiveUniformTickets`, unproved); that is the
Pearl+ target's proof, not this statement's. -/
def RemoteSoundness : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α),
    V.SpotSound .remote → V.PouwSound .remote → V.WiresSound .remote → GateARow → V.SoundnessAt .remote

end Verity
