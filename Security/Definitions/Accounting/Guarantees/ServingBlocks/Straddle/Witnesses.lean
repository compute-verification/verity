import Definitions.Accounting.Guarantees.ServingBlocks.Straddle
import Definitions.Accounting.ServingBlocks.Straddle.Instances

/-!
# The uncertified cap across the certified boundary: honest and nontrivial lemmas

Each guarantee of `Accounting.Guarantees.ServingBlocks.Straddle` has its statements here, at the instances of
`Security/Definitions/Accounting/ServingBlocks/Instances.lean` and
`Security/Definitions/Accounting/ServingBlocks/Straddle/Instances.lean`. `<Guarantee>_honest`: its premises hold together
at `Honest`, at its dark remainder where they read one. `<Guarantee>_nontrivial`: at an explicit circuit every premise
but one named premise holds, that one fails, and the weight is above the bound, so the guarantee is false without it.

`UncertifiedStructural_nontrivial` is the attack on no straddling: a counted operation whose first gate is a certified
product and whose representative is an uncertified accumulation that reads it. Every premise of `UncertifiedCrossing`
holds there, so it also shows that the replacing cap needs its crossing term. `StraddleBound_nontrivial_overlap` shows
the straddle bound attained there at `μ = 1`.

The proofs are in `Security/Proofs/Accounting/ServingBlocks/Straddle/Witnesses.lean`.
-/

namespace Accounting.Guarantees.ServingBlocks

open Verity Finset Accounting.ServingBlocks

/-- **The structural cap's premises hold** at `Honest`'s dark remainder, with no blocks and `ω_max = μ = s = 1`: its one
operation there, gate 3's, is computed by gate 3 alone. -/
def UncertifiedStructural_honest : Prop :=
  0 < 1 ∧ (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Honest.J.V, Honest.J.ω v ≤ 1) ∧ Overlap Honest.J 1 ∧ BodySize Honest.J 1 ∧
  NoStraddle Honest.J ((univ \ Honest.S).biUnion Honest.P.gates) (Fin.elim0 : Fin 0 → Finset (Fin Honest.d.D.N))

/-- **The structural cap is false without no straddling**: at `Straddle` read as one server with gate 1 its inference
block, with `ω_max = μ = 1` and `s = 2`, the operation gate 2 represents is computed by the certified gate 1 and the
uncertified gate 2. Every weight is at most 1, every gate is in at most one body, every body has two gates and leads to
its representative; the operation straddles, and its weight `1` is above `ω_max μ |U| / s = 1/2`. -/
def UncertifiedStructural_nontrivial : Prop :=
  0 < 2 ∧ (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Straddle.J.V, Straddle.J.ω v ≤ 1) ∧ Overlap Straddle.J 1 ∧ BodySize Straddle.J 2 ∧
  OperationBodies Straddle.d Straddle.J ∧ ¬ NoStraddle Straddle.J Straddle.server Straddle.certified ∧
  ¬ UncertifiedCap Straddle.J Straddle.server Straddle.certified
    (1 * (1 : ℕ) * (uncertified Straddle.server Straddle.certified).card / (2 : ℕ))

/-- **The straddle bound's premises hold** at `Honest`, with `ω_max = μ = 1`. -/
def StraddleBound_honest : Prop :=
  OperationBodies Honest.d Honest.J ∧ Overlap Honest.J 1 ∧ (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Honest.J.V, Honest.J.ω v ≤ 1)

/-- **The straddle bound is false without operation bodies**: at `Detached` read as one server with gate 1 its inference
block, with `ω_max = μ = 1`, the operation gate 2 represents takes in gate 1, which no gate of it reads. Every gate is in
at most one body and every weight is at most 1; no unregistered bit crosses the certified boundary, yet the operation
straddles with weight `1`. -/
def StraddleBound_nontrivial : Prop :=
  ¬ OperationBodies Detached.d Detached.J ∧ Overlap Detached.J 1 ∧ (0 : ℝ) ≤ 1 ∧
  (∀ v ∈ Detached.J.V, Detached.J.ω v ≤ 1) ∧
  1 * (1 : ℕ) * (crossing Detached.d Detached.server Detached.certified : ℝ) <
    Detached.J.straddle Detached.server Detached.certified

/-- **The straddle bound is false without overlap, and attained with it**: at `Straddle` read as one server with gate 1
its inference block, with `ω_max = 1`, the bodies lead to their representatives and every weight is at most 1. At
`μ = 1` the straddling weight `1` equals `ω_max μ` times the one bit crossing, gate 1 into gate 2; no gate is in at most
`0` bodies, and at `μ = 0` the weight is above the bound `0`. -/
def StraddleBound_nontrivial_overlap : Prop :=
  OperationBodies Straddle.d Straddle.J ∧ Overlap Straddle.J 1 ∧ ¬ Overlap Straddle.J 0 ∧ (0 : ℝ) ≤ 1 ∧
  (∀ v ∈ Straddle.J.V, Straddle.J.ω v ≤ 1) ∧
  Straddle.J.straddle Straddle.server Straddle.certified =
    1 * (1 : ℕ) * (crossing Straddle.d Straddle.server Straddle.certified : ℝ) ∧
  1 * (0 : ℕ) * (crossing Straddle.d Straddle.server Straddle.certified : ℝ) <
    Straddle.J.straddle Straddle.server Straddle.certified

/-- **The crossing cap's premises hold** at `Honest`, with `ω_max = μ = s = 1`. -/
def UncertifiedCrossing_honest : Prop :=
  0 < 1 ∧ OperationBodies Honest.d Honest.J ∧ Overlap Honest.J 1 ∧ BodySize Honest.J 1 ∧ (0 : ℝ) ≤ 1 ∧
  (∀ v ∈ Honest.J.V, Honest.J.ω v ≤ 1)

/-- **The crossing cap is false without body size**: at `Counter`'s one gate with no blocks, with `ω_max = μ = 1` and
`s = 2`, the operation is computed by its one gate, which reads only a registered gate. Its body leads to its
representative and every gate is in at most one body; no body has two gates, and the weight `1` is above
`ω_max μ (|U| / s + crossing) = 1/2`. -/
def UncertifiedCrossing_nontrivial : Prop :=
  0 < 2 ∧ OperationBodies Counter.d Counter.J ∧ Overlap Counter.J 1 ∧ ¬ BodySize Counter.J 2 ∧ (0 : ℝ) ≤ 1 ∧
  (∀ v ∈ Counter.J.V, Counter.J.ω v ≤ 1) ∧
  ¬ UncertifiedCap Counter.J (Counter.P.gates 0) (Fin.elim0 : Fin 0 → Finset (Fin Counter.d.D.N))
    (1 * (1 : ℕ) * ((uncertified (Counter.P.gates 0) (Fin.elim0 : Fin 0 → Finset (Fin Counter.d.D.N))).card /
      (2 : ℕ) + crossing Counter.d (Counter.P.gates 0) (Fin.elim0 : Fin 0 → Finset (Fin Counter.d.D.N))))

/-- **The crossing cap is false without operation bodies**: at `Detached` read as one server with gate 1 its inference
block, with `ω_max = μ = 1` and `s = 2`, every gate is in at most one body and every body has two gates; no unregistered
bit crosses the certified boundary, and the weight `1` is above `ω_max μ (|U| / s + crossing) = 1/2`. -/
def UncertifiedCrossing_nontrivial_bodies : Prop :=
  0 < 2 ∧ ¬ OperationBodies Detached.d Detached.J ∧ Overlap Detached.J 1 ∧ BodySize Detached.J 2 ∧ (0 : ℝ) ≤ 1 ∧
  (∀ v ∈ Detached.J.V, Detached.J.ω v ≤ 1) ∧
  ¬ UncertifiedCap Detached.J Detached.server Detached.certified
    (1 * (1 : ℕ) * ((uncertified Detached.server Detached.certified).card / (2 : ℕ) +
      crossing Detached.d Detached.server Detached.certified))

/-- **The local cap's premises hold** at `Honest`'s dark remainder, with no blocks and `ω_max = μ = s = 1`. -/
def UncertifiedLocal_honest : Prop :=
  0 < 1 ∧ OperationBodies Honest.d Honest.J ∧ Overlap Honest.J 1 ∧
  UncertifiedBodySize Honest.J ((univ \ Honest.S).biUnion Honest.P.gates)
    (Fin.elim0 : Fin 0 → Finset (Fin Honest.d.D.N)) 1 ∧
  (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Honest.J.V, Honest.J.ω v ≤ 1)

/-- **The local cap is false without uncertified body size**: at `Straddle` read as one server with gate 1 its inference
block, with `ω_max = μ = 1` and `s = 2`, the bodies lead to their representatives and every gate is in at most one body;
the operation gate 2 represents has one gate outside the block, and its weight `1` is above
`ω_max μ (|U| / s + Σ_i |in(B i) \ θ|) = 1/2`. -/
def UncertifiedLocal_nontrivial : Prop :=
  0 < 2 ∧ OperationBodies Straddle.d Straddle.J ∧ Overlap Straddle.J 1 ∧
  ¬ UncertifiedBodySize Straddle.J Straddle.server Straddle.certified 2 ∧ (0 : ℝ) ≤ 1 ∧
  (∀ v ∈ Straddle.J.V, Straddle.J.ω v ≤ 1) ∧
  ¬ UncertifiedCap Straddle.J Straddle.server Straddle.certified
    (1 * (1 : ℕ) * ((uncertified Straddle.server Straddle.certified).card / (2 : ℕ) +
      ∑ i, (Straddle.d.unregistered (Straddle.certified i)).card))

/-- **The local cap is false without operation bodies**: at `Detached` read as one server with gate 2, the operation's
representative, its inference block, with `ω_max = μ = 1` and `s = 2`, no operation's representative is outside the
block and every gate is in at most one body; the block reads no unregistered bit, and the weight `1` is above
`ω_max μ (|U| / s + Σ_i |in(B i) \ θ|) = 1/2`. -/
def UncertifiedLocal_nontrivial_bodies : Prop :=
  0 < 2 ∧ ¬ OperationBodies Detached.d Detached.J ∧ Overlap Detached.J 1 ∧
  UncertifiedBodySize Detached.J Detached.server Detached.represented 2 ∧ (0 : ℝ) ≤ 1 ∧
  (∀ v ∈ Detached.J.V, Detached.J.ω v ≤ 1) ∧
  ¬ UncertifiedCap Detached.J Detached.server Detached.represented
    (1 * (1 : ℕ) * ((uncertified Detached.server Detached.represented).card / (2 : ℕ) +
      ∑ i, (Detached.d.unregistered (Detached.represented i)).card))

end Accounting.Guarantees.ServingBlocks
