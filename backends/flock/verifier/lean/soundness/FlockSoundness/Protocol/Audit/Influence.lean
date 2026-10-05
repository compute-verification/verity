import FlockSoundness.Protocol.Audit.OneStage
import FlockSoundness.Protocol.Audit.Harm

/-!
# Influence through a separator

An accepted audit bounds which units are wrong; this file bounds what the wrong units can make the audited computation
deliver. `D` is the set of delivered wires: what the consumer receives (the served tokens). It can be smaller than the
circuit's outputs, since commitments and leaves are outputs that the verifier sees and nobody is delivered.

* `Circuit.FeedsAvoiding K g t`: `g` feeds `t` along a path none of whose gates is in `K`.
* `Partition.Separates K B D` (a downstream cut): `K` meets every path from a gate of a unit of `B` to a wire of `D`.
* `outputs_eq_of_separates`, **the influence cap**: two transcripts with the anchors' inputs, whose wrong units lie in
  `B`, and that agree on a committed separator of `B`, deliver the same values on `D`. So they deliver at most `2^|K|`
  values on `D` (`card_delivered_le`): in a Boolean circuit a gate is one bit, and `|K|` is the separator's width.
* `separates_exits`: the committed wires of the units of `B` separate `B`. For a replay unit, those are its exports.
  `Separates.mono` and `Separates.union` build separators for larger sets from smaller ones.
* `influenceSet D e a δ`: what the transcripts with the anchors' inputs deliver on `D` when their wrong units form a set
  whose escape `e B` is at least `δ`. It is fixed by the circuit, the partition, the anchors, the law and `δ`, never by
  the prover, and `card_influenceSet_le` bounds its size by `∑_{B : δ ≤ e B} 2^|K B|`.
* `audit_influence` (one-stage) and `Refines.twoStage_influence` (two-stage, over replay units, with `effEscape`): the
  audit accepts while the delivered values lie outside the influence set with probability at most
  `δ + ε_ks + δ_link + δ_in`. The influence in bits is `log₂ |influenceSet|`.
* `card_influenceSet_le_harm` and `card_admissible_le`: the influence is at most `H + log₂ ∑_{j ≤ K} C(n, j)` bits. `H` is
  a harm bound (`IsHarmBound`) with each unit's harm the width it adds to a separator, for example its exits
  (`card_exits_le`): the value bits. The second term is the choice of which units are wrong, when `miss (K + 1) < δ`.
* `audit_exfiltration`: a prover that picks its strategy from a uniform message `m`, and a receiver that sees only what
  is delivered, recover `m` from an accepted audit with probability at most `2^H · ∑_{j ≤ K} C(n, j) / |M|` plus the
  audit's error: the value bits and the location term together. `card_guess_le` is its counting step.
-/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal

namespace Circuit

variable (C : Circuit)

/-- `g` feeds `t` along a path none of whose gates is in `K`, both ends included. -/
inductive FeedsAvoiding (K : Finset (Fin C.N)) : Fin C.N → Fin C.N → Prop
  | refl {t : Fin C.N} : t ∉ K → FeedsAvoiding K t t
  | step {g h t : Fin C.N} : g ∉ K → g ∈ (C.op h).args → FeedsAvoiding K h t → FeedsAvoiding K g t

end Circuit

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

/-- **A separator of the units `B` from the delivered wires `D`** (a downstream cut): every path from a gate of a unit
of `B` to a wire of `D` meets `K`. -/
def Separates (K : Finset (Fin C.N)) (B : Finset (Fin n)) (D : Finset (Fin C.N)) : Prop :=
  ∀ g t u, P.unit g = some u → u ∈ B → t ∈ D → ¬ C.FeedsAvoiding K g t

variable {P}

variable (P)

/-- What a transcript delivers on the wires `D`. -/
def out (D : Finset (Fin C.N)) (X : Fin C.N → Bool) : D → Bool := fun t => X t

section Harm

variable {L : Law n} {a : Fin C.N → Bool} {D : Finset (Fin C.N)}

end Harm

section OneStage

variable {L : Law n} {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}
  {X : (R : Reg) → Cont L Reg session R → (Fin C.N → Bool)}
  {ext : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → Option (Fin C.N → Bool)}
  {a : Fin C.N → Bool} {εks δlink δin : ℝ≥0∞} {D : Finset (Fin C.N)}

end OneStage

section TwoStage

variable {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc} {a : Fin C.N → Bool} {εks δlink δin : ℝ≥0∞}
  {D : Finset (Fin C.N)}

end TwoStage

end Partition

end FlockSoundness.Audit
