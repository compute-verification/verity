module

public import Mathlib.Analysis.SpecialFunctions.Log.Base
public import Definitions.Flock.Audit.Harm

/-!
# A sampling design's targets

The two targets a sampling design is planned for (`optimize.py`'s work and leak targets), which the partitioning
guarantees (`Proofs.Flock.Soundness.Audit.Partitioning`) and the planner's checker (`Properties.Spot.PlanCheck`)
read:

* `MeetsTarget L w C δ`, the free-compute target: every set of incorrect proof blocks whose work is at least `C` escapes
  with probability at most `δ`;
* `locBits n K = log₂ ∑_{j ≤ K} C(n, j)`, the location term of `audit_exfiltration`, and `CertifiesLeak L b δ ℓ`, the
  exfiltration target: with unit `u` exporting `b u` bits, `audit_exfiltration`'s bound certifies at most `ℓ` bits at
  `δ`.
-/

@[expose] public section

namespace FlockSoundness.Audit.Partitioning

open Finset
open scoped ENNReal

variable {n : ℕ}

/-- **The free-compute target**: every set of incorrect proof blocks whose work is at least `C` escapes with probability at most
`δ`. -/
def MeetsTarget (L : Law n) (w : Fin n → ℝ) (C δ : ℝ) : Prop :=
  ∀ B : Finset (Fin n), C ≤ ∑ u ∈ B, w u → (L.escape B).toReal ≤ δ

/-- The location term of `audit_exfiltration`: `log₂ ∑_{j ≤ K} C(n, j)`, the bits that say which units are wrong when at
most `K` of `n` are. -/
noncomputable def locBits (n K : ℕ) : ℝ := Real.logb 2 (∑ j ∈ range (K + 1), n.choose j : ℕ)

/-- **The exfiltration target**: with unit `u` exporting `b u` bits, `audit_exfiltration`'s bound certifies at most `ℓ`
bits at `δ`. That is, a harm bound `H` of the widths at `δ` and a count `K` with `miss (K + 1) < δ`, with
`H + locBits n K ≤ ℓ`, so that a uniform message of `m` bits is recovered from an accepted audit with probability at
most `2^(ℓ − m)` plus `δ` and the audit's error. -/
def CertifiesLeak (L : Law n) (b : Fin n → ℕ) (δ ℓ : ℝ) : Prop :=
  ∃ H K : ℕ, IsHarmBound L (fun u => (b u : ℝ)) (ENNReal.ofReal δ) H ∧ L.miss (K + 1) < ENNReal.ofReal δ ∧
    (H : ℝ) + locBits n K ≤ ℓ

end FlockSoundness.Audit.Partitioning
