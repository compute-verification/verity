import Proofs.Flock.Soundness.Audit.Harm
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Base

/-!
# Optimal partitioning: the model

A computation is divided into `n` proof units. Unit `u` does work `w u`, and proving it costs `c u`. The verifier draws
units by a law `L` (`Law`), a drawn wrong unit is caught, and the prover fixes its transcript before the draw. These are
the definitions the partitioning guarantees (`Partitioning.Guarantees`) read:

* `cost L c = ∑_u incl(u) · c(u)`, the verifier's expected proving cost, and `meanDraws L = ∑_u incl(u)`, the expected
  number of units it draws;
* `unitCost f o s = f + o · s`, the simple cost model: a fixed cost `f` per drawn unit plus an overhead `o` per unit of
  work;
* `servedCost F C f o δ w β h s`, the leading cost of cutting work `F` into units of work `s` when serving commits every
  unit: proving's `c(s) · (F/C) · Λ` plus the commitment's `w · (F/s) · β · h`;
* `MeetsTarget L w C δ`, the free-compute target: every set of wrong units whose work is at least `C` escapes with
  probability at most `δ`;
* `locBits n K = log₂ ∑_{j ≤ K} C(n, j)`, the location term of `audit_exfiltration`, and `CertifiesLeak L b δ ℓ`, the
  exfiltration target: with unit `u` exporting `b u` bits, `audit_exfiltration`'s bound certifies at most `ℓ` bits at
  `δ`;
* `detectionRisk N k p m = −ln(1 − p + p · C(N − m, k) / C(N, k))`, the detection risk of a replay unit holding `m` wrong
  proof units out of `N`, when the verifier draws the replay unit with probability `p` and then `k` of its proof units
  uniformly;
* `nested R N k a b`, the law that draws so: over `R` replay units of `N` proof units each, numbered as `R * N` proof
  units, each replay unit with probability `a / b`, then `k` of its proof units uniformly; and `wrongIn B i`, the number
  of proof units of `B` in replay unit `i`.
-/

namespace FlockSoundness.Audit.Partitioning

open Finset
open scoped ENNReal

variable {n : ℕ}

/-- The verifier's expected proving cost: each unit's cost `c u`, weighted by its inclusion probability. -/
noncomputable def cost (L : Law n) (c : Fin n → ℝ) : ℝ := ∑ u, (L.incl u).toReal * c u

/-- The expected number of units a law draws. -/
noncomputable def meanDraws (L : Law n) : ℝ := ∑ u, (L.incl u).toReal

/-- The simple cost model: a fixed cost `f` per drawn unit plus an overhead `o` per unit of work `s`. -/
def unitCost (f o s : ℝ) : ℝ := f + o * s

/-- The leading served cost of cutting work `F` into units of work `s`: proving, `c(s) · (F/C) · Λ` with `Λ = ln(1/δ)`,
plus serving's commitment of every one of the `F/s` units, drawn or not, `β` bytes each at `h` a byte, weighted `w`
against proving. -/
noncomputable def servedCost (F C f o δ w β h s : ℝ) : ℝ :=
  unitCost f o s * (F / C * Real.log δ⁻¹) + w * (F / s * β * h)

/-- **The free-compute target**: every set of wrong units whose work is at least `C` escapes with probability at most
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

/-- **The detection risk** of a replay unit holding `m` wrong proof units out of `N`: the verifier draws the replay unit
with probability `p`, then `k` of its proof units uniformly, and the replay unit escapes with probability
`1 − p + p · C(N − m, k) / C(N, k)`. This is its negative logarithm. -/
noncomputable def detectionRisk (N k : ℕ) (p : ℝ) (m : ℕ) : ℝ :=
  -Real.log (1 - p + p * ((N - m).choose k : ℝ) / (N.choose k : ℝ))

/-- **The nested law** over `R` replay units of `N` proof units each, proof unit `j` of replay unit `i` being
`finProdFinEquiv (i, j)`: each replay unit is drawn independently with probability `a / b` (`Law.bernoulli`), and inside
each drawn one, `k` of its `N` proof units uniformly (`Law.subset`), independently across replay units. -/
def nested (R N k a b : ℕ) [NeZero b] (hk : k ≤ N) : Law (R * N) where
  Ω := (Law.bernoulli R a b).Ω × (Fin R → (Law.subset N k hk).Ω)
  draw ω := Finset.univ.filter fun v =>
    (finProdFinEquiv.symm v).1 ∈ (Law.bernoulli R a b).draw ω.1 ∧
      (finProdFinEquiv.symm v).2 ∈ (Law.subset N k hk).draw (ω.2 (finProdFinEquiv.symm v).1)

/-- The number of proof units of `B` in replay unit `i`, numbered as in `nested`. -/
def wrongIn {R N : ℕ} (B : Finset (Fin (R * N))) (i : Fin R) : ℕ :=
  (Finset.univ.filter fun j : Fin N => finProdFinEquiv (i, j) ∈ B).card

end FlockSoundness.Audit.Partitioning
