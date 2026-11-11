module

public import Mathlib.Analysis.SpecialFunctions.Log.Base
public import Definitions.Flock.Audit.Partitioning

/-!
# Optimal partitioning: the model

A computation is divided into `n` proof blocks. Unit `u` does work `w u`, and proving it costs `c u`. The verifier samples
units by a sampling design `L` (`Law`), a sampled incorrect proof block is caught, and the prover fixes its transcript before the sample. These are
the definitions the partitioning guarantees (`Partitioning.Guarantees`) read:

* `cost L c = ∑_u incl(u) · c(u)`, the verifier's expected proving cost, and `meanDraws L = ∑_u incl(u)`, the expected
  number of units it samples;
* `unitCost f o s = f + o · s`, the simple cost model: a fixed cost `f` per sampled proof block plus an overhead `o` per unit of
  work;
* `servedCost F C f o δ w β h s`, the leading cost of cutting work `F` into units of work `s` when serving commits every
  unit: proving's `c(s) · (F/C) · Λ` plus the commitment's `w · (F/s) · β · h`;
* the targets, `MeetsTarget` (free compute) and `CertifiesLeak` (exfiltration) with its location term `locBits`, in
  `Definitions.Flock.Audit.Partitioning`;
* `detectionRisk N k p m = −ln(1 − p + p · C(N − m, k) / C(N, k))`, the detection risk of a replay unit holding `m` wrong
  proof blocks out of `N`, when the verifier samples the replay unit with probability `p` and then `k` of its proof blocks
  uniformly;
* `nested R N k a b`, the sampling design that samples so: over `R` replay units of `N` proof blocks each, numbered as `R * N` proof
  blocks, each replay unit with probability `a / b`, then `k` of its proof blocks uniformly; and `wrongIn B i`, the number
  of proof blocks of `B` in replay unit `i`.
-/

@[expose] public section

namespace FlockSoundness.Audit.Partitioning

open Finset
open scoped ENNReal

variable {n : ℕ}

/-- The verifier's expected proving cost: each unit's cost `c u`, weighted by its inclusion probability. -/
noncomputable def cost (L : Law n) (c : Fin n → ℝ) : ℝ := ∑ u, (L.incl u).toReal * c u

/-- The expected number of units a sampling design samples. -/
noncomputable def meanDraws (L : Law n) : ℝ := ∑ u, (L.incl u).toReal

/-- The simple cost model: a fixed cost `f` per sampled proof block plus an overhead `o` per unit of work `s`. -/
def unitCost (f o s : ℝ) : ℝ := f + o * s

/-- The leading served cost of cutting work `F` into units of work `s`: proving, `c(s) · (F/C) · Λ` with `Λ = ln(1/δ)`,
plus serving's commitment of every one of the `F/s` units, sampled or not, `β` bytes each at `h` a byte, weighted `w`
against proving. -/
noncomputable def servedCost (F C f o δ w β h s : ℝ) : ℝ :=
  unitCost f o s * (F / C * Real.log δ⁻¹) + w * (F / s * β * h)

/-- **The detection risk** of a replay unit holding `m` wrong proof blocks out of `N`: the verifier samples the replay unit
with probability `p`, then `k` of its proof blocks uniformly, and the replay unit escapes with probability
`1 − p + p · C(N − m, k) / C(N, k)`. This is its negative logarithm. -/
noncomputable def detectionRisk (N k : ℕ) (p : ℝ) (m : ℕ) : ℝ :=
  -Real.log (1 - p + p * ((N - m).choose k : ℝ) / (N.choose k : ℝ))

/-- **The nested law** over `R` replay units of `N` proof blocks each, proof block `j` of replay unit `i` being
`finProdFinEquiv (i, j)`: each replay unit is sampled independently with probability `a / b` (`Law.bernoulli`), and inside
each sampled one, `k` of its `N` proof blocks uniformly (`Law.subset`), independently across replay units. -/
def nested (R N k a b : ℕ) [NeZero b] (hk : k ≤ N) : Law (R * N) where
  Ω := (Law.bernoulli R a b).Ω × (Fin R → (Law.subset N k hk).Ω)
  draw ω := Finset.univ.filter fun v =>
    (finProdFinEquiv.symm v).1 ∈ (Law.bernoulli R a b).draw ω.1 ∧
      (finProdFinEquiv.symm v).2 ∈ (Law.subset N k hk).draw (ω.2 (finProdFinEquiv.symm v).1)

/-- The number of proof blocks of `B` in replay unit `i`, numbered as in `nested`. -/
def wrongIn {R N : ℕ} (B : Finset (Fin (R * N))) (i : Fin R) : ℕ :=
  (Finset.univ.filter fun j : Fin N => finProdFinEquiv (i, j) ∈ B).card

end FlockSoundness.Audit.Partitioning
