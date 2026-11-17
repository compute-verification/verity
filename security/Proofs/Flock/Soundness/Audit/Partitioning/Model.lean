module

public import Mathlib.Analysis.SpecialFunctions.Log.Base
public import Definitions.Flock.Audit.Partitioning

/-! Optimal partitioning: the model the partitioning guarantees read. -/

@[expose] public section

namespace FlockSoundness.Audit.Partitioning

open Finset
open scoped ENNReal

variable {n : ℕ}

noncomputable def cost (L : SamplingDesign n) (c : Fin n → ℝ) : ℝ := ∑ u, (L.incl u).toReal * c u

noncomputable def meanSamples (L : SamplingDesign n) : ℝ := ∑ u, (L.incl u).toReal

def unitCost (f o s : ℝ) : ℝ := f + o * s

/-- The leading served cost of cutting work `F` into units of work `s`: proving, `c(s) · (F/C) · Λ` with `Λ = ln(1/δ)`,
plus serving's commitment of every one of the `F/s` units, sampled or not, `β` bytes each at `h` a byte, weighted `w`
against proving. -/
noncomputable def servedCost (F C f o δ w β h s : ℝ) : ℝ :=
  unitCost f o s * (F / C * Real.log δ⁻¹) + w * (F / s * β * h)

/-- **The detection risk** of a replay block holding `m` wrong proof blocks out of `N`: the verifier samples the replay block
with probability `p`, then `k` of its proof blocks uniformly, and the replay block escapes with probability
`1 − p + p · C(N − m, k) / C(N, k)`. This is its negative logarithm. -/
noncomputable def detectionRisk (N k : ℕ) (p : ℝ) (m : ℕ) : ℝ :=
  -Real.log (1 - p + p * ((N - m).choose k : ℝ) / (N.choose k : ℝ))

/-- **The nested sampling design** over `R` replay blocks of `N` proof blocks each, proof block `j` of replay block `i` being
`finProdFinEquiv (i, j)`: each replay block is sampled independently with probability `a / b` (`SamplingDesign.bernoulli`), and inside
each sampled one, `k` of its `N` proof blocks uniformly (`SamplingDesign.subset`), independently across replay blocks. -/
def nested (R N k a b : ℕ) [NeZero b] (hk : k ≤ N) : SamplingDesign (R * N) where
  Ω := (SamplingDesign.bernoulli R a b).Ω × (Fin R → (SamplingDesign.subset N k hk).Ω)
  draw ω := Finset.univ.filter fun v =>
    (finProdFinEquiv.symm v).1 ∈ (SamplingDesign.bernoulli R a b).draw ω.1 ∧
      (finProdFinEquiv.symm v).2 ∈ (SamplingDesign.subset N k hk).draw (ω.2 (finProdFinEquiv.symm v).1)

/-- The number of proof blocks of `B` in replay block `i`, numbered as in `nested`. -/
def incorrectIn {R N : ℕ} (B : Finset (Fin (R * N))) (i : Fin R) : ℕ :=
  (Finset.univ.filter fun j : Fin N => finProdFinEquiv (i, j) ∈ B).card

end FlockSoundness.Audit.Partitioning
