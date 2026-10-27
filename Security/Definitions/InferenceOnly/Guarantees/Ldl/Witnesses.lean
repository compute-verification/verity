import Definitions.InferenceOnly.Guarantees.Ldl
import Definitions.InferenceOnly.Ldl.Instances

/-!
# Local description length: honest and nontrivial lemmas

Each guarantee of `InferenceOnly.Ldl.Guarantees` has two statements here, at the one-product instance `InferenceOnly.Ldl.Honest`
(`Security/Definitions/InferenceOnly/Ldl/Instances.lean`). `<Guarantee>_honest`: its premises hold together there.
`<Guarantee>_nontrivial`: at the same instance with one parameter changed, every premise but one named premise holds,
that one fails, and the guarantee's bound is false, so the guarantee is false without it.

The proofs are in `Security/Proofs/InferenceOnly/Ldl/Witnesses.lean`.
-/

namespace InferenceOnly.Ldl.Guarantees

open Verity

/-- **Lemma 1's premises hold** at `Honest`, at `k = 1`, `t = 0`, `aBlk = 0`, extracting each block's activation tuple:
the block reads one unregistered input bit, the run takes the (empty) registration, and the decoder that prints its
program evaluates the block from that bit. -/
def BlockDescription_honest : Prop :=
  ReadCap Honest.d Honest.R.P 1 ∧ Honest.R.Consistent (Honest.d.D.eval Honest.inp) ∧
    Assumptions.EvaluatesBlocks Honest.D 0 Honest.R Honest.a 0

/-- **Lemma 1 is false without the read cap**: at `Honest` with `k = 0`, the run takes the registration and the decoder
evaluates the block, the read cap fails (the block reads one bit, more than `0`), and the block's activation tuple has
description length `1`, above `k + ⌈log₂ B⌉ + aBlk = 0`. -/
def BlockDescription_nontrivial : Prop :=
  Honest.R.Consistent (Honest.d.D.eval Honest.inp) ∧ Assumptions.EvaluatesBlocks Honest.D 0 Honest.R Honest.a 0 ∧
    ¬ ReadCap Honest.d Honest.R.P 0 ∧
    ¬ ∀ u, LDL Honest.D 0 (Honest.a u (Honest.d.D.eval Honest.inp)) ≤ ((0 + Nat.clog 2 1 + 0 : ℕ) : ℕ∞)

/-- **Theorem 3's premises hold** at `Honest`, at `c = 1`, `V = {1}` (the product's gate), unit weight and `q = 1`:
the block has exactly one gate, and its counted weight is `1`. -/
def Counting_honest : Prop :=
  Exact Honest.P 1 ∧ ({1} : Finset (Fin Honest.C.N)) ⊆ Honest.C.computed ∧
    ∀ u, ∑ _v ∈ ({1} : Finset (Fin Honest.C.N)) ∩ Honest.P.gates u, (1 : ℝ) ≤ 1

/-- **Theorem 3 is false without the per-block weight cap**: at `Honest` with weight `2` on the product's gate and
`q = T₀ = 1`, the block is exact, `V` is its gate and `q, T₀ > 0`, the weight cap fails (the block's counted weight `2`
is above `q`), and `(c / q) Σ ω = 2` is above `|C| = 1`. -/
def Counting_nontrivial : Prop :=
  Exact Honest.P 1 ∧ ({1} : Finset (Fin Honest.C.N)) ⊆ Honest.C.computed ∧ (0 : ℝ) < 1 ∧
    ¬ (∀ u, ∑ _v ∈ ({1} : Finset (Fin Honest.C.N)) ∩ Honest.P.gates u, (2 : ℝ) ≤ 1) ∧
    ¬ (((1 : ℕ) : ℝ) / 1 * ∑ _v ∈ ({1} : Finset (Fin Honest.C.N)), (2 : ℝ) ≤ Honest.C.size Honest.C.computed ∧
      ((1 : ℕ) : ℝ) / (1 * 1) * ∑ _v ∈ ({1} : Finset (Fin Honest.C.N)), (2 : ℝ) ≤
        (Honest.C.size Honest.C.computed : ℝ) / 1)

/-- **Corollary 4b's premises hold** at `Honest`, at `(c, k) = (1, 1)`, fan-out `e = 1`, `t = 0`, `aBlk = 0`, and the
activation rate `β = 1` at deficit `Λ = 0`: the product's one activation bit costs one bit. -/
def FanOutBound_honest : Prop :=
  CKPartition Honest.d Honest.R.P 1 1 ∧ Honest.R.Consistent (Honest.d.D.eval Honest.inp) ∧ Honest.M.FanOut 1 ∧
    Assumptions.EvaluatesBlocks Honest.D 0 Honest.R Honest.a 0 ∧
    Assumptions.ActivationRate 1 0 0 Honest.D Honest.R Honest.M Honest.a (Honest.d.D.eval Honest.inp)

/-- **Corollary 4b is false without the activation rate**: at `Honest` with `β = 2` and `T₀ = 1`, the partition, the
registration, fan-out, `β, T₀ > 0` and the decoder's evaluation hold, the activation rate fails (the activation tuple's
description length `1` is below `β |V_I| − Λ = 2`), and `c β M / (T₀ e h) = 2` is above the overhead `|C| / T₀ = 1`,
at `h = k + ⌈log₂ B⌉ + aBlk + Λ = 1`. -/
def FanOutBound_nontrivial : Prop :=
  CKPartition Honest.d Honest.R.P 1 1 ∧ Honest.R.Consistent (Honest.d.D.eval Honest.inp) ∧ Honest.M.FanOut 1 ∧
    (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ Assumptions.EvaluatesBlocks Honest.D 0 Honest.R Honest.a 0 ∧
    ¬ Assumptions.ActivationRate 2 0 0 Honest.D Honest.R Honest.M Honest.a (Honest.d.D.eval Honest.inp) ∧
    ¬ ((1 : ℕ) : ℝ) * 2 * Honest.M.J.card / (1 * ((1 : ℕ) : ℝ) * ((1 : ℕ) + Nat.clog 2 1 + (0 : ℕ) + 0)) ≤
      Honest.d.D.size Honest.d.D.computed / 1

end InferenceOnly.Ldl.Guarantees
