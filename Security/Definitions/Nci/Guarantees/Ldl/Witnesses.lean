import Definitions.Nci.Guarantees.Ldl
import Definitions.Nci.Ldl.Instances

/-!
# Local description length: honest and nontrivial lemmas

Each guarantee of `Nci.Ldl.Guarantees` has two statements here, at the one-product instance `Nci.Ldl.Honest`
(`Security/Definitions/Nci/Ldl/Instances.lean`). `<Guarantee>_honest`: its premises hold together there.
`<Guarantee>_nontrivial`: at the same instance with one parameter changed, so that one named premise fails and every
other holds, the guarantee's bound is false, so the guarantee is false without that premise.

The proofs, and that the changed instance meets every other premise and fails the named one, are in
`Security/Proofs/Nci/Ldl/Witnesses.lean`.
-/

namespace Nci.Ldl.Guarantees

open Verity

/-- **Lemma 1's premises hold** at `Honest`, at `k = 1`, `t = 0`, `aBlk = 0`, extracting each block's activation tuple:
the block reads one unregistered input bit, the run takes the (empty) registration, and the decoder that prints its
program evaluates the block from that bit. -/
def BlockDescription_honest : Prop :=
  ReadCap Honest.d Honest.R.P 1 ∧ Honest.R.Consistent (Honest.d.D.eval Honest.inp) ∧
    Assumptions.EvaluatesBlocks Honest.D 0 Honest.R Honest.a 0

/-- **Lemma 1 is false without the read cap**: at `Honest` with `k = 0`, the decoder still evaluates the block and the
run takes the registration, the block reads one bit, more than `0`, and its activation tuple's description length `1`
is above `k + ⌈log₂ B⌉ + aBlk = 0`. -/
def BlockDescription_nontrivial : Prop :=
  ¬ LDL Honest.D 0 (Honest.a 0 (Honest.d.D.eval Honest.inp)) ≤ ((0 + Nat.clog 2 1 + 0 : ℕ) : ℕ∞)

/-- **Theorem 3's premises hold** at `Honest`, at `c = 1`, `V = {1}` (the product's gate), unit weight and `q = 1`:
the block has exactly one gate, and its counted weight is `1`. -/
def Counting_honest : Prop :=
  Exact Honest.P 1 ∧ ({1} : Finset (Fin Honest.C.N)) ⊆ Honest.C.computed ∧
    ∀ u, ∑ _v ∈ ({1} : Finset (Fin Honest.C.N)) ∩ Honest.P.gates u, (1 : ℝ) ≤ 1

/-- **Theorem 3 is false without the per-block weight cap**: at `Honest` with weight `2` on the product's gate and
`q = 1`, the block is exact and `V` is its gate, the block's counted weight `2` is above `q`, and `(c / q) Σ ω = 2` is
above `|C| = 1`. -/
def Counting_nontrivial : Prop :=
  ¬ ((1 : ℕ) : ℝ) / 1 * ∑ _v ∈ ({1} : Finset (Fin Honest.C.N)), (2 : ℝ) ≤ Honest.C.size Honest.C.computed

/-- **Corollary 4b's premises hold** at `Honest`, at `(c, k) = (1, 1)`, fan-out `e = 1`, `t = 0`, `aBlk = 0`, and the
activation rate `β = 1` at deficit `Λ = 0`: the product's one activation bit costs one bit. -/
def FanOutBound_honest : Prop :=
  CKPartition Honest.d Honest.R.P 1 1 ∧ Honest.R.Consistent (Honest.d.D.eval Honest.inp) ∧ Honest.M.FanOut 1 ∧
    Assumptions.EvaluatesBlocks Honest.D 0 Honest.R Honest.a 0 ∧
    Assumptions.ActivationRate 1 0 0 Honest.D Honest.R Honest.M Honest.a (Honest.d.D.eval Honest.inp)

/-- **Corollary 4b is false without the activation rate**: at `Honest` with `β = 2`, every premise but
`ActivationRate` holds, the activation tuple's description length `1` is below `β |V_I| − Λ = 2`, and
`c β M / (T₀ e h) = 2` is above the overhead `|C| / T₀ = 1` at `T₀ = 1`, `h = k + ⌈log₂ B⌉ + aBlk + Λ = 1`. -/
def FanOutBound_nontrivial : Prop :=
  ¬ ((1 : ℕ) : ℝ) * 2 * Honest.M.J.card / (1 * ((1 : ℕ) : ℝ) * ((1 : ℕ) + Nat.clog 2 1 + (0 : ℕ) + 0)) ≤
    Honest.d.D.size Honest.d.D.computed / 1

end Nci.Ldl.Guarantees
