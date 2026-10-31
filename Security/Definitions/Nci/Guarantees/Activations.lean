import Definitions.Nci.Assumptions.Activations
import Definitions.Nci.Assumptions
import Definitions.Nci.Dark

/-!
# The inference-only policy's activations bound over all of training ([ACTIVATIONS-BOUND])

Under `ActivationsIncompressibleAll`, every small subcircuit of the reference circuit `r` computes at most
`Λ = max(e, c_r · K / (4 dd b_w))` counted multiplications per `b_a` bits it reads, where `e` bounds how many counted
multiplications one activation enters, `c_r = T.reuse` how many times a thread uses one weight, and `dd` how many
distinct activations of its own thread each linked activation is computed from (`ActivationsReads`). The bound is
`Threading.bound K b_w dd e`, Theorem 1's of the thread draft, with no threading hypothesis.

A subcircuit can compute complete activations itself and use them again, a cascade through many layers of one thread.
The bound does not charge such an activation to the products it is computed from, which would give a factor
`1 / (1 − e/dd)` and diverge once one activation enters more multiplications than a dot product has terms (Kimi K2's MoE
input enters 37,248, against an inner dimension of 7,168). It counts a thread in which the subcircuit completes any
activation by its weights instead, at most `c_r` multiplications per weight used, and charges that thread the `dd`
activations of its earliest complete one, none of which the subcircuit completes.

Through a partial containment (`ActivationsBoundOn`) the bound holds for the counted multiplications the declaration
computes, with no term for the dark gates: a partial sum added outside the declaration, or any other value computed
off the books, enters it as an input, fresh input by `ThetaRegistered`, and does not make an activation whole in a
piece. `ActivationsTrainingOn` is the training form, with the counted multiplications off the books as dark gates.
-/

namespace Nci.Guarantees

open Verity

/-- **The activations bound for one subcircuit**: under `ActivationsIncompressibleAll` at read cap `K`, every small
subcircuit `R` of the reference circuit `r` computes at most `Λ = max(e, c_r · K / (4 dd b_w))` counted multiplications
per `b_a` fresh input bits it reads, where each activation enters at most `e` counted multiplications and each
activation computed from counted multiplications is computed from at least `dd` distinct activations of its own thread. -/
def ActivationsReads : Prop :=
  ∀ (r : RegisteredCircuit) (T : Threading r) (L : T.Links) (c : ℕ) (k b_w b_a : ℝ) (dd e : ℕ),
    0 < b_w → 0 < b_a → 0 < dd →
    Assumptions.EntersAtMost T e → Assumptions.DrawsAtLeast L dd →
    Assumptions.ActivationsIncompressibleAll r T L c k b_w b_a →
    ∀ R, Small r c k R → b_a * (T.mults R).card ≤ novelReads r R * T.bound k b_w dd e

/-- **The activations bound over all of training, with dark gates** ([ACTIVATIONS-BOUND]): a declaration `d` that
complies at `(c, k)` and contains the reference circuit `r` partly, computing the gates `S` of `r` and leaving the rest
dark (`PartialContainment`), computes at most `(k / c) Λ size(d) / b_a` of `r`'s counted multiplications, with
`Λ = max(e, c_r · K / (4 dd b_w))` at read cap `K ≥ k`, under `ActivationsIncompressibleAll` on `r` at `K` and
`ThetaRegistered`. There is no term for the dark gates, so it holds however much of `r`, the partial sums included,
is computed off the books. -/
def ActivationsBoundOn : Prop :=
  ∀ (c k : ℕ) (K b_w b_a : ℝ) (r d : RegisteredCircuit) (T : Threading r) (L : T.Links) (dd e : ℕ)
    (M : PartialContainment r d),
    0 < c → (k : ℝ) ≤ K → 0 < b_w → 0 < b_a → 0 < dd → Complies c k d → Assumptions.ThetaRegistered r d M →
    Assumptions.EntersAtMost T e → Assumptions.DrawsAtLeast L dd →
    Assumptions.ActivationsIncompressibleAll r T L c K b_w b_a →
      b_a * (T.M ∩ M.S).card ≤ (k / c : ℝ) * T.bound K b_w dd e * d.D.size d.D.computed

/-- **Training is slowed, by the activations bound, with dark gates**: when counted multiplications are a fraction `κ`
of the reference circuit's flops (2 flops each), a declaration that complies at `(c, k)` and contains it partly, with
its registered information free of the reference circuit's fresh input (`ThetaRegistered`), is larger by a factor of
at least `κ c b_a / (2 k Λ)`, less what its dark gates absorb:
`κ b_a size(r) ≤ 2 b_a #dark + 2 (k / c) Λ size(d)`. -/
def ActivationsTrainingOn : Prop :=
  ∀ (c k : ℕ) (K b_w b_a κ : ℝ) (r d : RegisteredCircuit) (T : Threading r) (L : T.Links) (dd e : ℕ)
    (M : PartialContainment r d),
    0 < c → (k : ℝ) ≤ K → 0 < b_w → 0 < b_a → 0 < dd → Complies c k d → Assumptions.ThetaRegistered r d M →
    Assumptions.EntersAtMost T e → Assumptions.DrawsAtLeast L dd →
    Assumptions.ActivationsIncompressibleAll r T L c K b_w b_a →
    Assumptions.Coverage κ r (2 * T.M.card) →
      κ * b_a * r.D.size r.D.computed ≤
        2 * b_a * M.dark.card + 2 * (k / c : ℝ) * T.bound K b_w dd e * d.D.size d.D.computed

/-- **The premise over all counted multiplications gives the chain's**: where a chain `ch` is part of the threading
(`ChainIn`), `ActivationsIncompressibleAll` gives `ActivationsIncompressible` for `ch` at the same caps, for nonnegative
rates: a chain's weight entries and input entries are distinct weight and activation gates, and an input entry whose
activation is whole in `R` is linked to an entry whole in `R`. -/
def ActivationsChain : Prop :=
  ∀ (r : RegisteredCircuit) (ch : Chain r.D) (T : Threading r) (L : T.Links) (c : ℕ) (k b_w b_a : ℝ),
    ChainIn ch T L → 0 ≤ b_w → 0 ≤ b_a →
    Assumptions.ActivationsIncompressibleAll r T L c k b_w b_a →
      Assumptions.ActivationsIncompressible r ch c k b_w b_a

end Nci.Guarantees
