import Definitions.Nci.Guarantees.Hardness
import Definitions.Nci.Ldl.Instances

/-!
# Priced reads: honest and nontrivial lemmas

Each guarantee of `Nci.Hardness.Guarantees` has two statements here, at the one-product instance `Nci.Ldl.Honest`
(`Security/Definitions/Nci/Ldl/Instances.lean`), its product a counted operation `Honest.O`. `<Guarantee>_honest`: its
premises hold together there, with the product priced at `β = 1` per activation occurrence. `<Guarantee>_nontrivial`:
at `β = 2` every premise but the price premise (`Priced` or `ActivationRate`) holds, the price premise fails, and the
guarantee's bound is false, so the guarantee is false without it.

The proofs are in `Security/Proofs/Nci/Hardness/Witnesses.lean`.
-/

namespace Nci.Hardness.Guarantees

open Verity Nci.Ldl

/-- **The per-block cap's premises hold** at `Honest`, at fan-out `e = 1` and block size `c = 1`: the product is
`Priced` at `β = 1` per activation occurrence and deficit `Λ = 0`, since the one block that computes it reads its one
activation bit. -/
def FanOutBlockCap_honest : Prop :=
  Honest.O.FanOut 1 ∧ Assumptions.Priced Honest.d Honest.O (fun _ => 1) 0 1 (fun _ => ∅)

/-- **The per-block cap is false without the price**: at `Honest` with `β = 2`, fan-out and `β > 0` hold, `Priced`
fails at the block `{1}` (it reads one bit, below `2`), and the block `{1}`, of one gate reading `k = 1` bit, computes
one product whole, above `q(1, 1) = e (k + Λ) / β = 1 / 2`. -/
def FanOutBlockCap_nontrivial : Prop :=
  Honest.O.FanOut 1 ∧ (0 : ℝ) < 2 ∧ ¬ Assumptions.Priced Honest.d Honest.O (fun _ => 2) 0 1 (fun _ => ∅) ∧
    ¬ ∀ I ⊆ Honest.d.D.computed, Honest.d.D.size I ≤ 1 → Honest.d.D.width (Honest.d.D.into I \ Honest.d.θ) ≤ 1 →
      ((Honest.O.whole I).card : ℝ) ≤ ((1 : ℕ) : ℝ) * ((1 : ℕ) + 0) / 2

/-- **Corollary 4b's priced premises hold** at `Honest`, at `(c, k) = (1, 1)`, fan-out `e = 1`, and the price `β = 1`
per activation occurrence at deficit `Λ = 0`: the product lies whole in the one block. -/
def FanOutBoundPriced_honest : Prop :=
  CKPartition Honest.d Honest.P 1 1 ∧ Honest.O.WholeIn Honest.P ∧ Honest.O.FanOut 1 ∧
    Assumptions.Priced Honest.d Honest.O (fun _ => 1) 0 1 (fun _ => ∅)

/-- **Corollary 4b under `Priced` is false without the price**: at `Honest` with `β = 2` and `T₀ = 1`, the partition,
wholeness, fan-out and `β, T₀ > 0` hold, `Priced` fails, and `c β M / (T₀ e (k + Λ)) = 2` is above the overhead
`|C| / T₀ = 1`. -/
def FanOutBoundPriced_nontrivial : Prop :=
  CKPartition Honest.d Honest.P 1 1 ∧ Honest.O.WholeIn Honest.P ∧ Honest.O.FanOut 1 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧
    ¬ Assumptions.Priced Honest.d Honest.O (fun _ => 2) 0 1 (fun _ => ∅) ∧
    ¬ ((1 : ℕ) : ℝ) * 2 * Honest.M.J.card / (1 * ((1 : ℕ) : ℝ) * ((1 : ℕ) + 0)) ≤
      Honest.d.D.size Honest.d.D.computed / 1

/-- **The LDL route's premises hold** at `Honest`, at `t = 0`, `aBlk = 0`, and the activation rate `β = 1` at deficit
`Λ = 0`: the product's gate holds its representative, the run takes the registration, and the decoder that prints its
program evaluates the block. -/
def PricedOfActivationRate_honest : Prop :=
  (∀ p ∈ Honest.M.J, Honest.M.rep p ∈ Honest.gates p) ∧ Honest.R.Consistent (Honest.d.D.eval Honest.inp) ∧
    Ldl.Assumptions.EvaluatesBlocks Honest.D 0 Honest.R Honest.a 0 ∧
    Ldl.Assumptions.ActivationRate 1 0 0 Honest.D Honest.R Honest.M Honest.a (Honest.d.D.eval Honest.inp)

/-- **The LDL route is false without the activation rate**: at `Honest` with `β = 2`, the representatives, `β ≥ 0`, the
registration and the decoder's evaluation hold, the activation rate fails, and at the block the price
`β |Ops(I)| − (Λ + ⌈log₂ B⌉ + aBlk) = 2` is above its one unregistered input bit. -/
def PricedOfActivationRate_nontrivial : Prop :=
  (∀ p ∈ Honest.M.J, Honest.M.rep p ∈ Honest.gates p) ∧ (0 : ℝ) ≤ 2 ∧
    Honest.R.Consistent (Honest.d.D.eval Honest.inp) ∧ Ldl.Assumptions.EvaluatesBlocks Honest.D 0 Honest.R Honest.a 0 ∧
    ¬ Ldl.Assumptions.ActivationRate 2 0 0 Honest.D Honest.R Honest.M Honest.a (Honest.d.D.eval Honest.inp) ∧
    ¬ ∀ u, ∑ _v ∈ Honest.O.ops (Honest.R.P.gates u), (2 : ℝ) - (0 + Nat.clog 2 1 + (0 : ℕ)) ≤
      novelReads Honest.d (Honest.R.P.gates u)

end Nci.Hardness.Guarantees
