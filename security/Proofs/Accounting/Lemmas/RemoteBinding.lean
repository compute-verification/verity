module

public import Definitions.Accounting.Remote.Fork
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# The receipts' binding bound in its fork form: a lemma

`ReceiptsBindingOfFork`, the step of `Definitions.Accounting.Guarantees.RemoteBinding`'s `ReceiptsBindingOfCR` that
assumes no hash: the binding term is at most `Σ_t √(|L_t|·s_t·(fork_t + split_t))`. `ReceiptsBindingOfCR` is it with each
window's fork finder meeting `Claims.StrictBirthday`, and is the guarantee `ledger.binding` charges. This is a
`def G : Prop` under the guarantees' namespace, proved by the `SecurityProofs` theorem of the same name, and audited with
the proofs: no record pins it.
-/

@[expose] public section

namespace Accounting.Guarantees.Remote

open Accounting.Remote.Draw

/-- **The receipts' binding term, in its fork form.** For a session `D` on the first blocks `prf` and fallbacks `rest`,
its receipts `R`, and sets `L_t` holding every receipt window `t` opens, the binding term is at most
`Σ_t √(|L_t|·s_t·(fork_t + split_t))`: `fork_t` the chance that two runs from window `t`'s cut on independent openings
coins hash a SHA-512 collision between the checks of one receipt's two openings (`Receipts.fork`), `split_t` the chance
that their roots split (`Receipts.split`). -/
def ReceiptsBindingOfFork : Prop :=
  ∀ {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} [Fintype X] (D : Session n m s κ X) (R : Receipts n κ X)
    (prf : Fin n → Pous.Bits κ → Blocks m) (rest : Coins n κ X → Fin n → ℕ → ℕ) (L : Fin n → Finset ℕ),
    (∀ t ω, D.opened prf rest t ω ⊆ L t) →
    (R.binding D prf rest).toReal ≤
      ∑ t, √((L t).card * s t * ((R.fork D prf rest t).toReal + (R.split D prf rest t).toReal))

end Accounting.Guarantees.Remote
