module

public import Definitions.Claims.Sampling
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
import Mathlib.RingTheory.WittVector.IsPoly
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.Monotonicity.Lemmas
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
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Presheaf

/-! **The confidence set is closed downward** at a one-class uniform `kd`-subset design (`d.rules = [.subset kd]`, as
`designOf kd` is): a subset of a set in `M_δ` is in `M_δ`, since dropping members of `B` only raises
`P_a(B) = C(n − |B|, kd)/C(n, kd)`. `RelS` is `faulty ∈ M_δ`, so where one commitment's incorrect set holds another's,
`RelS` failing at the smaller fails it at the larger. -/

@[expose] public section

namespace Claims.Sampling

theorem window_mono {o n : ℕ} {B B' : Finset ℕ} (h : B' ⊆ B) : window o n B' ⊆ window o n B :=
  Finset.image_subset_image (Finset.filter_subset_filter _ h)

theorem classesPa_nil_right (rs : List OneStage) (o : ℕ) (B : Finset ℕ) : classesPa rs [] o B = 1 := by
  cases rs <;> rfl

theorem OneStage.Pa_subset_def (kd : ℕ) (ns : List ℕ) (B : Finset ℕ) :
    (OneStage.subset kd).Pa ns B = ((ns.sum - B.card).choose kd : ℝ) / (ns.sum.choose kd : ℝ) := rfl

/-- **`P_a` is antitone** at a one-class uniform `kd`-subset design. -/
theorem Design.Pa_anti_subset {d : Design} {kd : ℕ} (hd : d.rules = [.subset kd]) (P : Partition)
    {B B' : Finset ℕ} (h : B' ⊆ B) : d.Pa P B ≤ d.Pa P B' := by
  obtain ⟨_ | ⟨c, cs⟩⟩ := P
  · simp only [Design.Pa, classesPa_nil_right, le_refl]
  · simp only [Design.Pa, hd, classesPa, OneStage.Pa_subset_def, mul_one]
    refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
    exact_mod_cast Nat.choose_le_choose kd (Nat.sub_le_sub_left (Finset.card_le_card (window_mono h)) _)

/-- **`M_δ` is closed downward** at a one-class uniform `kd`-subset design. -/
theorem Design.M_down_subset {d : Design} {kd : ℕ} (hd : d.rules = [.subset kd]) (P : Partition) (δ : ℝ)
    {B B' : Finset ℕ} (h : B' ⊆ B) (hB : B ∈ d.M P δ) : B' ∈ d.M P δ :=
  lt_of_lt_of_le hB (Design.Pa_anti_subset hd P h)

end Claims.Sampling
