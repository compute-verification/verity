module

public import Proofs.Flock.Bridge.Lowers
public import Proofs.Flock.Bridge.Statement
public import Proofs.Flock.EndToEndHidden.Statement
public import Proofs.Flock.Claims.Sampling.EndToEndSample
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
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init

/-!
# The bridge's slot count, `P_a` and scope

`partOf P κ ns` is one class, the population's `n` proof blocks in strata of sizes `ns` (`blocks_length_partOf`), each
unit's block holding the input rows it owns. A one-stage design `r` misses a set `B` on it with `r`'s own miss on the
strata `ns` at `B`'s population slots (`pa_partOf_window`); `subset kd`'s is the subset miss, whatever the strata
(`pa_partOf_subset`), so `C(n − K₀, kd)/C(n, kd)` at the first `K₀` slots (`pa_subset_range`) and `0` under `All`'s
`subset n` at any `B` holding a slot (`pa_all`); `pa_partOf`: `P_a(B) = r.Pa ns B` for `B` of population slots.

Scope: `designOf r` is well formed on `partOf P κ ns` when `r` is well formed on the strata `ns`
(`designOf_wellFormed`).
-/

@[expose] public section

namespace Flock.Bridge

open Spot.Sampling

variable {C : Verity.Boolean.Circuit} {n : ℕ} {κ : Fin C.N → Option Bool}

theorem blocks_length_partOf (P : Spot.Sampling.Partition C n) {ns : List ℕ} (hns : ns.sum = n) :
    (partOf P κ ns).blocks.length = n := by
  simp [blocks_partOf P hns]

theorem window_zero_filter (n : ℕ) (B : Finset ℕ) : Claims.Sampling.window 0 n B = B.filter (· < n) := by
  ext v
  simp [Claims.Sampling.window]

/-- `designOf r`'s `P_a` on `partOf P κ ns`: `r`'s miss on the strata `ns` at `B`'s population slots. -/
theorem pa_partOf_window (P : Spot.Sampling.Partition C n) {ns : List ℕ} (hns : ns.sum = n)
    (r : Claims.Sampling.OneStage) (B : Finset ℕ) :
    (designOf r).Pa (partOf P κ ns) B = r.Pa ns (Claims.Sampling.window 0 n B) := by
  show Claims.Sampling.classesPa [r] [ns.splitLengths ((List.finRange n).map (blockOf P κ))] 0 B = _
  simp only [Claims.Sampling.classesPa, flatten_strata_partOf P hns, strataSizes_partOf P hns, List.length_map,
    List.length_finRange, mul_one]

/-- `subset kd`'s `P_a` on `partOf P κ ns`: the population's subset miss at `B`'s population slots, whatever the
strata. -/
theorem pa_partOf_subset (P : Spot.Sampling.Partition C n) {ns : List ℕ} (hns : ns.sum = n) (kd : ℕ)
    (B : Finset ℕ) :
    (designOf (.subset kd)).Pa (partOf P κ ns) B =
      ((n - (Claims.Sampling.window 0 n B).card).choose kd : ℝ) / (n.choose kd : ℝ) := by
  rw [pa_partOf_window P hns]
  simp only [Claims.Sampling.OneStage.Pa, hns]

/-- `subset kd`'s `P_a` on `partOf P κ ns` at the first `K₀` population slots: `C(n − K₀, kd)/C(n, kd)`. -/
theorem pa_subset_range (P : Spot.Sampling.Partition C n) {ns : List ℕ} (hns : ns.sum = n) (kd : ℕ) {K₀ : ℕ}
    (hK₀ : K₀ ≤ n) :
    (designOf (.subset kd)).Pa (partOf P κ ns) (Finset.range K₀) = ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) := by
  have hF : (Finset.range K₀).filter (· < n) = Finset.range K₀ :=
    Finset.filter_true_of_mem fun v hv => (Finset.mem_range.1 hv).trans_le hK₀
  rw [pa_partOf_subset P hns, window_zero_filter, hF, Finset.card_range]

/-- **`All` misses no proof block**: `subset n`'s `P_a` on `partOf P κ ns`, `All`'s sampling design over the `n` units, is `0` at any
`B` holding a population slot. -/
theorem pa_all (P : Spot.Sampling.Partition C n) {ns : List ℕ} (hns : ns.sum = n) {B : Finset ℕ}
    (hB : ∃ v ∈ B, v < n) : (designOf (.subset n)).Pa (partOf P κ ns) B = 0 := by
  obtain ⟨v, hv, hvn⟩ := hB
  have h1 : 0 < (B.filter (· < n)).card := Finset.card_pos.2 ⟨v, Finset.mem_filter.2 ⟨hv, hvn⟩⟩
  have hlt : n - (B.filter (· < n)).card < n := by omega
  rw [pa_partOf_subset P hns, window_zero_filter, Nat.choose_eq_zero_of_lt hlt, Nat.cast_zero, zero_div]

theorem pa_partOf (P : Spot.Sampling.Partition C n) {ns : List ℕ} (hns : ns.sum = n)
    (r : Claims.Sampling.OneStage) (B : Finset ℕ) (hB : B ⊆ Finset.range n) :
    (designOf r).Pa (partOf P κ ns) B = r.Pa ns B := by
  rw [pa_partOf_window P hns, Claims.Sampling.EndToEndSample.window_zero fun v hv => Finset.mem_range.1 (hB hv)]

/-! ## Scope -/

/-- **`designOf r` is well formed on `partOf P κ ns`**: one rule for one class, `r`, well formed on the strata `ns`. -/
theorem designOf_wellFormed (P : Spot.Sampling.Partition C n) {ns : List ℕ} (hns : ns.sum = n)
    {r : Claims.Sampling.OneStage} (hr : r.WellFormed ns) : (designOf r).WellFormed (partOf P κ ns) := by
  refine ⟨rfl, fun p hp => ?_⟩
  simp only [designOf, classes_partOf, List.zip_cons_cons, List.zip_nil_right, List.mem_cons,
    List.not_mem_nil, or_false] at hp
  subst hp
  show r.WellFormed (Claims.Sampling.strataSizes (ns.splitLengths ((List.finRange n).map (blockOf P κ))))
  rw [strataSizes_partOf P hns]
  exact hr

end Flock.Bridge
