module

public import Proofs.Flock.Bridge.Lowers
public import Proofs.Flock.Bridge.Targets
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

/-! The bridge's slot count, `P_a`, count lemma and scope. -/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness.Audit

variable {C : Verity.Boolean.Circuit} {n : ℕ} {κ : Fin C.N → Option Bool}

theorem blocks_length_partOf (P : FlockSoundness.Audit.Partition C n) {ns : List ℕ} (hns : ns.sum = n) :
    (partOf P κ ns).blocks.length = n := by
  simp [blocks_partOf P hns]

theorem partBlocks : PartBlocksTarget := fun P _ _ hns => blocks_length_partOf P hns

theorem window_zero_filter (n : ℕ) (B : Finset ℕ) : Claims.Sampling.window 0 n B = B.filter (· < n) := by
  ext v
  simp [Claims.Sampling.window]

theorem pa_partOf_window (P : FlockSoundness.Audit.Partition C n) {ns : List ℕ} (hns : ns.sum = n)
    (r : Claims.Sampling.OneStage) (B : Finset ℕ) :
    (designOf r).Pa (partOf P κ ns) B = r.Pa ns (Claims.Sampling.window 0 n B) := by
  show Claims.Sampling.classesPa [r] [ns.splitLengths ((List.finRange n).map (blockOf P κ))] 0 B = _
  simp only [Claims.Sampling.classesPa, flatten_strata_partOf P hns, strataSizes_partOf P hns, List.length_map,
    List.length_finRange, mul_one]

theorem pa_partOf_subset (P : FlockSoundness.Audit.Partition C n) {ns : List ℕ} (hns : ns.sum = n) (kd : ℕ)
    (B : Finset ℕ) :
    (designOf (.subset kd)).Pa (partOf P κ ns) B =
      ((n - (Claims.Sampling.window 0 n B).card).choose kd : ℝ) / (n.choose kd : ℝ) := by
  rw [pa_partOf_window P hns]
  simp only [Claims.Sampling.OneStage.Pa, hns]

theorem pa_subset_range (P : FlockSoundness.Audit.Partition C n) {ns : List ℕ} (hns : ns.sum = n) (kd : ℕ) {K₀ : ℕ}
    (hK₀ : K₀ ≤ n) :
    (designOf (.subset kd)).Pa (partOf P κ ns) (Finset.range K₀) = ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) := by
  have hF : (Finset.range K₀).filter (· < n) = Finset.range K₀ :=
    Finset.filter_true_of_mem fun v hv => (Finset.mem_range.1 hv).trans_le hK₀
  rw [pa_partOf_subset P hns, window_zero_filter, hF, Finset.card_range]

theorem pa_all (P : FlockSoundness.Audit.Partition C n) {ns : List ℕ} (hns : ns.sum = n) {B : Finset ℕ}
    (hB : ∃ v ∈ B, v < n) : (designOf (.subset n)).Pa (partOf P κ ns) B = 0 := by
  obtain ⟨v, hv, hvn⟩ := hB
  have h1 : 0 < (B.filter (· < n)).card := Finset.card_pos.2 ⟨v, Finset.mem_filter.2 ⟨hv, hvn⟩⟩
  have hlt : n - (B.filter (· < n)).card < n := by omega
  rw [pa_partOf_subset P hns, window_zero_filter, Nat.choose_eq_zero_of_lt hlt, Nat.cast_zero, zero_div]

theorem pa_partOf (P : FlockSoundness.Audit.Partition C n) {ns : List ℕ} (hns : ns.sum = n)
    (r : Claims.Sampling.OneStage) (B : Finset ℕ) (hB : B ⊆ Finset.range n) :
    (designOf r).Pa (partOf P κ ns) B = r.Pa ns B := by
  rw [pa_partOf_window P hns, Claims.Sampling.EndToEndSample.window_zero fun v hv => Finset.mem_range.1 (hB hv)]

theorem pa : PaTarget := fun P κ _ hns r B hB => pa_partOf (κ := κ) P hns r B hB

theorem choose_lt_of_lt {a b k : ℕ} (hk : 1 ≤ k) (hab : a < b) (hkb : k ≤ b) : a.choose k < b.choose k := by
  obtain ⟨m, rfl⟩ : ∃ m, b = m + 1 := ⟨b - 1, by omega⟩
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  rw [Nat.choose_succ_succ']
  have h1 : a.choose (j + 1) ≤ m.choose (j + 1) := Nat.choose_le_choose _ (by omega)
  have h2 : 0 < m.choose j := Nat.choose_pos (by omega)
  omega

theorem count : CountTarget := by
  classical
  intro C n P ns hns kd K₀ hkd hkdn H Cm κ desc q roots w h
  have hle : (designOf (.subset kd)).Pa (partOf P κ ns)
      (Claims.Sampling.faulty H Cm (stmtOf P κ (.subset kd) desc q roots) (partOf P κ ns) w) ≤
      ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) := not_lt.1 h
  rw [pa_partOf_subset P hns, window_zero_filter] at hle
  have hFn : ((Claims.Sampling.faulty H Cm (stmtOf P κ (.subset kd) desc q roots) (partOf P κ ns) w).filter
      (· < n)).card ≤ n := by
    calc _ ≤ (Finset.range n).card :=
          Finset.card_le_card fun v hv => Finset.mem_range.2 (Finset.mem_filter.1 hv).2
      _ = n := Finset.card_range n
  generalize (Claims.Sampling.faulty H Cm (stmtOf P κ (.subset kd) desc q roots) (partOf P κ ns) w).filter (· < n) = F
    at hle hFn ⊢
  have hpos : (0 : ℝ) < (n.choose kd : ℝ) := by exact_mod_cast Nat.choose_pos hkdn
  have hc : (n - F.card).choose kd ≤ (n - K₀).choose kd := by
    exact_mod_cast (div_le_div_iff_of_pos_right hpos).1 hle
  by_contra hlt
  rw [not_le, lt_min_iff] at hlt
  obtain ⟨h1, h2⟩ := hlt
  have := choose_lt_of_lt hkd (show n - K₀ < n - F.card by omega) (show kd ≤ n - F.card by omega)
  omega

theorem designOf_wellFormed (P : FlockSoundness.Audit.Partition C n) {ns : List ℕ} (hns : ns.sum = n)
    {r : Claims.Sampling.OneStage} (hr : r.WellFormed ns) : (designOf r).WellFormed (partOf P κ ns) := by
  refine ⟨rfl, fun p hp => ?_⟩
  simp only [designOf, classes_partOf, List.zip_cons_cons, List.zip_nil_right, List.mem_cons,
    List.not_mem_nil, or_false] at hp
  subst hp
  show r.WellFormed (Claims.Sampling.strataSizes (ns.splitLengths ((List.finRange n).map (blockOf P κ))))
  rw [strataSizes_partOf P hns]
  exact hr

end Flock.Bridge
