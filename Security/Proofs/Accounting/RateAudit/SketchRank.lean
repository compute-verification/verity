import Mathlib
import Proofs.Accounting.RateAudit.Sketch

/-!
# Signs of rank `r`: the tensor-core sketch's fiber is at most `256^(32 − r)`

Two contents of one sector that give one sketch, with the rest of the tile fixed, have equal projections under the
sector's block of signs, exactly (`project_eq_of_sketch_eq`, `SketchBinds`'s argument). The block's columns over `ℚ`
hold `rank` independent ones (`exists_linearIndependent'`); two such contents that agree off those columns agree on
them too, so the contents of a fiber are told apart by their other `32 − rank` bytes (`card_fiber_le`). The guarantees
are `Accounting.SecurityProofs.SketchRankFiber` and `RateAuditTensorSketchRank`.
-/

namespace Accounting.SecurityProofs

open Accounting.RateSketch
open Pous (pr)
open scoped ENNReal
open Bandwidth (pr_mono)

theorem int8_injective : Function.Injective int8 := by
  intro a b h
  unfold int8 at h
  have := a.isLt
  have := b.isLt
  ext
  split_ifs at h <;> omega

set_option linter.constructorNameAsVariable false in
/-- Two contents of sector `u` that give one sketch, with the rest of the tile fixed, have one projection under the
sector's block. -/
theorem project_eq_of_sketch_eq {k m : ℕ} (σ : SketchSigns k m) (x : Fin m → Sector) (u : Fin m) (a b : Sector)
    (h : sketch σ (Function.update x u a) = sketch σ (Function.update x u b)) :
    Accounting.Tag.project (σ.block u) (fun l => int8 (a l)) =
      Accounting.Tag.project (σ.block u) (fun l => int8 (b l)) := by
  refine funext fun i => ?_
  have := congrFun h i
  simp only [sketch] at this
  have hrest : ∀ c : Sector,
      ∑ u' ∈ Finset.univ.erase u, Accounting.Tag.project (σ.block u') (fun l => int8 (Function.update x u c u' l)) i =
        ∑ u' ∈ Finset.univ.erase u, Accounting.Tag.project (σ.block u') (fun l => int8 (x u' l)) i := fun c =>
    Finset.sum_congr rfl (fun u' hu' => by rw [Function.update_of_ne (Finset.ne_of_mem_erase hu')])
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ u), ← Finset.add_sum_erase _ _ (Finset.mem_univ u), hrest,
    hrest, Function.update_self, Function.update_self] at this
  push_cast at this
  have h2 := add_right_cancel this
  rw [ZMod.intCast_eq_intCast_iff_dvd_sub] at h2
  have ha' := abs_project_le (σ.block u) a i
  have hb' := abs_project_le (σ.block u) b i
  have hlt : |Accounting.Tag.project (σ.block u) (fun l => int8 (b l)) i -
      Accounting.Tag.project (σ.block u) (fun l => int8 (a l)) i| < ((2 ^ 32 : ℕ) : ℤ) := by
    rw [abs_lt]
    rw [abs_le] at ha' hb'
    constructor <;> push_cast <;> linarith [ha'.1, ha'.2, hb'.1, hb'.2]
  exact (sub_eq_zero.1 (Int.eq_zero_of_abs_lt_dvd h2 hlt)).symm

set_option linter.constructorNameAsVariable false in
/-- **The fiber at rank `r`.** If sector `u`'s block has rank at least `r` over `ℚ`, at most `256^(32 − r)` contents of
the sector give the sketch `d` with the rest of the tile `x` fixed. -/
theorem card_fiber_le {k m r : ℕ} (σ : SketchSigns k m) (x : Fin m → Sector) (u : Fin m) (d : Fin k → ZMod (2 ^ 32))
    (hr : r ≤ ((Accounting.Tag.signMatrix (σ.block u)).map (Int.cast : ℤ → ℚ)).rank) :
    (Finset.univ.filter fun a : Sector => sketch σ (Function.update x u a) = d).card ≤ 256 ^ (32 - r) := by
  set A : Matrix (Fin k) (Fin 32) ℚ := (Accounting.Tag.signMatrix (σ.block u)).map (Int.cast : ℤ → ℚ) with hA
  obtain ⟨κ, c, hc, hspan, hli⟩ := exists_linearIndependent' ℚ A.col
  let : Fintype κ := Fintype.ofInjective c hc
  have hrank : A.rank = Fintype.card κ := by
    rw [Matrix.rank_eq_finrank_span_cols, ← hspan, finrank_span_eq_card hli]
  set J : Finset (Fin 32) := Finset.univ.map ⟨c, hc⟩ with hJdef
  have hJ : J.card = Fintype.card κ := by simp [J]
  have key : ∀ a b : Sector, sketch σ (Function.update x u a) = d → sketch σ (Function.update x u b) = d →
      (∀ l, l ∉ J → a l = b l) → a = b := by
    intro a b ha hb hoff
    have hp := project_eq_of_sketch_eq σ x u a b (ha.trans hb.symm)
    set g : κ → ℚ := fun j => ((int8 (a (c j)) - int8 (b (c j)) : ℤ) : ℚ) with hg
    have hsum : ∑ j, g j • (A.col ∘ c) j = 0 := by
      funext i
      have hi := congrFun hp i
      simp only [Accounting.Tag.project] at hi
      have hz : ∑ l, Accounting.Tag.sgn (σ.block u i l) * (int8 (a l) - int8 (b l)) = 0 := by
        simp only [mul_sub, Finset.sum_sub_distrib, hi, sub_self]
      have hz' : ∑ l ∈ J, ((Accounting.Tag.sgn (σ.block u i l) * (int8 (a l) - int8 (b l)) : ℤ) : ℚ) = 0 := by
        rw [Finset.sum_subset (Finset.subset_univ J) (fun l _ hl => by rw [hoff l hl, sub_self, mul_zero,
          Int.cast_zero]), ← Int.cast_sum, hz, Int.cast_zero]
      rw [hJdef, Finset.sum_map] at hz'
      simp only [Finset.sum_apply, Pi.smul_apply, Function.comp_apply, Matrix.col_apply, hA, Matrix.map_apply,
        Accounting.Tag.signMatrix, smul_eq_mul, Pi.zero_apply, hg]
      rw [← hz']
      refine Finset.sum_congr rfl fun j _ => ?_
      simp only [Function.Embedding.coeFn_mk]
      push_cast
      ring
    have hzero := Fintype.linearIndependent_iff.1 hli g hsum
    funext l
    by_cases hl : l ∈ J
    · obtain ⟨j, -, rfl⟩ := Finset.mem_map.1 hl
      have := hzero j
      simp only [hg, Int.cast_eq_zero, sub_eq_zero] at this
      exact int8_injective this
    · exact hoff l hl
  calc (Finset.univ.filter fun a : Sector => sketch σ (Function.update x u a) = d).card
      ≤ (Finset.univ : Finset ({l // l ∉ J} → Fin 256)).card :=
        Finset.card_le_card_of_injOn (fun a l => a l.1) (fun _ _ => Finset.mem_univ _) (fun a ha b hb hab => by
          simp only [Finset.coe_filter, Finset.mem_univ, true_and] at ha hb
          exact key a b ha hb fun l hl => congrFun hab ⟨l, hl⟩)
    _ = 256 ^ (32 - J.card) := by
        rw [Finset.card_univ, Fintype.card_fun, Fintype.card_fin, Fintype.card_subtype_compl, Fintype.card_fin,
          Fintype.card_coe]
    _ ≤ 256 ^ (32 - r) := Nat.pow_le_pow_right (by norm_num) (by rw [hJ, ← hrank]; omega)

/-- **Signs of rank `r` leave the fiber at most `256^(32 − r)`** (`Accounting.Guarantees.SketchRankFiber`). -/
theorem SketchRankFiber : Accounting.Guarantees.SketchRankFiber := by
  intro k m r σ hr
  unfold fib
  refine max_le (Nat.one_le_pow _ _ (by norm_num)) (Finset.sup_le fun p _ => ?_)
  obtain ⟨x, u, d⟩ := p
  exact card_fiber_le σ x u d (hr u)

/-- **The rate audit at the deployed tag, priced by the signs' rank**
(`Accounting.Guarantees.RateAuditTensorSketchRank`). -/
theorem RateAuditTensorSketchRank : Accounting.Guarantees.RateAuditTensorSketchRank := by
  intro N m ρ h S L q s k F cTag cEnc ks r A hr
  refine (RateAuditSketch N m ρ h S L q 256 s k F cTag cEnc (256 ^ (32 - r)) Sector (SketchSigns ks m)
    (Fin ks → ZMod (2 ^ 32)) sketch A).trans ?_
  refine add_le_add (add_le_add le_rfl (pr_mono fun σ hlt => ?_)) (le_of_eq ?_)
  · by_contra hn
    push Not at hn
    exact absurd (SketchRankFiber ks m r σ hn) (not_le.2 hlt)
  · have hX : Fintype.card Sector = 2 ^ 256 := by simp [Sector]
    have h256 : (256 : ℝ≥0∞) ^ (32 - r) = 2 ^ (256 - 8 * r) := by
      rw [show (256 : ℝ≥0∞) = 2 ^ 8 by norm_num, ← pow_mul, show 8 * (32 - r) = 256 - 8 * r by omega]
    have hsplit : (2 : ℝ≥0∞) ^ 256 = 2 ^ (256 - 8 * r) * 2 ^ (8 * r) := by
      rw [← pow_add, Nat.sub_add_cancel (by omega)]
    rw [hX, Nat.cast_mul, Nat.cast_pow, Nat.cast_pow, Nat.cast_ofNat, Nat.cast_ofNat, h256, hsplit,
      mul_comm (q : ℝ≥0∞), ENNReal.mul_div_mul_left _ _ (by positivity) (ENNReal.pow_ne_top ENNReal.ofNat_ne_top)]

end Accounting.SecurityProofs
