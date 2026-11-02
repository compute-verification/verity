import Proofs.Flock.Claims.Complete.Zk.RankBridge
import Proofs.Flock.Claims.Complete.Zk.RankPrimal
import Proofs.Flock.Level3.Pinned

/-!
# The deployed rank bound

`DeployedRankBound`, proved (`deployedRankBound`): at the deployed table's four points, of ten word coordinates with
the `c` claims' first six pinned, at most `2^-122` of the coins leave the mask slot's weights rank-deficient.
`card_deficient_le` bounds the deficient coins by a term per support `S ≠ ∅`; each of the fifteen is at most
`3·2^-128` of the coins, `45·2^-128 ≤ 2^-122` together. `U_T` is the points' span over the first `T` coordinates.

* `two64_le_ud`: at an odd point the six pinned coordinates' span has `2^64` elements (`pinned_indep`).
* `sum_inv_ud_le`: `E[1/|π_e U_6|^p] ≤ Bc(6, p)` at every point (`chain` at an even one).
* `term_single`: a singleton support, from `U_6`.
* `term_odd`, `term_02`: two or three points, one fresh step from `U_6`, at `|π_S U_6| = 2^64` where an odd point is
  in `S` and by `chain` at `{0, 2}`.
* `term_univ`: all four, one fresh step from `U_7`, where every triple's projection of `U_7` is the whole `2^128`
  (`Good`; `bad_triple_le` where one falls short).
* `maskRankBound_of_small`: `MaskRankBound` from `SmallSlotBound` alone, so `maskRankBound`: `MaskRankBound` holds,
  `SmallSlotBound` by the verifier's floor (`smallSlotBound`).
-/

namespace FlockComplete.MaskRank

open Finset FlockLevel3

section BcFacts

theorem Bc_zero_right (q : ℝ) : ∀ t, Bc q t 0 = 1
  | 0 => by simp [Bc]
  | t + 1 => by simp [Bc, cB, Bc_zero_right q t]

theorem Bc_nonneg {q : ℝ} (hq : 0 ≤ q) : ∀ t p, 0 ≤ Bc q t p
  | 0, p => by simp only [Bc]; positivity
  | t + 1, p => by
    simp only [Bc]
    have := cB_nonneg hq p
    have := Bc_nonneg hq t (2 * p)
    have := Bc_nonneg hq t (p - 1)
    positivity

/-- `Bc` is at least its no-collision term `(1/2)^(2^t·p)`. -/
theorem Bc_ge {q : ℝ} (hq : 0 ≤ q) : ∀ t p, (1 / 2 : ℝ) ^ (2 ^ t * p) ≤ Bc q t p
  | 0, p => by simp [Bc]
  | t + 1, p => by
    simp only [Bc]
    have h1 := Bc_ge hq t (2 * p)
    have h2 := mul_nonneg (cB_nonneg hq p) (Bc_nonneg hq t (p - 1))
    calc (1 / 2 : ℝ) ^ (2 ^ (t + 1) * p) = (1 / 2) ^ (2 ^ t * (2 * p)) := by
          congr 1
          rw [pow_succ]
          ring
      _ ≤ _ := le_add_of_le_of_nonneg h1 h2

end BcFacts

section Deployed

open Classical

/-- The deployed points' span over the first `T` coordinates. -/
noncomputable def Ud (r : Fin 4 → Fin 10 → GF128) (T : ℕ) : AddSubgroup (Fin 4 → GF128) :=
  spanX (col deployedFix r) (pre 10 T)

/-- Point `e`'s projection over the first six coordinates. -/
noncomputable def ud (e : Fin 4) (r : Fin 4 → Fin 10 → GF128) : ℕ := csz (projS {e} (Ud r 6))

theorem deployedFix_late (i : Fin 4) {t : Fin 10} (ht : 6 ≤ t.val) : deployedFix i t = none := by
  unfold deployedFix
  split_ifs with h
  · omega
  · rfl

theorem deployedFix_even {i : Fin 4} (hi : i.val % 2 = 0) (t : Fin 10) : deployedFix i t = none := by
  unfold deployedFix
  split_ifs with h
  · omega
  · rfl

theorem card_GF : (Fintype.card GF128 : ℝ) = 2 ^ 128 := by
  rw [card_F]
  norm_num

theorem csz_Ud_le (r : Fin 4 → Fin 10 → GF128) {T : ℕ} (hT : T ≤ 10) : csz (Ud r T) ≤ 2 ^ 2 ^ T :=
  csz_spanX_pre_le _ T hT

theorem ud_le (e : Fin 4) (r : Fin 4 → Fin 10 → GF128) : ud e r ≤ 2 ^ 64 :=
  (csz_projS_le _ _).trans ((csz_Ud_le r (by norm_num)).trans (by norm_num))

theorem pre_seven : pre 10 7 = {0, 1, 2, 3, 4, 5, 6} := by decide

/-- **The pinned rows' size**: at an odd point, the six pinned coordinates' span has `2^64` elements. With the
seventh pinned value `pinnedF 0` as a further coordinate, the `128` eq-products are `GF(2)`-independent
(`pinned_indep`), so that span has `2^128` elements, at most the six's size squared. -/
theorem two64_le_ud {e : Fin 4} (he : e.val % 2 = 1) (r : Fin 4 → Fin 10 → GF128) : 2 ^ 64 ≤ ud e r := by
  set c := col deployedFix r with hc
  set c' : Fin 10 → Fin 4 → GF128 := Function.update c 6 (fun _ => pinnedF 0) with hc'
  have h6 : (6 : Fin 10) ∉ pre 10 6 := by decide
  have hU : spanX c' (pre 10 6) = Ud r 6 := spanX_congr fun t ht => by
    have : t ≠ 6 := ne_of_mem_of_not_mem ht h6
    rw [hc', Function.update_of_ne this, hc]
  have h7 : pre 10 7 = insert (6 : Fin 10) (pre 10 6) := pre_succ (D := 10) (T := 6) (by norm_num)
  set W := spanX c' (pre 10 7) with hWdef
  have hW : projS {e} W =
      projS {e} (Ud r 6) ⊔ (projS {e} (Ud r 6)).map (AddMonoidHom.mulLeft (c' 6)) := by
    rw [hWdef, h7, spanX_insert _ h6, hU, projS_sup_mul]
  have hsq : csz (projS {e} W) ≤ ud e r ^ 2 := by
    rw [hW]
    have h1 := csz_sup_mul_Kc (projS {e} (Ud r 6)) (c' 6)
    have h2 := one_le_Kc (projS {e} (Ud r 6)) (c' 6)
    unfold ud
    calc _ ≤ csz (projS {e} (Ud r 6) ⊔ (projS {e} (Ud r 6)).map (AddMonoidHom.mulLeft (c' 6))) *
          Kc (projS {e} (Ud r 6)) (c' 6) := Nat.le_mul_of_pos_right _ h2
      _ = _ := h1
  let xb : Fin 128 → Fin 10 → Bool := fun b t =>
    if t.val < 6 then b.val.testBit (t.val + 1) else b.val.testBit 0
  let gen : Fin 128 → Fin 4 → GF128 := fun b => ∏ t ∈ pre 10 7, if xb b t then c' t else 1 + c' t
  have hgen : ∀ b, gen b ∈ W := fun b => AddSubgroup.subset_closure ⟨xb b, rfl⟩
  have hgen_e : ∀ b, gen b e = eqAtPinned b.val := by
    intro b
    simp only [gen, Finset.prod_apply]
    unfold eqAtPinned
    refine Finset.prod_nbij' (fun t => (⟨(t.val + 1) % 7, Nat.mod_lt _ (by norm_num)⟩ : Fin 7))
      (fun k => (⟨(k.val + 6) % 7, by omega⟩ : Fin 10)) ?_ ?_ ?_ ?_ ?_
    · intro t _
      exact Finset.mem_univ _
    · intro k _
      simp only [pre, Finset.mem_filter, Finset.mem_univ, true_and]
      show (k.val + 6) % 7 < 7
      omega
    · intro t ht
      simp only [pre, Finset.mem_filter, Finset.mem_univ, true_and] at ht
      ext
      show ((t.val + 1) % 7 + 6) % 7 = t.val
      omega
    · intro k _
      ext
      show ((k.val + 6) % 7 + 1) % 7 = k.val
      omega
    · intro t ht
      simp only [pre, Finset.mem_filter, Finset.mem_univ, true_and] at ht
      rw [apply_ite (fun f : Fin 4 → GF128 => f e), Pi.add_apply, Pi.one_apply]
      by_cases h6 : t.val < 6
      · have hne : t ≠ 6 := fun h => by subst h; simp at h6
        have hmod : (t.val + 1) % 7 = t.val + 1 := Nat.mod_eq_of_lt (by omega)
        have hval : c' t e = Flock.pinned[t.val + 1]! := by
          rw [hc', Function.update_of_ne hne, hc]
          simp only [col, deployedFix, he, h6, and_self, ↓reduceIte]
          rfl
        have hx : xb b t = b.val.testBit (t.val + 1) := by simp only [xb, h6, ↓reduceIte]
        rw [hval, hx]
        simp only [pinnedF, GF128.of, hmod]
      · have ht6 : t = 6 := by
          ext
          show t.val = 6
          omega
        subst ht6
        have hx : xb b 6 = b.val.testBit 0 := by simp [xb]
        have h6' : c' 6 e = pinnedF 0 := by simp [c']
        rw [hx, h6']
        rfl
  let Φ : (Fin 128 → ZMod 2) → Fin 4 → GF128 := fun cv => ∑ b, if cv b = 1 then zS {e} (gen b) else 0
  have hΦmem : ∀ cv, Φ cv ∈ projS {e} W := fun cv =>
    AddSubgroup.sum_mem _ fun b _ => by
      split
      · exact ⟨gen b, hgen b, rfl⟩
      · exact (projS {e} W).zero_mem
  have hite : ∀ (z : ZMod 2) (x : GF128), (if z = 1 then x else 0) = (z.val : GF128) * x := by
    intro z x
    by_cases hz : z = 1
    · subst hz
      have h1 : (1 : ZMod 2).val = 1 := rfl
      simp [h1]
    · have hz0 : z = 0 := by
        by_contra h
        exact hz (zmod2_eq_one h)
      subst hz0
      have h0 : (0 : ZMod 2).val = 0 := rfl
      simp [h0, hz]
  have hΦe : ∀ cv, Φ cv e = ∑ b : Fin 128, ((cv b).val : GF128) * eqAtPinned b.val := by
    intro cv
    simp only [Φ, Finset.sum_apply]
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [← hite, apply_ite (fun f : Fin 4 → GF128 => f e)]
    simp [zS, hgen_e]
  have hinj : Set.InjOn Φ (univ : Finset (Fin 128 → ZMod 2)) := by
    intro cv _ cv' _ h
    have h1 := congrFun h e
    rw [hΦe, hΦe] at h1
    have e1 : ∀ x : Fin 128, (((cv - cv') x).val : GF128) * eqAtPinned x.val =
        ((cv x).val : GF128) * eqAtPinned x.val - ((cv' x).val : GF128) * eqAtPinned x.val := by
      intro x
      rw [Pi.sub_apply, ZMod.natCast_val, ZMod.natCast_val, ZMod.natCast_val, ZMod.cast_sub (dvd_refl 2), sub_mul]
    have h3 : ∑ x : Fin 128, (((cv x).val : GF128) * eqAtPinned x.val -
        ((cv' x).val : GF128) * eqAtPinned x.val) = (∑ x : Fin 128, ((cv x).val : GF128) * eqAtPinned x.val) -
          ∑ x : Fin 128, ((cv' x).val : GF128) * eqAtPinned x.val := Finset.sum_sub_distrib _ _
    have h2 := pinned_indep (cv - cv') (by
      rw [Finset.sum_congr rfl fun x _ => e1 x, h3, h1, sub_self])
    exact sub_eq_zero.1 h2
  have hbig : 2 ^ 128 ≤ csz (projS {e} W) := by
    have hcard : (univ : Finset (Fin 128 → ZMod 2)).card = 2 ^ 128 := by
      simp [Finset.card_univ, ZMod.card]
    rw [← hcard]
    refine Finset.card_le_card_of_injOn Φ (fun cv _ => ?_) hinj
    simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq]
    exact hΦmem cv
  have : (2 ^ 64) ^ 2 ≤ ud e r ^ 2 := le_trans (by norm_num) (hbig.trans hsq)
  exact (Nat.pow_le_pow_iff_left (by norm_num)).1 this

/-- **One point's inverse moments**: `E_r[1/|π_e U_6|^p] ≤ Bc(6, p)`, by the chain at an even point and the pinned
size at an odd one. -/
theorem sum_inv_ud_le (e : Fin 4) (p : ℕ) :
    ∑ r : Fin 4 → Fin 10 → GF128, 1 / (ud e r : ℝ) ^ p ≤
      Fintype.card (Fin 4 → Fin 10 → GF128) * Bc (Fintype.card GF128) 6 p := by
  have hq : (0 : ℝ) ≤ Fintype.card GF128 := Nat.cast_nonneg _
  rcases Nat.mod_two_eq_zero_or_one e.val with he | he
  · let j : Fin 4 := ⟨(e.val + 2) % 4, by omega⟩
    have hj : e ≠ j := fun h => by
      rw [Fin.ext_iff] at h
      simp only [j] at h
      omega
    have hje : j.val % 2 = 0 := by
      simp only [j]
      omega
    have h := chain deployedFix hj (deployedFix_even he) (deployedFix_even hje) 6 (by norm_num) p 0
    simp only [pow_zero, mul_one, Bc_zero_right] at h
    exact h
  · have hpt : ∀ r, 1 / (ud e r : ℝ) ^ p ≤ Bc (Fintype.card GF128) 6 p := by
      intro r
      have h64 : (2 : ℝ) ^ 64 ≤ ud e r := by exact_mod_cast two64_le_ud he r
      calc 1 / (ud e r : ℝ) ^ p ≤ 1 / ((2 : ℝ) ^ 64) ^ p := by gcongr
        _ = (1 / 2) ^ (2 ^ 6 * p) := by
          rw [← pow_mul, one_div_pow]
          norm_num
        _ ≤ _ := Bc_ge hq 6 p
    calc _ ≤ ∑ _r : Fin 4 → Fin 10 → GF128, Bc (Fintype.card GF128) 6 p := Finset.sum_le_sum fun r _ => hpt r
      _ = _ := by rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]

end Deployed

section Terms

open Classical

theorem one_le_supp {w : Fin 4 → GF128} (hw : w ≠ 0) : 1 ≤ (supp w).card := by
  obtain ⟨k, hk⟩ : ∃ k, w k ≠ 0 := by
    by_contra h
    push Not at h
    exact hw (funext h)
  exact Finset.card_pos.2 ⟨k, Finset.mem_filter.2 ⟨mem_univ _, hk⟩⟩

theorem csz_projS_odd {S : Finset (Fin 4)} (hS : ∃ e ∈ S, e.val % 2 = 1) (r : Fin 4 → Fin 10 → GF128) :
    csz (projS S (Ud r 6)) = 2 ^ 64 := by
  obtain ⟨e, heS, he⟩ := hS
  refine le_antisymm ((csz_projS_le _ _).trans ((csz_Ud_le r (by norm_num)).trans (by norm_num))) ?_
  exact (two64_le_ud he r).trans (csz_projS_mono (Finset.singleton_subset_iff.2 heS) _)

theorem arith_ker {k u : ℕ} (hk1 : 1 ≤ k) (hu : 0 < u) (h : k * u ≤ 2 ^ 64) :
    ((k : ℝ) - 1) * k ≤ 2 ^ 128 / (u : ℝ) ^ 2 - 1 := by
  have hk1' : (1 : ℝ) ≤ k := by exact_mod_cast hk1
  have hu' : (0 : ℝ) < u := by exact_mod_cast hu
  have hku : (k : ℝ) * u ≤ 2 ^ 64 := by exact_mod_cast h
  have hk : (k : ℝ) ≤ 2 ^ 64 / u := by rw [le_div_iff₀ hu']; exact hku
  have hk0 : (0 : ℝ) ≤ k := by linarith
  calc ((k : ℝ) - 1) * k ≤ (k : ℝ) ^ 2 - 1 := by nlinarith
    _ ≤ (2 ^ 64 / u) ^ 2 - 1 := by gcongr
    _ = 2 ^ 128 / (u : ℝ) ^ 2 - 1 := by rw [div_pow]; norm_num

theorem arith_u {u : ℕ} (hu : 0 < u) (h : u ≤ 2 ^ 64) : 0 ≤ 2 ^ 128 / (u : ℝ) ^ 2 - 1 := by
  have hu' : (0 : ℝ) < u := by exact_mod_cast hu
  have h1 : (u : ℝ) ≤ 2 ^ 64 := by exact_mod_cast h
  have h2 : (u : ℝ) ^ 2 ≤ 2 ^ 128 := by
    calc (u : ℝ) ^ 2 ≤ (2 ^ 64) ^ 2 := by gcongr
      _ = 2 ^ 128 := by norm_num
  have : 1 ≤ 2 ^ 128 / (u : ℝ) ^ 2 := (one_le_div (by positivity)).2 h2
  linarith

theorem kerE_le_ud {S : Finset (Fin 4)} (hS : ∃ e ∈ S, e.val % 2 = 1) (r : Fin 4 → Fin 10 → GF128) {e : Fin 4}
    (he : e ∈ S) :
    ((kerE (projS S (Ud r 6)) e : ℝ) - 1) * kerE (projS S (Ud r 6)) e ≤ 2 ^ 128 / (ud e r : ℝ) ^ 2 - 1 := by
  have h := kerE_mul_le (projS S (Ud r 6)) e
  rw [projS_projS (Finset.singleton_subset_iff.2 he), csz_projS_odd hS r] at h
  exact arith_ker (one_le_kerE _ _) (csz_pos _) h

theorem Xd_term_nonneg (e : Fin 4) (r : Fin 4 → Fin 10 → GF128) : 0 ≤ 2 ^ 128 / (ud e r : ℝ) ^ 2 - 1 :=
  arith_u (csz_pos _) (ud_le e r)

/-- The six coordinates' excess, summed over the points. -/
noncomputable def Xd (r : Fin 4 → Fin 10 → GF128) : ℝ := ∑ e : Fin 4, (2 ^ 128 / (ud e r : ℝ) ^ 2 - 1)

/-- **The excess at a support with an odd point**: there `|π_S U_6| = 2^64`, and each point's kernel is at most
`2^64/|π_e U_6|`. -/
theorem excess_le {S : Finset (Fin 4)} (hS : ∃ e ∈ S, e.val % 2 = 1) (r : Fin 4 → Fin 10 → GF128) :
    ∑ e ∈ S, ((kerE (projS S (Ud r 6)) e : ℝ) - 1) * kerE (projS S (Ud r 6)) e / (Fintype.card GF128 : ℝ) ^ 1 +
      (csz (projS S (Ud r 6)) : ℝ) ^ 2 / (Fintype.card GF128 : ℝ) ^ S.card ≤
    Xd r / Fintype.card GF128 + 2 ^ 128 / (Fintype.card GF128 : ℝ) ^ S.card := by
  rw [csz_projS_odd hS r, pow_one]
  refine add_le_add ?_ (le_of_eq (by push_cast; ring))
  rw [← Finset.sum_div]
  gcongr
  calc ∑ e ∈ S, ((kerE (projS S (Ud r 6)) e : ℝ) - 1) * kerE (projS S (Ud r 6)) e
      ≤ ∑ e ∈ S, (2 ^ 128 / (ud e r : ℝ) ^ 2 - 1) := Finset.sum_le_sum fun e he => kerE_le_ud hS r he
    _ ≤ Xd r := Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ S) fun e _ _ => Xd_term_nonneg e r

theorem sum_Xd_le : ∑ r, Xd r ≤
    Fintype.card (Fin 4 → Fin 10 → GF128) * (4 * (2 ^ 128 * Bc (Fintype.card GF128) 6 2 - 1)) := by
  unfold Xd
  rw [Finset.sum_comm]
  have h : ∀ e : Fin 4, ∑ r : Fin 4 → Fin 10 → GF128, (2 ^ 128 / (ud e r : ℝ) ^ 2 - 1) ≤
      Fintype.card (Fin 4 → Fin 10 → GF128) * (2 ^ 128 * Bc (Fintype.card GF128) 6 2 - 1) := by
    intro e
    have h1 := sum_inv_ud_le e 2
    have h2 := mul_le_mul_of_nonneg_left h1 (by positivity : (0 : ℝ) ≤ 2 ^ 128)
    rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one]
    have e1 : ∑ r : Fin 4 → Fin 10 → GF128, (2 : ℝ) ^ 128 / (ud e r : ℝ) ^ 2 =
        2 ^ 128 * ∑ r : Fin 4 → Fin 10 → GF128, 1 / (ud e r : ℝ) ^ 2 := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun r _ => by ring
    rw [e1]
    linarith
  calc _ ≤ ∑ _e : Fin 4, Fintype.card (Fin 4 → Fin 10 → GF128) * (2 ^ 128 * Bc (Fintype.card GF128) 6 2 - 1) :=
        Finset.sum_le_sum fun e _ => h e
    _ = _ := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        push_cast
        ring

/-- **A step at a free coordinate `T`**, the coins' column summed out. -/
theorem sum_le_of_step {T : ℕ} (hT : T < 10) (hfreeT : ∀ i, deployedFix i ⟨T, hT⟩ = none)
    (f : AddSubgroup (Fin 4 → GF128) → ℝ) (g : (Fin 4 → Fin 10 → GF128) → ℝ)
    (hg : ∀ r, ∑ a : Fin 4 → GF128, f (Ud r T ⊔ (Ud r T).map (AddMonoidHom.mulLeft a)) ≤
      Fintype.card (Fin 4 → GF128) * g r) :
    ∑ r, f (spanX (col deployedFix r) (pre 10 (T + 1))) ≤ ∑ r, g r := by
  have hA : (0 : ℝ) < Fintype.card (Fin 4 → GF128) := by exact_mod_cast Fintype.card_pos
  have hstep := sum_step deployedFix (not_mem_pre hT) f
  rw [← pre_succ hT] at hstep
  refine le_of_mul_le_mul_left ?_ hA
  rw [hstep, Finset.mul_sum]
  refine Finset.sum_le_sum fun r _ => ?_
  simp only [colA_free hfreeT]
  exact hg r

end Terms

section Numerics

set_option exponentiation.threshold 1024

theorem B62_le : (2 : ℝ) ^ 128 * Bc (2 ^ 128) 6 2 ≤ 1 + 1 / 2 ^ 30 := by
  simp only [Bc, cB]
  norm_num

theorem B64_le : Bc (2 ^ 128 : ℝ) 6 4 ≤ 1 / 2 ^ 200 := by
  simp only [Bc, cB]
  norm_num

theorem B65_le : (2 : ℝ) ^ 128 * Bc (2 ^ 128) 6 5 ≤ 3 / 2 ^ 128 := by
  simp only [Bc, cB]
  norm_num

theorem B62_nonneg : 0 ≤ Bc (2 ^ 128 : ℝ) 6 2 := Bc_nonneg (by positivity) 6 2

end Numerics

section PerSupport

open Classical

set_option exponentiation.threshold 1024

theorem sum_affine_le (K0 K1 B : ℝ) (hK1 : 0 ≤ K1) (x : (Fin 4 → Fin 10 → GF128) → ℝ)
    (hX : ∑ r, x r ≤ Fintype.card (Fin 4 → Fin 10 → GF128) * B) :
    ∑ r, (K0 + K1 * x r) ≤ Fintype.card (Fin 4 → Fin 10 → GF128) * (K0 + K1 * B) := by
  rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, ← Finset.mul_sum]
  have := mul_le_mul_of_nonneg_left hX hK1
  linarith

theorem sum_Xd_le' : ∑ r, Xd r ≤ Fintype.card (Fin 4 → Fin 10 → GF128) * (4 * (2 ^ 128 * Bc (2 ^ 128) 6 2 - 1)) := by
  have h := sum_Xd_le
  rwa [card_GF] at h

/-- **A singleton support**: `E[min(1, |F|/|π_e U_6|^5)] ≤ |F|·Bc(6, 5) ≤ 3·2^-128`. -/
theorem term_single (e : Fin 4) :
    ∑ r : Fin 4 → Fin 10 → GF128, min 1 ((Fintype.card GF128 : ℝ) ^ ({e} : Finset (Fin 4)).card /
      (csz (projS {e} (spanX (col deployedFix r) (pre 10 6))) : ℝ) ^ ((pre 10 6)ᶜ.card + 1)) ≤
    Fintype.card (Fin 4 → Fin 10 → GF128) * (3 / 2 ^ 128) := by
  have hc : (pre 10 6)ᶜ.card + 1 = 5 := by decide
  rw [hc, Finset.card_singleton, pow_one]
  calc _ ≤ ∑ r : Fin 4 → Fin 10 → GF128, (Fintype.card GF128 : ℝ) * (1 / (ud e r : ℝ) ^ 5) :=
        Finset.sum_le_sum fun r _ => (min_le_right _ _).trans (le_of_eq (by rw [mul_one_div]; rfl))
    _ = (Fintype.card GF128 : ℝ) * ∑ r, 1 / (ud e r : ℝ) ^ 5 := by rw [Finset.mul_sum]
    _ ≤ (Fintype.card GF128 : ℝ) * (Fintype.card (Fin 4 → Fin 10 → GF128) * Bc (Fintype.card GF128) 6 5) := by
        gcongr
        exact sum_inv_ud_le e 5
    _ = Fintype.card (Fin 4 → Fin 10 → GF128) * ((2 : ℝ) ^ 128 * Bc (2 ^ 128) 6 5) := by
        rw [card_GF]
        ring
    _ ≤ _ := by
        gcongr
        exact B65_le

theorem arith_odd {s : ℕ} (h2 : 2 ≤ s) (h3 : s ≤ 3) :
    ((2 : ℝ) ^ 128) ^ s / ((2 : ℝ) ^ 64) ^ (2 * 4) ≤ (2 ^ 128) ^ 3 / 2 ^ 512 ∧
      ((2 : ℝ) ^ 128) ^ s / ((2 : ℝ) ^ 64) ^ (2 * 4) * (2 ^ 32) ^ 4 ≤ 1 ∧
        (2 : ℝ) ^ 128 / (2 ^ 128) ^ s ≤ 2 ^ 128 / (2 ^ 128) ^ 2 := by
  have hs3 : ((2 : ℝ) ^ 128) ^ s ≤ (2 ^ 128) ^ 3 := pow_le_pow_right₀ (by norm_num) h3
  have hs2 : ((2 : ℝ) ^ 128) ^ 2 ≤ (2 ^ 128) ^ s := pow_le_pow_right₀ (by norm_num) h2
  refine ⟨?_, ?_, ?_⟩
  · calc _ ≤ ((2 : ℝ) ^ 128) ^ 3 / ((2 : ℝ) ^ 64) ^ (2 * 4) := by gcongr
      _ = _ := by norm_num
  · calc _ ≤ ((2 : ℝ) ^ 128) ^ 3 / ((2 : ℝ) ^ 64) ^ (2 * 4) * (2 ^ 32) ^ 4 := by gcongr
      _ = 1 := by norm_num
  · exact div_le_div_of_nonneg_left (by positivity) (by positivity) hs2

/-- **A support of two or three points with an odd one**: one fresh step from `U_6`, at `|π_S U_6| = 2^64`, with
`α ≤ 2^-128` and `M = 2^32` (`sum_env_le`). -/
theorem term_odd {S : Finset (Fin 4)} (hS : ∃ e ∈ S, e.val % 2 = 1) (h2 : 2 ≤ S.card) (h3 : S.card ≤ 3) :
    ∑ r : Fin 4 → Fin 10 → GF128, min 1 ((Fintype.card GF128 : ℝ) ^ S.card /
      (csz (projS S (spanX (col deployedFix r) (pre 10 7))) : ℝ) ^ ((pre 10 7)ᶜ.card + 1)) ≤
    Fintype.card (Fin 4 → Fin 10 → GF128) * (3 / 2 ^ 128) := by
  have hc : (pre 10 7)ᶜ.card + 1 = 4 := by decide
  rw [hc]
  obtain ⟨ha1, ha2, ha3⟩ := arith_odd h2 h3
  set K0 : ℝ := (2 ^ 128) ^ 3 / 2 ^ 512 + 2 / 2 ^ 32 * (2 ^ 128 / (2 ^ 128) ^ 2) with hK0
  set K1 : ℝ := 2 / 2 ^ 32 / 2 ^ 128 with hK1
  have hA0 : (0 : ℝ) ≤ Fintype.card (Fin 4 → GF128) := Nat.cast_nonneg _
  have hpt : ∀ r : Fin 4 → Fin 10 → GF128, ∑ a : Fin 4 → GF128,
      min 1 ((Fintype.card GF128 : ℝ) ^ S.card /
        (csz (projS S (Ud r 6 ⊔ (Ud r 6).map (AddMonoidHom.mulLeft a))) : ℝ) ^ 4) ≤
      Fintype.card (Fin 4 → GF128) * (K0 + K1 * Xd r) := by
    intro r
    simp only [projS_sup_mul]
    have hv := csz_projS_odd hS r
    have hvR : (csz (projS S (Ud r 6)) : ℝ) = 2 ^ 64 := by rw [hv]; norm_num
    have henv := sum_env_le (projS_le S (Ud r 6)) (m := 1) (fun w _ hw => one_le_supp hw) (p := 4)
      (by norm_num) (M := 2 ^ 32) (by positivity) (by rw [hvR, card_GF]; exact ha2)
    refine henv.trans (mul_le_mul_of_nonneg_left ?_ hA0)
    rw [card_GF]
    have hm : min 1 (((2 : ℝ) ^ 128) ^ S.card / (csz (projS S (Ud r 6)) : ℝ) ^ (2 * 4)) ≤
        (2 ^ 128) ^ 3 / 2 ^ 512 := (min_le_right _ _).trans (by rw [hvR]; exact ha1)
    have hx := excess_le hS r
    rw [card_GF] at hx
    have hx' : Xd r / 2 ^ 128 + 2 ^ 128 / ((2 : ℝ) ^ 128) ^ S.card ≤ Xd r / 2 ^ 128 + 2 ^ 128 / (2 ^ 128) ^ 2 := by
      linarith
    have := mul_le_mul_of_nonneg_left (hx.trans hx') (by positivity : (0 : ℝ) ≤ 2 / 2 ^ 32)
    have e : (2 ^ 128) ^ 3 / 2 ^ 512 + 2 / 2 ^ 32 * (Xd r / 2 ^ 128 + 2 ^ 128 / ((2 : ℝ) ^ 128) ^ 2) =
        K0 + K1 * Xd r := by rw [hK0, hK1]; ring
    linarith
  have hsum := sum_le_of_step (T := 6) (by norm_num) (fun i => deployedFix_late i (by decide))
    (fun W => min 1 ((Fintype.card GF128 : ℝ) ^ S.card / (csz (projS S W) : ℝ) ^ 4)) _ hpt
  refine hsum.trans ((sum_affine_le K0 K1 _ (by positivity) Xd sum_Xd_le').trans ?_)
  gcongr
  have hB := B62_le
  rw [hK0, hK1]
  linarith

theorem arith_02 {v u0 u2 k0 k2 : ℝ} (hv : 0 < v) (hu0 : 0 < u0) (hu2 : 0 < u2) (h0 : u0 ≤ v) (h2 : u2 ≤ v)
    (hk0 : 1 ≤ k0) (hk2 : 1 ≤ k2) (hk0u : k0 * u0 ≤ v) (hk2u : k2 * u2 ≤ v) :
    min 1 (((2 : ℝ) ^ 128) ^ 2 / v ^ (2 * 4)) + 2 / (v ^ 2 / 2 ^ 64) *
        ((k0 - 1) * k0 / (2 ^ 128) ^ 1 + (k2 - 1) * k2 / (2 ^ 128) ^ 1 + v ^ 2 / (2 ^ 128) ^ 2) ≤
      (2 ^ 128) ^ 2 * (1 / (u0 ^ 4 * u2 ^ 4)) + 2 ^ 65 / 2 ^ 128 * (1 / u0 ^ 2 + 1 / u2 ^ 2) +
        2 ^ 65 / (2 ^ 128) ^ 2 := by
  have hm : min 1 (((2 : ℝ) ^ 128) ^ 2 / v ^ (2 * 4)) ≤ (2 ^ 128) ^ 2 * (1 / (u0 ^ 4 * u2 ^ 4)) := by
    refine (min_le_right _ _).trans ?_
    rw [mul_one_div]
    gcongr
    calc u0 ^ 4 * u2 ^ 4 ≤ v ^ 4 * v ^ 4 := by gcongr
      _ = v ^ (2 * 4) := by ring
  have hsq : ∀ {k u : ℝ}, 0 < u → 1 ≤ k → k * u ≤ v → (k - 1) * k ≤ v ^ 2 / u ^ 2 := by
    intro k u hu hk hku
    have hk' : k ≤ v / u := by rw [le_div_iff₀ hu]; exact hku
    calc (k - 1) * k ≤ k ^ 2 := by nlinarith
      _ ≤ (v / u) ^ 2 := by gcongr
      _ = v ^ 2 / u ^ 2 := by rw [div_pow]
  have t0 := hsq hu0 hk0 hk0u
  have t2 := hsq hu2 hk2 hk2u
  have e1 : 2 / (v ^ 2 / 2 ^ 64) * ((k0 - 1) * k0 / (2 ^ 128) ^ 1 + (k2 - 1) * k2 / (2 ^ 128) ^ 1 +
      v ^ 2 / (2 ^ 128) ^ 2) = 2 ^ 65 / 2 ^ 128 * ((k0 - 1) * k0 / v ^ 2 + (k2 - 1) * k2 / v ^ 2) +
        2 ^ 65 / (2 ^ 128) ^ 2 := by
    field_simp
  have b0 : (k0 - 1) * k0 / v ^ 2 ≤ 1 / u0 ^ 2 := by
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    have := (le_div_iff₀ (by positivity : (0 : ℝ) < u0 ^ 2)).1 t0
    linarith
  have b2 : (k2 - 1) * k2 / v ^ 2 ≤ 1 / u2 ^ 2 := by
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    have := (le_div_iff₀ (by positivity : (0 : ℝ) < u2 ^ 2)).1 t2
    linarith
  rw [e1]
  have : 2 ^ 65 / 2 ^ 128 * ((k0 - 1) * k0 / v ^ 2 + (k2 - 1) * k2 / v ^ 2) ≤
      2 ^ 65 / 2 ^ 128 * (1 / u0 ^ 2 + 1 / u2 ^ 2) := by gcongr
  linarith

/-- **The support `{0, 2}`**: one fresh step from `U_6`, with `M = |π_S U_6|²/2^64`; the main term is
`|F|²·E[1/(|π_0 U_6| |π_2 U_6|)^4] ≤ |F|²·Bc(6, 4)²` (`chain`). -/
theorem term_02 :
    ∑ r : Fin 4 → Fin 10 → GF128, min 1 ((Fintype.card GF128 : ℝ) ^ ({0, 2} : Finset (Fin 4)).card /
      (csz (projS {0, 2} (spanX (col deployedFix r) (pre 10 7))) : ℝ) ^ ((pre 10 7)ᶜ.card + 1)) ≤
    Fintype.card (Fin 4 → Fin 10 → GF128) * (3 / 2 ^ 128) := by
  have hc : (pre 10 7)ᶜ.card + 1 = 4 := by decide
  have h02 : (0 : Fin 4) ≠ 2 := by decide
  rw [hc, Finset.card_pair h02]
  have hA0 : (0 : ℝ) ≤ Fintype.card (Fin 4 → GF128) := Nat.cast_nonneg _
  set g : (Fin 4 → Fin 10 → GF128) → ℝ := fun r =>
    (2 ^ 128) ^ 2 * (1 / ((ud 0 r : ℝ) ^ 4 * (ud 2 r : ℝ) ^ 4)) +
      2 ^ 65 / 2 ^ 128 * (1 / (ud 0 r : ℝ) ^ 2 + 1 / (ud 2 r : ℝ) ^ 2) + 2 ^ 65 / (2 ^ 128) ^ 2 with hg
  have hpt : ∀ r : Fin 4 → Fin 10 → GF128, ∑ a : Fin 4 → GF128,
      min 1 ((Fintype.card GF128 : ℝ) ^ 2 /
        (csz (projS {0, 2} (Ud r 6 ⊔ (Ud r 6).map (AddMonoidHom.mulLeft a))) : ℝ) ^ 4) ≤
      Fintype.card (Fin 4 → GF128) * g r := by
    intro r
    simp only [projS_sup_mul]
    set V := projS {0, 2} (Ud r 6) with hV
    have hv : (0 : ℝ) < csz V := by exact_mod_cast csz_pos V
    have hαM : (Fintype.card GF128 : ℝ) ^ ({0, 2} : Finset (Fin 4)).card / (csz V : ℝ) ^ (2 * 4) *
        ((csz V : ℝ) ^ 2 / 2 ^ 64) ^ 4 ≤ 1 := by
      rw [Finset.card_pair h02, card_GF]
      refine le_of_eq ?_
      generalize (csz V : ℝ) = x at hv ⊢
      field_simp
    have henv := sum_env_le (projS_le {0, 2} (Ud r 6)) (m := 1) (fun w _ hw => one_le_supp hw) (p := 4)
      (by norm_num) (M := (csz V : ℝ) ^ 2 / 2 ^ 64) (div_pos (pow_pos hv 2) (by norm_num)) hαM
    rw [Finset.card_pair h02, Finset.sum_pair h02, card_GF] at henv
    rw [card_GF]
    refine henv.trans (mul_le_mul_of_nonneg_left ?_ hA0)
    have hk : ∀ e ∈ ({0, 2} : Finset (Fin 4)), kerE V e * ud e r ≤ csz V := fun e he => by
      have h := kerE_mul_le V e
      rwa [hV, projS_projS (Finset.singleton_subset_iff.2 he)] at h
    have hu : ∀ e ∈ ({0, 2} : Finset (Fin 4)), ud e r ≤ csz V := fun e he =>
      csz_projS_mono (Finset.singleton_subset_iff.2 he) _
    have m0 : (0 : Fin 4) ∈ ({0, 2} : Finset (Fin 4)) := by decide
    have m2 : (2 : Fin 4) ∈ ({0, 2} : Finset (Fin 4)) := by decide
    exact arith_02 hv (by exact_mod_cast csz_pos _) (by exact_mod_cast csz_pos _)
      (by exact_mod_cast hu 0 m0) (by exact_mod_cast hu 2 m2) (by exact_mod_cast one_le_kerE V 0)
      (by exact_mod_cast one_le_kerE V 2) (by exact_mod_cast hk 0 m0) (by exact_mod_cast hk 2 m2)
  have hsum := sum_le_of_step (T := 6) (by norm_num) (fun i => deployedFix_late i (by decide))
    (fun W => min 1 ((Fintype.card GF128 : ℝ) ^ 2 / (csz (projS {0, 2} W) : ℝ) ^ 4)) g hpt
  refine hsum.trans ?_
  have hch := chain deployedFix h02 (deployedFix_even (by decide)) (deployedFix_even (by decide)) 6 (by norm_num) 4 4
  have h0 := sum_inv_ud_le 0 2
  have h2 := sum_inv_ud_le 2 2
  rw [card_GF] at hch h0 h2
  have hB4 := B64_le
  have hB4' : Bc (2 ^ 128 : ℝ) 6 4 * Bc (2 ^ 128) 6 4 ≤ 1 / 2 ^ 200 * (1 / 2 ^ 200) :=
    mul_le_mul hB4 hB4 (Bc_nonneg (by positivity) 6 4) (by positivity)
  have hB2 := B62_le
  have hR0 : (0 : ℝ) ≤ Fintype.card (Fin 4 → Fin 10 → GF128) := Nat.cast_nonneg _
  calc ∑ r, g r = (2 ^ 128) ^ 2 * ∑ r : Fin 4 → Fin 10 → GF128, 1 / ((ud 0 r : ℝ) ^ 4 * (ud 2 r : ℝ) ^ 4) +
        2 ^ 65 / 2 ^ 128 * (∑ r : Fin 4 → Fin 10 → GF128, 1 / (ud 0 r : ℝ) ^ 2 +
          ∑ r : Fin 4 → Fin 10 → GF128, 1 / (ud 2 r : ℝ) ^ 2) +
        Fintype.card (Fin 4 → Fin 10 → GF128) * (2 ^ 65 / (2 ^ 128) ^ 2) := by
        simp only [hg, Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    _ ≤ (2 ^ 128) ^ 2 * (Fintype.card (Fin 4 → Fin 10 → GF128) * (Bc (2 ^ 128) 6 4 * Bc (2 ^ 128) 6 4)) +
        2 ^ 65 / 2 ^ 128 * (Fintype.card (Fin 4 → Fin 10 → GF128) * Bc (2 ^ 128) 6 2 +
          Fintype.card (Fin 4 → Fin 10 → GF128) * Bc (2 ^ 128) 6 2) +
        Fintype.card (Fin 4 → Fin 10 → GF128) * (2 ^ 65 / (2 ^ 128) ^ 2) := by
        gcongr
        exact hch
    _ ≤ _ := by
        rw [← sub_nonneg]
        have key : (2 ^ 128) ^ 2 * (Bc (2 ^ 128 : ℝ) 6 4 * Bc (2 ^ 128) 6 4) +
            2 ^ 65 / 2 ^ 128 * (2 * Bc (2 ^ 128) 6 2) + 2 ^ 65 / (2 ^ 128) ^ 2 ≤ 3 / 2 ^ 128 := by
          nlinarith
        nlinarith

end PerSupport

section Inj

open Classical

variable {ι F : Type} [Fintype ι] [DecidableEq ι] [Field F] [Fintype F] [DecidableEq F]

/-- A map that keeps a subgroup's size is injective on it. -/
theorem eq_zero_of_csz_map {V : AddSubgroup (ι → F)} {f : (ι → F) →+ (ι → F)}
    (h : csz V ≤ csz (V.map f)) {w : ι → F} (hw : w ∈ V) (hf : f w = 0) : w = 0 := by
  unfold csz at h
  have himg : univ.filter (· ∈ V.map f) = (univ.filter (· ∈ V)).image f := by
    ext z
    simp [AddSubgroup.mem_map]
  rw [himg] at h
  have hinj := Finset.card_image_iff.1 (le_antisymm Finset.card_image_le h)
  exact hinj (Finset.mem_filter.2 ⟨mem_univ _, hw⟩) (Finset.mem_filter.2 ⟨mem_univ _, V.zero_mem⟩)
    (hf.trans (map_zero f).symm)

end Inj

section Univ

open Classical

set_option exponentiation.threshold 1024

theorem projS_univ (U : AddSubgroup (Fin 4 → GF128)) : projS univ U = U := by
  unfold projS
  have : (zS (univ : Finset (Fin 4)) : (Fin 4 → GF128) →+ (Fin 4 → GF128)) = AddMonoidHom.id _ :=
    AddMonoidHom.ext fun a => funext fun i => by simp [zS]
  rw [this, AddSubgroup.map_id]

theorem le_coordSub_univ (U : AddSubgroup (Fin 4 → GF128)) : U ≤ coordSub univ := by
  have := projS_le univ U
  rwa [projS_univ] at this

theorem ind_lt_le {x K N : ℕ} (h : x * K = N) (hK : 1 ≤ K) : (if x < N then (1 : ℝ) else 0) ≤ K - 1 := by
  split
  · rename_i hx
    have hK2 : 2 ≤ K := by
      by_contra hc
      have : K = 1 := by omega
      subst this
      omega
    have : (2 : ℝ) ≤ K := by exact_mod_cast hK2
    linarith
  · have : (1 : ℝ) ≤ K := by exact_mod_cast hK
    linarith

theorem compl_odd : ∀ e : Fin 4, ∃ x ∈ ({e}ᶜ : Finset (Fin 4)), x.val % 2 = 1 := by decide

theorem card_compl_single (e : Fin 4) : ({e}ᶜ : Finset (Fin 4)).card = 3 := by
  rw [Finset.card_compl, Finset.card_singleton, Fintype.card_fin]

/-- **A triple's projection of `U_7` falls short of `2^128`** with probability at most `E[X]/|F| + |F|^-2`: from
`|π_T U_6| = 2^64` one step doubles the exponent unless `K(V, a) ≥ 2`, and `E_a[K − 1]` is `sum_Kc_sub_le`'s. -/
theorem bad_triple_le (e : Fin 4) :
    ∑ r : Fin 4 → Fin 10 → GF128, (if csz (projS {e}ᶜ (Ud r 7)) < 2 ^ 128 then (1 : ℝ) else 0) ≤
      Fintype.card (Fin 4 → Fin 10 → GF128) *
        (2 ^ 128 / (2 ^ 128) ^ 3 + 1 / 2 ^ 128 * (4 * (2 ^ 128 * Bc (2 ^ 128) 6 2 - 1))) := by
  have hS := compl_odd e
  have h3 := card_compl_single e
  have hA0 : (0 : ℝ) ≤ Fintype.card (Fin 4 → GF128) := Nat.cast_nonneg _
  have hpt : ∀ r : Fin 4 → Fin 10 → GF128, ∑ a : Fin 4 → GF128,
      (if csz (projS {e}ᶜ (Ud r 6 ⊔ (Ud r 6).map (AddMonoidHom.mulLeft a))) < 2 ^ 128 then (1 : ℝ) else 0) ≤
      Fintype.card (Fin 4 → GF128) * (2 ^ 128 / (2 ^ 128) ^ 3 + 1 / 2 ^ 128 * Xd r) := by
    intro r
    simp only [projS_sup_mul]
    have hv := csz_projS_odd hS r
    have hK := sum_Kc_sub_le (projS_le {e}ᶜ (Ud r 6)) (m := 1) (fun w _ hw => one_le_supp hw)
    have hx := excess_le hS r
    rw [card_GF, h3] at hK hx
    calc _ ≤ ∑ a : Fin 4 → GF128, ((Kc (projS {e}ᶜ (Ud r 6)) a : ℝ) - 1) :=
          Finset.sum_le_sum fun a _ => ind_lt_le (by rw [csz_sup_mul_Kc, hv]; norm_num) (one_le_Kc _ _)
      _ ≤ _ := hK.trans (mul_le_mul_of_nonneg_left (by rw [pow_one] at hx ⊢; linarith) hA0)
  have hsum := sum_le_of_step (T := 6) (by norm_num) (fun i => deployedFix_late i (by decide))
    (fun W => if csz (projS {e}ᶜ W) < 2 ^ 128 then (1 : ℝ) else 0) _ hpt
  exact hsum.trans (sum_affine_le _ _ _ (by positivity) Xd sum_Xd_le')

/-- The coins at which every triple's projection of `U_7` is the whole `2^128`. -/
def Good (r : Fin 4 → Fin 10 → GF128) : Prop := ∀ e : Fin 4, 2 ^ 128 ≤ csz (projS {e}ᶜ (Ud r 7))

theorem csz_U7 {r : Fin 4 → Fin 10 → GF128} (hr : Good r) : csz (Ud r 7) = 2 ^ 128 :=
  le_antisymm ((csz_Ud_le r (by norm_num)).trans (by norm_num)) ((hr 0).trans (csz_projS_le _ _))

/-- On `Good`, no nonzero point of `U_7` has support one: zeroing its point's complement would kill it. -/
theorem good_supp {r : Fin 4 → Fin 10 → GF128} (hr : Good r) {w : Fin 4 → GF128} (hw : w ∈ Ud r 7)
    (hw0 : w ≠ 0) : 2 ≤ (supp w).card := by
  by_contra hlt
  push Not at hlt
  obtain ⟨e, he⟩ := Finset.card_eq_one.1 (show (supp w).card = 1 by have := one_le_supp hw0; omega)
  have hz : zS ({e}ᶜ : Finset (Fin 4)) w = 0 := by
    funext i
    by_cases hi : i ∈ ({e}ᶜ : Finset (Fin 4))
    · have hni : i ∉ supp w := by
        rw [he]
        simpa using hi
      simp only [supp, Finset.mem_filter, Finset.mem_univ, true_and, not_not] at hni
      simp [zS, hi, hni]
    · simp only [Finset.mem_compl, Finset.mem_singleton, not_not] at hi
      simp [zS, hi]
  exact hw0 (eq_zero_of_csz_map ((le_of_eq (csz_U7 hr)).trans (hr e)) hw hz)

theorem good_kerE {r : Fin 4 → Fin 10 → GF128} (hr : Good r) (e : Fin 4) :
    kerE (Ud r 7) e * ud e r ≤ 2 ^ 128 := by
  have h := kerE_mul_le (Ud r 7) e
  rw [csz_U7 hr] at h
  have hu : ud e r ≤ csz (projS {e} (Ud r 7)) :=
    csz_mono (AddSubgroup.map_mono (spanX_mono _ (pre_mono (by norm_num))))
  exact (Nat.mul_le_mul_left _ hu).trans h

theorem arith_ker2 {k u : ℕ} (hk1 : 1 ≤ k) (hu : 0 < u) (h : k * u ≤ 2 ^ 128) :
    ((k : ℝ) - 1) * k / (2 ^ 128) ^ 2 ≤ 1 / (u : ℝ) ^ 2 := by
  have hk1' : (1 : ℝ) ≤ k := by exact_mod_cast hk1
  have hu' : (0 : ℝ) < u := by exact_mod_cast hu
  have hku : (k : ℝ) * u ≤ 2 ^ 128 := by exact_mod_cast h
  have hk : (k : ℝ) ≤ 2 ^ 128 / u := by rw [le_div_iff₀ hu']; exact hku
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  calc ((k : ℝ) - 1) * k * (u : ℝ) ^ 2 ≤ (k * u) ^ 2 := by nlinarith
    _ ≤ (2 ^ 128) ^ 2 := by gcongr
    _ = 1 * (2 ^ 128) ^ 2 := by ring

noncomputable def Yd (r : Fin 4 → Fin 10 → GF128) : ℝ := ∑ e : Fin 4, 1 / (ud e r : ℝ) ^ 2

theorem Yd_nonneg (r : Fin 4 → Fin 10 → GF128) : 0 ≤ Yd r :=
  Finset.sum_nonneg fun e _ => by positivity

theorem sum_Yd_le : ∑ r, Yd r ≤ Fintype.card (Fin 4 → Fin 10 → GF128) * (4 * Bc (2 ^ 128) 6 2) := by
  unfold Yd
  rw [Finset.sum_comm]
  calc _ ≤ ∑ _e : Fin 4, (Fintype.card (Fin 4 → Fin 10 → GF128) : ℝ) * Bc (2 ^ 128) 6 2 :=
        Finset.sum_le_sum fun e _ => by
          have h := sum_inv_ud_le e 2
          rwa [card_GF] at h
    _ = _ := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        push_cast
        ring

/-- On `Good` the step to `U_8` is from `|U_7| = |F|` with supports at least two: `α = |F|^-2`, `M = 2^85`. -/
theorem good_step {r : Fin 4 → Fin 10 → GF128} (hr : Good r) :
    ∑ a : Fin 4 → GF128, min 1 ((Fintype.card GF128 : ℝ) ^ 4 /
        (csz (Ud r 7 ⊔ (Ud r 7).map (AddMonoidHom.mulLeft a)) : ℝ) ^ 3) ≤
      Fintype.card (Fin 4 → GF128) * (1 / 2 ^ 256 + 2 / 2 ^ 85 * (Yd r + 1 / 2 ^ 256)) := by
  have hU : (csz (Ud r 7) : ℝ) = 2 ^ 128 := by rw [csz_U7 hr]; norm_num
  have hA0 : (0 : ℝ) ≤ Fintype.card (Fin 4 → GF128) := Nat.cast_nonneg _
  have hc4 : (univ : Finset (Fin 4)).card = 4 := by rw [Finset.card_univ, Fintype.card_fin]
  have henv := sum_env_le (le_coordSub_univ (Ud r 7)) (m := 2) (fun w hw hw0 => good_supp hr hw hw0) (p := 3)
    (by norm_num) (M := 2 ^ 85) (by positivity) (by rw [hc4, hU, card_GF]; norm_num)
  rw [hc4, hU, card_GF] at henv
  rw [card_GF]
  refine henv.trans (mul_le_mul_of_nonneg_left ?_ hA0)
  have hk : ∑ e ∈ (univ : Finset (Fin 4)), ((kerE (Ud r 7) e : ℝ) - 1) * kerE (Ud r 7) e / ((2 : ℝ) ^ 128) ^ 2 ≤
      Yd r := Finset.sum_le_sum fun e _ =>
    arith_ker2 (one_le_kerE _ _) (csz_pos _) (good_kerE hr e)
  have hm : min 1 (((2 : ℝ) ^ 128) ^ 4 / ((2 : ℝ) ^ 128) ^ (2 * 3)) ≤ 1 / 2 ^ 256 :=
    (min_le_right _ _).trans (by norm_num)
  have e1 : ((2 : ℝ) ^ 128) ^ 2 / ((2 : ℝ) ^ 128) ^ 4 = 1 / 2 ^ 256 := by norm_num
  rw [e1]
  have := mul_le_mul_of_nonneg_left (add_le_add_right hk (1 / 2 ^ 256)) (by positivity : (0 : ℝ) ≤ 2 / 2 ^ 85)
  linarith

theorem bad_ind_le (r : Fin 4 → Fin 10 → GF128) :
    (if Good r then (0 : ℝ) else 1) ≤
      ∑ e : Fin 4, (if csz (projS {e}ᶜ (Ud r 7)) < 2 ^ 128 then (1 : ℝ) else 0) := by
  split
  · exact Finset.sum_nonneg fun e _ => by split <;> norm_num
  · rename_i h
    obtain ⟨e, he⟩ : ∃ e : Fin 4, csz (projS {e}ᶜ (Ud r 7)) < 2 ^ 128 := by
      unfold Good at h
      push Not at h
      exact h
    refine le_trans ?_ (Finset.single_le_sum (f := fun e : Fin 4 =>
      if csz (projS {e}ᶜ (Ud r 7)) < 2 ^ 128 then (1 : ℝ) else 0)
      (fun e _ => by split <;> norm_num) (mem_univ e))
    simp only [he, ↓reduceIte, le_refl]

theorem arith_univ {b : ℝ} (hb : 2 ^ 128 * b ≤ 1 + 1 / 2 ^ 30) :
    (1 / 2 ^ 256 + 2 / 2 ^ 85 * (1 / 2 ^ 256) + 2 / 2 ^ 85 * (4 * b)) +
      4 * (2 ^ 128 / (2 ^ 128) ^ 3 + 1 / 2 ^ 128 * (4 * (2 ^ 128 * b - 1))) ≤ (3 : ℝ) / 2 ^ 128 := by
  norm_num
  linarith

/-- **The full support**: one fresh step from `U_7` (`sum_env_le`); where some triple's projection of `U_7` falls
short of `|F|` (`bad_triple_le`) the step's survival is bounded by one. -/
theorem term_univ :
    ∑ r : Fin 4 → Fin 10 → GF128, min 1 ((Fintype.card GF128 : ℝ) ^ (univ : Finset (Fin 4)).card /
      (csz (projS univ (spanX (col deployedFix r) (pre 10 8))) : ℝ) ^ ((pre 10 8)ᶜ.card + 1)) ≤
    Fintype.card (Fin 4 → Fin 10 → GF128) * (3 / 2 ^ 128) := by
  have hc : (pre 10 8)ᶜ.card + 1 = 3 := by decide
  have hc4 : (univ : Finset (Fin 4)).card = 4 := by rw [Finset.card_univ, Fintype.card_fin]
  rw [hc, hc4]
  simp only [projS_univ]
  have hA0 : (0 : ℝ) ≤ Fintype.card (Fin 4 → GF128) := Nat.cast_nonneg _
  set K0 : ℝ := 1 / 2 ^ 256 + 2 / 2 ^ 85 * (1 / 2 ^ 256) with hK0
  set g : (Fin 4 → Fin 10 → GF128) → ℝ := fun r => (K0 + 2 / 2 ^ 85 * Yd r) + (if Good r then 0 else 1) with hg
  have hpt : ∀ r : Fin 4 → Fin 10 → GF128, ∑ a : Fin 4 → GF128,
      min 1 ((Fintype.card GF128 : ℝ) ^ 4 / (csz (Ud r 7 ⊔ (Ud r 7).map (AddMonoidHom.mulLeft a)) : ℝ) ^ 3) ≤
      Fintype.card (Fin 4 → GF128) * g r := by
    intro r
    have hY := Yd_nonneg r
    by_cases hr : Good r
    · refine (good_step hr).trans (mul_le_mul_of_nonneg_left (le_of_eq ?_) hA0)
      simp only [hg, hK0, hr, ↓reduceIte]
      ring
    · calc _ ≤ ∑ _a : Fin 4 → GF128, (1 : ℝ) := Finset.sum_le_sum fun a _ => min_le_left _ _
        _ = Fintype.card (Fin 4 → GF128) * 1 := by
            rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
        _ ≤ _ := by
            refine mul_le_mul_of_nonneg_left ?_ hA0
            simp only [hg, hr, ↓reduceIte]
            have : 0 ≤ K0 := by rw [hK0]; positivity
            nlinarith
  have hsum := sum_le_of_step (T := 7) (by norm_num) (fun i => deployedFix_late i (by decide))
    (fun W => min 1 ((Fintype.card GF128 : ℝ) ^ 4 / (csz W : ℝ) ^ 3)) g hpt
  refine hsum.trans ?_
  have hY := sum_affine_le K0 (2 / 2 ^ 85) _ (by positivity) Yd sum_Yd_le
  have hbad : ∑ r : Fin 4 → Fin 10 → GF128, (if Good r then (0 : ℝ) else 1) ≤
      Fintype.card (Fin 4 → Fin 10 → GF128) *
        (4 * (2 ^ 128 / (2 ^ 128) ^ 3 + 1 / 2 ^ 128 * (4 * (2 ^ 128 * Bc (2 ^ 128) 6 2 - 1)))) := by
    calc _ ≤ ∑ r : Fin 4 → Fin 10 → GF128, ∑ e : Fin 4,
          (if csz (projS {e}ᶜ (Ud r 7)) < 2 ^ 128 then (1 : ℝ) else 0) := Finset.sum_le_sum fun r _ => bad_ind_le r
      _ = ∑ e : Fin 4, ∑ r : Fin 4 → Fin 10 → GF128,
          (if csz (projS {e}ᶜ (Ud r 7)) < 2 ^ 128 then (1 : ℝ) else 0) := Finset.sum_comm
      _ ≤ ∑ _e : Fin 4, (Fintype.card (Fin 4 → Fin 10 → GF128) : ℝ) *
          (2 ^ 128 / (2 ^ 128) ^ 3 + 1 / 2 ^ 128 * (4 * (2 ^ 128 * Bc (2 ^ 128) 6 2 - 1))) :=
          Finset.sum_le_sum fun e _ => bad_triple_le e
      _ = _ := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
          push_cast
          ring
  have ha := arith_univ B62_le
  have hR0 : (0 : ℝ) ≤ Fintype.card (Fin 4 → Fin 10 → GF128) := Nat.cast_nonneg _
  have hsplit : ∑ r, g r = ∑ r, (K0 + 2 / 2 ^ 85 * Yd r) + ∑ r : Fin 4 → Fin 10 → GF128,
      (if Good r then (0 : ℝ) else 1) := Finset.sum_add_distrib
  rw [hsplit]
  generalize (Fintype.card (Fin 4 → Fin 10 → GF128) : ℝ) = R at hY hbad hR0 ⊢
  generalize Bc (2 ^ 128 : ℝ) 6 2 = b at hY hbad ha
  have := mul_le_mul_of_nonneg_left ha hR0
  rw [hK0] at hY
  nlinarith

end Univ

section Assembly

open Classical

/-- The coordinates each support's term keeps: six at a singleton, eight at the full support, seven otherwise. -/
def TS (S : Finset (Fin 4)) : ℕ := if S.card = 1 then 6 else if S.card = 4 then 8 else 7

theorem six_le_TS (S : Finset (Fin 4)) : 6 ≤ TS S := by
  unfold TS
  split_ifs <;> omega

theorem no_odd_eq : ∀ S : Finset (Fin 4), 2 ≤ S.card → S.card ≤ 3 → (¬ ∃ e ∈ S, e.val % 2 = 1) → S = {0, 2} := by
  decide

theorem card_nonempty : (univ.filter fun S : Finset (Fin 4) => S.Nonempty).card = 15 := by decide

theorem term_le (S : Finset (Fin 4)) (hS : S.Nonempty) :
    ∑ r : Fin 4 → Fin 10 → GF128, min 1 ((Fintype.card GF128 : ℝ) ^ S.card /
      (csz (projS S (spanX (col deployedFix r) (pre 10 (TS S)))) : ℝ) ^ ((pre 10 (TS S))ᶜ.card + 1)) ≤
    Fintype.card (Fin 4 → Fin 10 → GF128) * (3 / 2 ^ 128) := by
  have h1 := hS.card_pos
  have h4 : S.card ≤ 4 := (Finset.card_le_univ S).trans (by rw [Fintype.card_fin])
  by_cases hc1 : S.card = 1
  · obtain ⟨e, rfl⟩ := Finset.card_eq_one.1 hc1
    have : TS {e} = 6 := by simp [TS]
    rw [this]
    exact term_single e
  by_cases hc4 : S.card = 4
  · have hU : S = univ := Finset.eq_univ_of_card S (by rw [hc4, Fintype.card_fin])
    subst hU
    have : TS univ = 8 := by simp [TS, hc4]
    rw [this]
    exact term_univ
  have hT : TS S = 7 := by simp [TS, hc1, hc4]
  rw [hT]
  by_cases hodd : ∃ e ∈ S, e.val % 2 = 1
  · exact term_odd hodd (by omega) (by omega)
  · rw [no_odd_eq S (by omega) (by omega) hodd]
    exact term_02

/-- **The deployed rank bound**: the coins at which the four points' mask weights are rank-deficient are at most
`15·3·2^-128 ≤ 2^-122` of all (`card_deficient_le`, each support's term at most `3·2^-128`). -/
theorem deployedRankBound : DeployedRankBound := by
  have hz : ((Pi.evalAddMonoidHom (fun _ : Fin 128 => ZMod 2) 0).comp unpackHom)
      (GF128.of (Flock.F128.monomial ((0 : Fin 128) : ℕ))) = 1 := by
    show unpack (GF128.of (Flock.F128.monomial ((0 : Fin 128) : ℕ))) 0 = 1
    rw [unpack_monomial]
    rfl
  have hfree : ∀ S, ∀ t ∉ pre 10 (TS S), ∀ i, deployedFix i t = none := fun S t ht i => by
    simp only [pre, Finset.mem_filter, Finset.mem_univ, true_and, not_lt] at ht
    exact deployedFix_late i ((six_le_TS S).trans ht)
  have hd := card_deficient_le _ hz deployedFix (fun S => pre 10 (TS S)) hfree
  have hsum : ∑ S ∈ univ.filter (fun S : Finset (Fin 4) => S.Nonempty), ∑ r : Fin 4 → Fin 10 → GF128,
      min 1 ((Fintype.card GF128 : ℝ) ^ S.card /
        (csz (projS S (spanX (col deployedFix r) (pre 10 (TS S)))) : ℝ) ^ ((pre 10 (TS S))ᶜ.card + 1)) ≤
      ∑ _S ∈ univ.filter (fun S : Finset (Fin 4) => S.Nonempty),
        (Fintype.card (Fin 4 → Fin 10 → GF128) : ℝ) * (3 / 2 ^ 128) :=
    Finset.sum_le_sum fun S hS => term_le S (Finset.mem_filter.1 hS).2
  rw [Finset.sum_const, card_nonempty, nsmul_eq_mul] at hsum
  have hfilt : (univ.filter fun r : Fin 4 → Fin 10 → GF128 =>
      ¬ RankFull fun i t => (deployedFix i t).getD (r i t)) =
      univ.filter fun r : Fin 4 → Fin 10 → GF128 => span (col deployedFix r) ≠ ⊤ :=
    Finset.filter_congr fun r _ => Iff.rfl
  unfold DeployedRankBound RankBound
  rw [hfilt]
  have hreal : ((univ.filter fun r : Fin 4 → Fin 10 → GF128 => span (col deployedFix r) ≠ ⊤).card : ℝ) * 2 ^ 122 ≤
      Fintype.card (Fin 4 → Fin 10 → GF128) := by
    have h := hd.trans hsum
    have hR : (0 : ℝ) ≤ Fintype.card (Fin 4 → Fin 10 → GF128) := Nat.cast_nonneg _
    generalize ((univ.filter fun r : Fin 4 → Fin 10 → GF128 => span (col deployedFix r) ≠ ⊤).card : ℝ) = n at h ⊢
    generalize (Fintype.card (Fin 4 → Fin 10 → GF128) : ℝ) = R at h hR ⊢
    norm_num at h ⊢
    linarith
  exact_mod_cast hreal

/-- **The rank bound at the verifier's coins** from the small slots' alone. -/
theorem maskRankBound_of_small (hS : SmallSlotBound) : MaskRankBound := maskRankBound_of deployedRankBound hS

/-- **The rank bound at the verifier's coins holds**: the deployed bound, and no slot the verifier admits is under
`2^13` bits (`smallSlotBound`). -/
theorem maskRankBound : MaskRankBound := maskRankBound_of_small smallSlotBound

end Assembly

end FlockComplete.MaskRank
