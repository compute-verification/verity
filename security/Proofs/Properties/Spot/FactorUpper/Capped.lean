module

public import Proofs.Properties.Spot.FactorUpper.Patterns

/-!
# Rankin's capped factor bounded above

`FactorUpperCappedSound`. A set's capped weight `2^{min(b(U), S)}` is at most the product over the keys (a stratum and
a width) of `2^{min(b_k(U), S)}`, as `min(a + b, S) ≤ min(a, S) + min(b, S)` (`capped_term_le`), and a sum over the sets
of a product over the keys is the product over the keys of their sums over counts, `Σ_j C(n, j) f(j)` (`binS`,
`sum_prod_cnt`). Each key's sum is at most the lesser of `(1 + 2^b w)^n` and its sets of at most `⌊S/b⌋` proof blocks
plus `2^S` times the rest (`binS_served`), the rest at most `(1 + w)^n − 1` and, at `z = n w < ⌊S/b⌋ + 2`, its closed
form (`tailS_le_closed`); `servedFac` bounds the lesser (`servedFac_bounds`). The whole sum over the sets avoiding the
cells sampled whole is also at most `2^S (∏ (1 + w)^n − 1)` (`served_sound`). The sets meeting one are bounded both
ways, by `∏_F G (∏_I G − 1)` (`whole_capped_le`) and by `2^S 2^{n_F} (2^{n_I} − 1)` (`whole_count_le`).
-/

@[expose] public section

namespace Properties.Spot.SecurityProofs.FactorUpperProof

open _root_.Spot.Interval
open _root_.Spot.FactorUpper
open Claims.Sampling
open Nat

/-- `Σ_{j≤n} C(n, j) f(j)`: a key's sum over its sets of `n` proof blocks, `f(j)` for each set of `j`. -/
noncomputable def binS (n : ℕ) (f : ℕ → ℝ) : ℝ := ∑ j ∈ Finset.range (n + 1), (n.choose j : ℝ) * f j

theorem binS_zero (f : ℕ → ℝ) : binS 0 f = f 0 := by simp [binS]

theorem binS_succ (n : ℕ) (f : ℕ → ℝ) : binS (n + 1) f = binS n f + binS n (fun j => f (j + 1)) := by
  unfold binS
  rw [Finset.sum_range_succ' _ (n + 1), Finset.sum_range_succ' (fun j => (n.choose j : ℝ) * f j)]
  simp only [Nat.choose_succ_succ', Nat.cast_add, add_mul, Finset.sum_add_distrib]
  rw [Finset.sum_range_succ (fun j => (n.choose (j + 1) : ℝ) * f (j + 1)) n]
  simp [Nat.choose_succ_self]
  ring

theorem binS_mono {n n' : ℕ} (h : n ≤ n') {f : ℕ → ℝ} (hf : ∀ j, 0 ≤ f j) : binS n f ≤ binS n' f := by
  unfold binS
  calc ∑ j ∈ Finset.range (n + 1), (n.choose j : ℝ) * f j
      ≤ ∑ j ∈ Finset.range (n + 1), (n'.choose j : ℝ) * f j :=
        Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_right (by exact_mod_cast Nat.choose_le_choose j h) (hf j)
    _ ≤ _ := Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.2 (by omega))
        fun j _ _ => mul_nonneg (Nat.cast_nonneg _) (hf j)

theorem binS_pow (n : ℕ) (x : ℝ) : binS n (fun j => x ^ j) = (1 + x) ^ n := by
  rw [add_comm, add_pow]
  unfold binS
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [one_pow, mul_one, mul_comm]

theorem one_le_binS (n : ℕ) {f : ℕ → ℝ} (hf : ∀ j, 0 ≤ f j) (h0 : 1 ≤ f 0) : 1 ≤ binS n f := by
  unfold binS
  have := Finset.single_le_sum (f := fun j => (n.choose j : ℝ) * f j) (fun j _ => mul_nonneg (Nat.cast_nonneg _) (hf j))
    (Finset.mem_range.2 (Nat.succ_pos n))
  simp only [Nat.choose_zero_right, Nat.cast_one, one_mul] at this
  linarith

/-- **Sums over sets factor over the keys**: `Σ_{A ⊆ D} ∏_k g_k(n_k(A)) = ∏_k Σ_{j≤n_k} C(n_k, j) g_k(j)`, `n_k(A)` the
proof blocks of `A` of key `k`. -/
theorem sum_prod_cnt {G : Type*} [DecidableEq G] (K : Finset G) (κ : ℕ → G) (D : Finset ℕ)
    (hD : ∀ v ∈ D, κ v ∈ K) (g : G → ℕ → ℝ) :
    ∑ A ∈ D.powerset, ∏ k ∈ K, g k (cntOf κ A k) = ∏ k ∈ K, binS (cntOf κ D k) (g k) := by
  induction D using Finset.induction_on generalizing g with
  | empty => simp [cntOf, binS_zero]
  | @insert v D hv ih =>
    have hD' : ∀ w ∈ D, κ w ∈ K := fun w hw => hD w (Finset.mem_insert_of_mem hw)
    have hk0 : κ v ∈ K := hD v (Finset.mem_insert_self _ _)
    set g' : G → ℕ → ℝ := Function.update g (κ v) (fun j => g (κ v) (j + 1)) with hg'
    rw [Finset.sum_powerset_insert hv]
    have hins : ∀ A ∈ D.powerset, ∏ k ∈ K, g k (cntOf κ (insert v A) k) = ∏ k ∈ K, g' k (cntOf κ A k) := by
      intro A hA
      have hvA : v ∉ A := fun h => hv (Finset.mem_powerset.1 hA h)
      refine Finset.prod_congr rfl fun k _ => ?_
      by_cases hkv : k = κ v
      · subst hkv; rw [cntOf_insert_self hvA, hg', Function.update_self]
      · rw [cntOf_insert_ne (Ne.symm hkv), hg', Function.update_of_ne hkv]
    rw [Finset.sum_congr rfl hins, ih hD' g, ih hD' g', ← Finset.mul_prod_erase K _ hk0,
      ← Finset.mul_prod_erase K _ hk0, ← Finset.mul_prod_erase K _ hk0]
    have hE' : ∏ k ∈ K.erase (κ v), binS (cntOf κ D k) (g' k) = ∏ k ∈ K.erase (κ v), binS (cntOf κ D k) (g k) :=
      Finset.prod_congr rfl fun k hk => by rw [hg', Function.update_of_ne (Finset.ne_of_mem_erase hk)]
    have hE : ∏ k ∈ K.erase (κ v), binS (cntOf κ (insert v D) k) (g k) =
        ∏ k ∈ K.erase (κ v), binS (cntOf κ D k) (g k) :=
      Finset.prod_congr rfl fun k hk => by rw [cntOf_insert_ne (Ne.symm (Finset.ne_of_mem_erase hk))]
    rw [hE', hE, cntOf_insert_self hv, binS_succ, hg', Function.update_self]
    ring

/-- `∏_{v∈U} f(κ v) = ∏_k f(k)^{n_k(U)}`. -/
theorem prod_cnt {G : Type*} [DecidableEq G] (K : Finset G) (κ : ℕ → G) (U : Finset ℕ) (hU : ∀ v ∈ U, κ v ∈ K)
    (f : G → ℝ) : ∏ v ∈ U, f (κ v) = ∏ k ∈ K, f k ^ cntOf κ U k := by
  rw [Finset.prod_comp]
  refine Finset.prod_subset (fun k hk => ?_) fun k _ hk => ?_
  · obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hk; exact hU v hv
  · have : cntOf κ U k = 0 := by
      unfold cntOf
      rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
      intro v hv hvk; exact hk (Finset.mem_image.2 ⟨v, hv, hvk⟩)
    unfold cntOf at this
    rw [this, pow_zero]

/-- `Σ_{v∈U} β(κ v) = Σ_k n_k(U) β(k)`. -/
theorem sum_cnt {G : Type*} [DecidableEq G] (K : Finset G) (κ : ℕ → G) (U : Finset ℕ) (hU : ∀ v ∈ U, κ v ∈ K)
    (β : G → ℕ) : ∑ v ∈ U, β (κ v) = ∑ k ∈ K, cntOf κ U k * β k := by
  rw [Finset.sum_comp]
  refine Finset.sum_subset (fun k hk => ?_) fun k _ hk => ?_
  · obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hk; exact hU v hv
  · have : cntOf κ U k = 0 := by
      unfold cntOf
      rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
      intro v hv hvk; exact hk (Finset.mem_image.2 ⟨v, hv, hvk⟩)
    unfold cntOf at this
    rw [this, zero_smul]

theorem min_sum_le {G : Type*} (K : Finset G) (a : G → ℕ) (S : ℕ) :
    min (∑ k ∈ K, a k) S ≤ ∑ k ∈ K, min (a k) S := by
  classical
  induction K using Finset.induction_on with
  | empty => simp
  | @insert k K hk ih => rw [Finset.sum_insert hk, Finset.sum_insert hk]; omega

/-- **A set's capped term is below its keys'**: `2^{min(b(U), S)} ∏_{v∈U} w_v ≤ ∏_k 2^{min(n_k b_k, S)} w_k^{n_k}`, as
`min(a + b, S) ≤ min(a, S) + min(b, S)`. -/
theorem capped_term_le {G : Type*} [DecidableEq G] (K : Finset G) (κ : ℕ → G) (U : Finset ℕ)
    (hU : ∀ v ∈ U, κ v ∈ K) (β : G → ℕ) (w : G → ℝ) (hw : ∀ k, 0 ≤ w k) (S : ℕ) :
    (2 : ℝ) ^ min (∑ v ∈ U, β (κ v)) S * ∏ v ∈ U, w (κ v) ≤
      ∏ k ∈ K, ((2 : ℝ) ^ min (cntOf κ U k * β k) S * w k ^ cntOf κ U k) := by
  rw [Finset.prod_mul_distrib, prod_cnt K κ U hU w, sum_cnt K κ U hU β, Finset.prod_pow_eq_pow_sum]
  exact mul_le_mul_of_nonneg_right (pow_le_pow_right₀ (by norm_num) (min_sum_le K _ S))
    (Finset.prod_nonneg fun k _ => pow_nonneg (hw k) _)

/-- `Σ_{m<j≤n} C(n, j) w^j`: a key's sets of more than `m` of its `n` proof blocks, at weight `w` each. -/
noncomputable def tailS (m n : ℕ) (w : ℝ) : ℝ :=
  ∑ j ∈ (Finset.range (n + 1)).filter (m < ·), (n.choose j : ℝ) * w ^ j

theorem tailS_nonneg (m n : ℕ) {w : ℝ} (hw : 0 ≤ w) : 0 ≤ tailS m n w :=
  Finset.sum_nonneg fun _ _ => by positivity

theorem tailS_le_grow (m n : ℕ) {w : ℝ} (hw : 0 ≤ w) : tailS m n w ≤ (1 + w) ^ n - 1 := by
  have h := binS_pow n w
  unfold binS at h
  have h0 : (0 : ℕ) ∈ Finset.range (n + 1) := Finset.mem_range.2 (Nat.succ_pos n)
  rw [← Finset.add_sum_erase _ _ h0] at h
  simp only [Nat.choose_zero_right, Nat.cast_one, pow_zero, mul_one] at h
  rw [← h, add_sub_cancel_left]
  unfold tailS
  refine Finset.sum_le_sum_of_subset_of_nonneg (fun j hj => ?_) fun j _ _ => by positivity
  rw [Finset.mem_filter] at hj
  exact Finset.mem_erase.2 ⟨by omega, hj.1⟩

/-- **The tail's closed form**: at `z = n w < m + 2`, `Σ_{m<j≤n} C(n, j) w^j ≤ z^{m+1}/(m+1)! · 1/(1 − z/(m+2))`, as
`C(n, j) ≤ n^j/j!` and `j! ≥ (m+1)! (m+2)^{j−m−1}`. -/
theorem tailS_le_closed (m n : ℕ) {w : ℝ} (hw : 0 ≤ w) (hz : (n : ℝ) * w < m + 2) :
    tailS m n w ≤ ((n : ℝ) * w) ^ (m + 1) / ((m + 1)! : ℝ) / (1 - (n : ℝ) * w / (m + 2)) := by
  set z := (n : ℝ) * w with hzdef
  have hz0 : 0 ≤ z := mul_nonneg (Nat.cast_nonneg _) hw
  have hm2 : (0 : ℝ) < m + 2 := by positivity
  set r := z / (m + 2) with hr
  have hr0 : 0 ≤ r := div_nonneg hz0 hm2.le
  have hr1 : r < 1 := (div_lt_one hm2).2 hz
  set A := z ^ (m + 1) / ((m + 1)! : ℝ) with hA
  have hA0 : 0 ≤ A := by positivity
  have hfilt : (Finset.range (n + 1)).filter (m < ·) = Finset.Ico (m + 1) (n + 1) := by
    ext j; simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_Ico]; omega
  have hterm : ∀ i, (n.choose (m + 1 + i) : ℝ) * w ^ (m + 1 + i) ≤ A * r ^ i := by
    intro i
    have hc := Nat.choose_le_pow_div (α := ℝ) (m + 1 + i) n
    have hf : ((m + 1)! : ℝ) * ((m : ℝ) + 2) ^ i ≤ ((m + 1 + i)! : ℝ) := by
      have h := @Nat.factorial_mul_pow_le_factorial (m + 1) i
      have e : ((m : ℝ) + 2) = ((m + 1 + 1 : ℕ) : ℝ) := by push_cast; ring
      rw [e]; exact_mod_cast h
    calc (n.choose (m + 1 + i) : ℝ) * w ^ (m + 1 + i)
        ≤ (n : ℝ) ^ (m + 1 + i) / ((m + 1 + i)! : ℝ) * w ^ (m + 1 + i) :=
          mul_le_mul_of_nonneg_right hc (pow_nonneg hw _)
      _ = z ^ (m + 1 + i) / ((m + 1 + i)! : ℝ) := by rw [hzdef, mul_pow]; ring
      _ ≤ z ^ (m + 1 + i) / (((m + 1)! : ℝ) * ((m : ℝ) + 2) ^ i) :=
          div_le_div_of_nonneg_left (pow_nonneg hz0 _) (by positivity) hf
      _ = A * r ^ i := by rw [hA, hr, div_pow, pow_add]; field_simp
  unfold tailS
  rw [hfilt, Finset.sum_Ico_eq_sum_range]
  calc ∑ i ∈ Finset.range (n + 1 - (m + 1)), (n.choose (m + 1 + i) : ℝ) * w ^ (m + 1 + i)
      ≤ ∑ i ∈ Finset.range (n + 1 - (m + 1)), A * r ^ i := Finset.sum_le_sum fun i _ => hterm i
    _ = A * ∑ i ∈ Finset.range (n + 1 - (m + 1)), r ^ i := by rw [Finset.mul_sum]
    _ ≤ A * (1 / (1 - r)) :=
        mul_le_mul_of_nonneg_left ((geom_le_min _ hr0 hr1).trans (min_le_right _ _)) hA0
    _ = A / (1 - r) := by ring

/-- A key's served value: `(1 + 2^b w)^n`, and its sets of at most `⌊S/b⌋` proof blocks plus `2^S` times the rest. -/
noncomputable def servVal (b n S : ℕ) (w : ℝ) : ℝ :=
  min ((1 + 2 ^ b * w) ^ n) (capP (S / b) n (2 ^ b * w) + 2 ^ S * tailS (S / b) n w)

theorem one_le_servVal (b n S : ℕ) {w : ℝ} (hw : 0 ≤ w) : 1 ≤ servVal b n S w := by
  unfold servVal
  have hb : 0 ≤ (2 : ℝ) ^ b * w := mul_nonneg (pow_nonneg (zero_le_two (α := ℝ)) b) hw
  have h1 : 1 ≤ (1 + 2 ^ b * w) ^ n := one_le_pow₀ (by linarith)
  have h2 := one_le_capP (S / b) n hb
  have h3 : 0 ≤ (2 : ℝ) ^ S * tailS (S / b) n w := mul_nonneg (by positivity) (tailS_nonneg _ _ hw)
  exact le_min h1 (by linarith)

/-- **A key's served sum**: `Σ_{j≤n} C(n, j) 2^{min(j b, S)} w^j` is at most its served value. -/
theorem binS_served (b n S : ℕ) {w : ℝ} (hw : 0 ≤ w) :
    binS n (fun j => (2 : ℝ) ^ min (j * b) S * w ^ j) ≤ servVal b n S w := by
  have hx : 0 ≤ (2 : ℝ) ^ b * w := mul_nonneg (by positivity) hw
  refine le_min ?_ ?_
  · rw [← binS_pow]
    unfold binS
    refine Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left ?_ (Nat.cast_nonneg _)
    show (2 : ℝ) ^ min (j * b) S * w ^ j ≤ (2 ^ b * w) ^ j
    rw [mul_pow, ← pow_mul, mul_comm b j]
    exact mul_le_mul_of_nonneg_right (pow_le_pow_right₀ (by norm_num) (min_le_left _ _)) (pow_nonneg hw _)
  · set m := S / b
    have hterm : ∀ j ∈ Finset.range (n + 1), (n.choose j : ℝ) * ((2 : ℝ) ^ min (j * b) S * w ^ j) ≤
        (if j ≤ m then (n.choose j : ℝ) * (2 ^ b * w) ^ j else 0) +
          2 ^ S * (if m < j then (n.choose j : ℝ) * w ^ j else 0) := by
      intro j _
      by_cases hjm : j ≤ m
      · rw [if_pos hjm, if_neg (by omega), mul_zero, add_zero]
        refine mul_le_mul_of_nonneg_left ?_ (Nat.cast_nonneg _)
        rw [mul_pow, ← pow_mul, mul_comm b j]
        exact mul_le_mul_of_nonneg_right (pow_le_pow_right₀ (by norm_num) (min_le_left _ _)) (pow_nonneg hw _)
      · rw [if_neg hjm, if_pos (by omega), zero_add]
        calc (n.choose j : ℝ) * ((2 : ℝ) ^ min (j * b) S * w ^ j)
            ≤ (n.choose j : ℝ) * ((2 : ℝ) ^ S * w ^ j) :=
              mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right
                (pow_le_pow_right₀ (by norm_num) (min_le_right _ _)) (pow_nonneg hw _)) (Nat.cast_nonneg _)
          _ = _ := by ring
    unfold binS
    refine (Finset.sum_le_sum hterm).trans ?_
    rw [Finset.sum_add_distrib, ← Finset.mul_sum]
    refine add_le_add ?_ (le_of_eq ?_)
    · rw [← Finset.sum_filter]
      unfold capP
      refine Finset.sum_le_sum_of_subset_of_nonneg (fun j hj => ?_) fun j _ _ => by positivity
      rw [Finset.mem_filter] at hj
      exact Finset.mem_range.2 (by omega)
    · rw [← Finset.sum_filter]; rfl

theorem FacBounds.mono {F : Fac} {A B : ℝ} (hA : 0 ≤ A) (hAB : A ≤ B) (h : FacBounds F B) : FacBounds F A :=
  ⟨hA, (Real.log_le_log (by linarith) (by linarith)).trans h.2.1, h.2.2.mono hA hAB⟩

/-- **`heavyTail` bounds a key's tail**: the grow form always, and the closed form once `z = n w < m + 2`. -/
theorem heavyTail_bounds {g : Group} {S : ℕ} (hg : groupOk g S) (lw : ℚ) (p : ℕ) :
    LogUB (heavyTail g lw S p) (tailS (S / g.b) g.n (Real.exp lw)) := by
  have hw := Real.exp_pos (lw : ℝ)
  have hO := pow_bounds (n := g.n) (lnN := g.lnN) p (one_bounds (x := Real.exp lw) (lx := lw) p hw.le le_rfl)
    fun _ => hg.2.1
  have hgrow : LogUB ((Fac.one lw p).pow g.n g.lnN p).l (tailS (S / g.b) g.n (Real.exp lw)) :=
    hO.2.2.mono (tailS_nonneg _ _ hw.le) (tailS_le_grow _ _ hw.le)
  unfold heavyTail
  dsimp only
  split_ifs with he
  swap
  · exact hgrow
  refine LogUB.omin hgrow ?_
  set m := S / g.b with hm
  set e : ℚ := expUp (g.lnN + lw - lnLower ((m + 2 : ℕ) : ℚ) p) p with he_def
  have hn : (0 : ℝ) < g.n := by exact_mod_cast hg.1
  have hz0 : 0 < (g.n : ℝ) * Real.exp lw := mul_pos hn hw
  have hlnz : Real.log ((g.n : ℝ) * Real.exp lw) ≤ (g.lnN : ℝ) + lw := by
    rw [Real.log_mul hn.ne' hw.ne', Real.log_exp]; linarith [hg.2.1]
  have hm2 : (0 : ℝ) < m + 2 := by positivity
  have hze : (g.n : ℝ) * Real.exp lw / (m + 2) ≤ (e : ℝ) := by
    have h1 := exp_le_expUp (g.lnN + lw - lnLower ((m + 2 : ℕ) : ℚ) p) p
    have h2 := lnLowerSound ((m + 2 : ℕ) : ℚ) p
    calc (g.n : ℝ) * Real.exp lw / (m + 2)
        = Real.exp (Real.log ((g.n : ℝ) * Real.exp lw) - Real.log ((m : ℝ) + 2)) := by
          rw [Real.exp_sub, Real.exp_log hz0, Real.exp_log hm2]
      _ ≤ Real.exp (((g.lnN + lw - lnLower ((m + 2 : ℕ) : ℚ) p : ℚ) : ℝ)) := by
          apply Real.exp_le_exp.2; push_cast at h2 ⊢; linarith
      _ ≤ (e : ℝ) := h1
  have he1 : (e : ℝ) < 1 := by exact_mod_cast he
  have hz : (g.n : ℝ) * Real.exp lw < m + 2 := by
    have h : (g.n : ℝ) * Real.exp lw / (m + 2) < 1 := lt_of_le_of_lt hze he1
    rwa [div_lt_one hm2] at h
  refine (tailS_le_closed m g.n hw.le hz).trans ?_
  have hd : 0 < 1 - (g.n : ℝ) * Real.exp lw / (m + 2) := by linarith
  have hfac : (g.lf : ℝ) + (lnLower ((m + 1 : ℕ) : ℚ) p : ℝ) ≤ Real.log ((m + 1)! : ℝ) := by
    rw [Nat.factorial_succ, Nat.cast_mul, Real.log_mul (by positivity) (by positivity)]
    have := lnLowerSound ((m + 1 : ℕ) : ℚ) p
    have hlf : (g.lf : ℝ) ≤ Real.log ((m)! : ℝ) := hg.2.2.2.2
    push_cast at this ⊢
    linarith
  have h1e := lnLowerSound (1 - e) p
  have hle : Real.log (1 - (e : ℝ)) ≤ Real.log (1 - (g.n : ℝ) * Real.exp lw / (m + 2)) :=
    Real.log_le_log (by linarith) (by linarith)
  have hpow : ((m : ℝ) + 1) * Real.log ((g.n : ℝ) * Real.exp lw) ≤ ((m : ℝ) + 1) * ((g.lnN : ℝ) + lw) :=
    mul_le_mul_of_nonneg_left hlnz (by positivity)
  rw [← Real.exp_log (show 0 < ((g.n : ℝ) * Real.exp lw) ^ (m + 1) / ((m + 1)! : ℝ) /
    (1 - (g.n : ℝ) * Real.exp lw / (m + 2)) by positivity)]
  apply Real.exp_le_exp.2
  rw [Real.log_div (by positivity) hd.ne', Real.log_div (by positivity) (by positivity), Real.log_pow]
  push_cast at h1e hfac ⊢
  linarith

theorem servedFac_bounds {L : Ln2} (hL : ln2Ok L) {g : Group} {S : ℕ} (hg : groupOk g S) (lw : ℚ) (p : ℕ) :
    FacBounds (servedFac L g lw S p) (servVal g.b g.n S (Real.exp lw) - 1) := by
  have hW := wholeFac_bounds hL hg lw p
  have hw := Real.exp_pos (lw : ℝ)
  have h1 := one_le_servVal g.b g.n S hw.le
  have hle : servVal g.b g.n S (Real.exp lw) - 1 ≤ (1 + 2 ^ g.b * Real.exp lw) ^ g.n - 1 :=
    sub_le_sub_right (min_le_left _ _) 1
  unfold servedFac
  by_cases hc : g.b = 0 ∨ g.n ≤ S / g.b
  · rw [if_pos hc]; exact hW.mono (by linarith) hle
  · rw [if_neg hc]
    simp only [not_or, not_le] at hc
    obtain ⟨hb, -⟩ := hc
    have hC := cappedFac_bounds hL hg lw p
    have hcap : capVal g S (2 ^ g.b * Real.exp lw) = capP (S / g.b) g.n (2 ^ g.b * Real.exp lw) := if_neg hb
    rw [hcap] at hC
    have hT0 := tailS_nonneg (S / g.b) g.n hw.le
    have hS := LogUB.scale (heavyTail_bounds hg lw p) hT0 (k := (2 : ℝ) ^ S) (c := (S : ℚ) * L.up)
      (by have := two_pow_le_exp hL S; push_cast at this ⊢; exact this)
    have hsum := lse_two hC.2.2 hS hC.1 (mul_nonneg (by positivity) hT0) p
    have hmin := min_le_right ((1 + 2 ^ g.b * Real.exp (lw : ℝ)) ^ g.n)
      (capP (S / g.b) g.n (2 ^ g.b * Real.exp lw) + 2 ^ S * tailS (S / g.b) g.n (Real.exp lw))
    have hlog : LogUB (omin (wholeFac L g lw p).l (lse ([(cappedFac L g lw S p).l,
        (heavyTail g lw S p).map ((S : ℚ) * L.up + ·)].filterMap id) p))
        (servVal g.b g.n S (Real.exp lw) - 1) :=
      LogUB.omin (hW.2.2.mono (by linarith) hle) (hsum.mono (by linarith) (by unfold servVal; linarith [hmin]))
    refine ⟨by linarith, ?_, hlog⟩
    show Real.log (1 + (servVal g.b g.n S (Real.exp lw) - 1)) ≤ ((min (wholeFac L g lw p).f (lnOnePlus _ p) : ℚ) : ℝ)
    push_cast
    exact le_min ((Real.log_le_log (by linarith) (by linarith)).trans hW.2.1)
      (log_one_add_le_lnOnePlus (by linarith) hlog p)

section Capped

variable {s : AuditStatement} {Pt : Partition} {d : Design.Two} {ρ : ℕ} {cs : List Cell}

/-- A set's capped term at weights `w` by key, as a product over its proof blocks. -/
theorem capped_exp (S : ℕ) (c : ℝ) (U : Finset ℕ) :
    (2 : ℝ) ^ min (bitsOf s Pt U) S * Real.exp (-(c * yOf Pt cs U)) =
      (2 : ℝ) ^ min (∑ v ∈ U, (keyOf s Pt v).2) S *
        ∏ v ∈ U, (fun k : ℕ × ℕ => Real.exp (-(c * (rateOf (cs.getD k.1 default) : ℝ)))) (keyOf s Pt v) := by
  unfold yOf
  rw [Finset.mul_sum, ← Finset.sum_neg_distrib, Real.exp_sum]
  rfl

theorem fin_keys_mem (hc : Properties.Spot.Covers s Pt d ρ cs) :
    ∀ v ∈ finBlocks Pt d ρ cs, keyOf s Pt v ∈ (finKeys cs (cs.map Cell.parts)).toFinset :=
  fun v hv => List.mem_toFinset.2 ((fin_keys hc).1 v hv)

/-- **The served sets' sum is above a chord's capped sum**: at slope `σ`, `Σ_{∅ ≠ U} 2^{min(b(U), S)} e^{−θ σ y(U)}`
over the sets of the cells sampled at a rate. -/
theorem served_sound (hc : Properties.Spot.Covers s Pt d ρ cs) (S : ℕ) (θ σ : ℚ) (p : ℕ) :
    LogUB (servedSets (Ln2.of p) (((finKeys cs (cs.map Cell.parts)).map (grp cs (cs.map Cell.parts) S p)).map
        fun g => (g, -(θ * σ * g.r))) S p)
      (∑ U ∈ (finBlocks Pt d ρ cs).powerset.erase ∅,
        (2 : ℝ) ^ min (bitsOf s Pt U) S * Real.exp (-((θ : ℝ) * σ * yOf Pt cs U))) := by
  set reads := cs.map Cell.parts
  set Kf := finKeys cs reads
  set G := grp cs reads S p
  set Bf := finBlocks Pt d ρ cs
  set L := Ln2.of p
  have hL : ln2Ok L := ln2Ok_of p
  set w : ℕ × ℕ → ℝ := fun k => Real.exp (-((θ : ℝ) * σ * (rateOf (cs.getD k.1 default) : ℝ))) with hwdef
  have hw : ∀ k, 0 ≤ w k := fun k => (Real.exp_pos _).le
  have hK := fin_keys_mem hc
  set X : Finset ℕ → ℝ := fun U => (2 : ℝ) ^ min (bitsOf s Pt U) S * Real.exp (-((θ : ℝ) * σ * yOf Pt cs U))
  have hX0 : X ∅ = 1 := by simp [X, bitsOf, yOf]
  have hXn : ∀ U, 0 ≤ X U := fun U => by positivity
  have herase : ∑ U ∈ Bf.powerset.erase ∅, X U = ∑ U ∈ Bf.powerset, X U - 1 := by
    rw [Finset.sum_erase_eq_sub (Finset.empty_mem_powerset Bf), hX0]
  have h0 : 0 ≤ ∑ U ∈ Bf.powerset.erase ∅, X U := Finset.sum_nonneg fun U _ => hXn U
  set gs := (Kf.map G).map fun g => (g, -(θ * σ * g.r))
  have hgs : ∀ x ∈ gs, groupOk x.1 S := by
    intro x hx
    simp only [gs, List.map_map, List.mem_map, Function.comp] at hx
    obtain ⟨k, hk, rfl⟩ := hx
    exact grp_ok (List.mem_filter.1 hk).1
  have hwk : ∀ k, Real.exp (((-(θ * σ * (G k).r) : ℚ) : ℝ)) = w k := by
    intro k
    simp only [hwdef, G, grp, Group.r, rateOf]
    push_cast
    ring_nf
  -- the product of the groups' served sums
  have hF := prod_map_bounds' gs (fun x => servedFac L x.1 x.2 S p) (fun x => servVal x.1.b x.1.n S (Real.exp x.2))
    (fun x hx => servedFac_bounds hL (hgs x hx) x.2 p) p
  have hFeq : (gs.map fun x => servVal x.1.b x.1.n S (Real.exp x.2)).prod =
      (Kf.map fun k => servVal k.2 (G k).n S (w k)).prod := by
    simp only [gs, List.map_map, Function.comp_def]
    congr 1
    refine List.map_congr_left fun k _ => ?_
    rw [hwk]
    rfl
  have hall : ∑ U ∈ Bf.powerset, X U ≤ (Kf.map fun k => servVal k.2 (G k).n S (w k)).prod := by
    calc ∑ U ∈ Bf.powerset, X U
        ≤ ∑ U ∈ Bf.powerset, ∏ k ∈ Kf.toFinset,
            ((2 : ℝ) ^ min (cntOf (keyOf s Pt) U k * k.2) S * w k ^ cntOf (keyOf s Pt) U k) :=
          Finset.sum_le_sum fun U hU => by
            simp only [X]
            rw [capped_exp S ((θ : ℝ) * σ) U]
            exact capped_term_le Kf.toFinset (keyOf s Pt) U (fun v hv => hK v (Finset.mem_powerset.1 hU hv))
              (fun k => k.2) w hw S
      _ = ∏ k ∈ Kf.toFinset, binS (cntOf (keyOf s Pt) Bf k) (fun j => (2 : ℝ) ^ min (j * k.2) S * w k ^ j) :=
          sum_prod_cnt Kf.toFinset (keyOf s Pt) Bf hK (fun k j => (2 : ℝ) ^ min (j * k.2) S * w k ^ j)
      _ ≤ ∏ k ∈ Kf.toFinset, servVal k.2 (G k).n S (w k) :=
          Finset.prod_le_prod₀ (fun k _ => le_trans zero_le_one
              (one_le_binS _ (fun j => by have := hw k; positivity) (by simp))) fun k _ =>
            (binS_mono ((fin_keys hc).2 k) fun j => by have := hw k; positivity).trans (binS_served k.2 _ S (hw k))
      _ = _ := List.prod_toFinset _ (finKeys_nodup cs reads)
  -- the sets' sum at `2^S`
  have hO := prod_map_bounds' gs (fun x => (Fac.one x.2 p).pow x.1.n x.1.lnN p) (fun x => (1 + Real.exp x.2) ^ x.1.n)
    (fun x hx => pow_bounds p (one_bounds p (Real.exp_pos _).le le_rfl) fun _ => (hgs x hx).2.1) p
  have hS := LogUB.scale hO.2.2 hO.1 (k := (2 : ℝ) ^ S) (c := (S : ℚ) * L.up)
    (by have := two_pow_le_exp hL S; push_cast at this ⊢; exact this)
  have hOeq : (gs.map fun x => (1 + Real.exp x.2) ^ x.1.n).prod = (Kf.map fun k => (1 + w k) ^ (G k).n).prod := by
    simp only [gs, List.map_map, Function.comp_def]
    congr 1
    refine List.map_congr_left fun k _ => ?_
    rw [hwk]
  have hgrow : ∑ U ∈ Bf.powerset.erase ∅, X U ≤ (2 : ℝ) ^ S * ((Kf.map fun k => (1 + w k) ^ (G k).n).prod - 1) := by
    have hle : ∀ U ∈ Bf.powerset, X U ≤ (2 : ℝ) ^ S * ∏ v ∈ U, w (keyOf s Pt v) := by
      intro U _
      simp only [X]
      rw [capped_exp S ((θ : ℝ) * σ) U]
      exact mul_le_mul_of_nonneg_right (pow_le_pow_right₀ (by norm_num) (min_le_right _ _))
        (Finset.prod_nonneg fun v _ => hw _)
    have hone : ∑ U ∈ Bf.powerset, ∏ v ∈ U, w (keyOf s Pt v) = ∏ v ∈ Bf, (1 + w (keyOf s Pt v)) :=
      (Finset.prod_one_add _).symm
    have hkeys := prod_le_keys Kf (finKeys_nodup cs reads) Bf (keyOf s Pt) (fin_keys hc).1 (fun k => (G k).n)
      (fun k _ => (fin_keys hc).2 k) (fun k => 1 + w k) (fun k => by linarith [hw k])
    rw [herase]
    calc ∑ U ∈ Bf.powerset, X U - 1 ≤ ∑ U ∈ Bf.powerset, X U - X ∅ := by rw [hX0]
      _ ≤ (2 : ℝ) ^ S * ∑ U ∈ Bf.powerset, ∏ v ∈ U, w (keyOf s Pt v) - (2 : ℝ) ^ S * 1 := by
          rw [← Finset.add_sum_erase _ _ (Finset.empty_mem_powerset Bf),
            ← Finset.add_sum_erase _ _ (Finset.empty_mem_powerset Bf), Finset.prod_empty, mul_add, mul_one]
          have := Finset.sum_le_sum fun U (hU : U ∈ Bf.powerset.erase ∅) => hle U (Finset.mem_of_mem_erase hU)
          rw [← Finset.mul_sum] at this
          linarith
      _ ≤ _ := by rw [hone, ← mul_sub]; exact mul_le_mul_of_nonneg_left (by linarith) (by positivity)
  refine LogUB.omin (hF.2.2.mono h0 ?_) (hS.mono h0 ?_)
  · rw [hFeq]; linarith
  · rw [hOeq]; exact hgrow

/-- The cells sampled at a rate hold all of the replay block's proof blocks of their keys, and none of the others'. -/
theorem cnt_finBlocks (k : ℕ × ℕ) :
    cntOf (keyOf s Pt) (finBlocks Pt d ρ cs) k =
      if (cs.getD k.1 default).rate.isSome then cntOf (keyOf s Pt) (blocksIn Pt d.part ρ) k else 0 := by
  unfold cntOf finBlocks
  rw [Finset.filter_filter]
  split_ifs with h
  · congr 1
    refine Finset.filter_congr fun v _ => ⟨fun h' => h'.2, ?_⟩
    rintro rfl
    exact ⟨h, rfl⟩
  · rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
    rintro v - ⟨hv, rfl⟩
    exact h hv

theorem powerset_filter_sub (B Bf : Finset ℕ) (h : Bf ⊆ B) : B.powerset.filter (· ⊆ Bf) = Bf.powerset := by
  ext U; simp only [Finset.mem_filter, Finset.mem_powerset]
  exact ⟨fun h' => h'.2, fun h' => ⟨h'.trans h, h'⟩⟩

theorem keys_disjoint (cs : List Cell) :
    Disjoint (finKeys cs (cs.map Cell.parts)).toFinset (infKeys cs (cs.map Cell.parts)).toFinset := by
  refine Finset.disjoint_left.2 fun k hf hi => ?_
  have h1 := (List.mem_filter.1 (List.mem_toFinset.1 hf)).2
  have h2 := (List.mem_filter.1 (List.mem_toFinset.1 hi)).2
  cases h : (cs.getD k.1 default).rate <;> simp_all

/-- **The capped sets meeting a cell sampled whole, by key**: `Σ_{U ⊄ F} 2^{min(b(U), S)} ≤ ∏_F G_k (∏_I G_k − 1)`, the
keys' served values at `w = 1`. -/
theorem whole_capped_le (hc : Properties.Spot.Covers s Pt d ρ cs) (S : ℕ) :
    ∑ U ∈ (blocksIn Pt d.part ρ).powerset.filter (fun U => ¬ U ⊆ finBlocks Pt d ρ cs), (2 : ℝ) ^ min (bitsOf s Pt U) S ≤
      ((finKeys cs (cs.map Cell.parts)).map fun k =>
          servVal k.2 (countIn ((cs.map Cell.parts).getD k.1 []) k.2) S 1).prod *
        (((infKeys cs (cs.map Cell.parts)).map fun k =>
          servVal k.2 (countIn ((cs.map Cell.parts).getD k.1 []) k.2) S 1).prod - 1) := by
  set reads := cs.map Cell.parts
  set B := blocksIn Pt d.part ρ
  set Bf := finBlocks Pt d ρ cs
  set Kf := (finKeys cs reads).toFinset
  set Ki := (infKeys cs reads).toFinset
  set K := Kf ∪ Ki
  set n : ℕ × ℕ → ℕ := fun k => countIn (reads.getD k.1 []) k.2
  set g : ℕ × ℕ → ℕ → ℝ := fun k j => (2 : ℝ) ^ min (j * k.2) S * (1 : ℝ) ^ j
  have hg : ∀ k j, 0 ≤ g k j := fun k j => by positivity
  have hg0 : ∀ k, g k 0 = 1 := fun k => by simp [g]
  have hkeys := covers_keys hc (D := B) le_rfl
  have hKB : ∀ v ∈ B, keyOf s Pt v ∈ K := by
    intro v hv
    have hk := hkeys.1 v hv
    cases h : (cs.getD (keyOf s Pt v).1 default).rate with
    | none => exact Finset.mem_union_right _ (List.mem_toFinset.2 (key_mem_inf hk h))
    | some r => exact Finset.mem_union_left _ (List.mem_toFinset.2 (key_mem_fin hk (by rw [h]; rfl)))
  have hBf : Bf ⊆ B := Finset.filter_subset _ _
  set PK : Finset ℕ → ℝ := fun U => ∏ k ∈ K, g k (cntOf (keyOf s Pt) U k)
  have hT : ∀ U ∈ B.powerset.filter (fun U => ¬ U ⊆ Bf), (2 : ℝ) ^ min (bitsOf s Pt U) S ≤ PK U := by
    intro U hU
    have hUB := Finset.mem_powerset.1 (Finset.mem_filter.1 hU).1
    have := capped_term_le K (keyOf s Pt) U (fun v hv => hKB v (hUB hv)) (fun k => k.2) (fun _ => 1)
      (fun _ => zero_le_one) S
    rw [Finset.prod_const_one, mul_one] at this
    exact this
  have hsplit : ∑ U ∈ B.powerset.filter (fun U => ¬ U ⊆ Bf), PK U =
      ∑ U ∈ B.powerset, PK U - ∑ U ∈ Bf.powerset, PK U := by
    rw [← Finset.sum_filter_add_sum_filter_not B.powerset (· ⊆ Bf), powerset_filter_sub B Bf hBf]
    ring
  have hall : ∑ U ∈ B.powerset, PK U = (∏ k ∈ Kf, binS (cntOf (keyOf s Pt) B k) (g k)) *
      ∏ k ∈ Ki, binS (cntOf (keyOf s Pt) B k) (g k) := by
    rw [sum_prod_cnt K (keyOf s Pt) B hKB g, Finset.prod_union (keys_disjoint cs)]
  have hfin : ∑ U ∈ Bf.powerset, PK U = ∏ k ∈ Kf, binS (cntOf (keyOf s Pt) B k) (g k) := by
    rw [sum_prod_cnt K (keyOf s Pt) Bf (fun v hv => hKB v (hBf hv)) g, Finset.prod_union (keys_disjoint cs)]
    have h1 : ∏ k ∈ Kf, binS (cntOf (keyOf s Pt) Bf k) (g k) = ∏ k ∈ Kf, binS (cntOf (keyOf s Pt) B k) (g k) :=
      Finset.prod_congr rfl fun k hk => by
        rw [cnt_finBlocks, if_pos (List.mem_filter.1 (List.mem_toFinset.1 hk)).2]
    have h2 : ∏ k ∈ Ki, binS (cntOf (keyOf s Pt) Bf k) (g k) = 1 :=
      Finset.prod_eq_one fun k hk => by
        have hr := (List.mem_filter.1 (List.mem_toFinset.1 hk)).2
        rw [cnt_finBlocks, if_neg (by cases h : (cs.getD k.1 default).rate <;> simp_all), binS_zero, hg0]
    rw [h1, h2, mul_one]
  have hle : ∀ k, binS (cntOf (keyOf s Pt) B k) (g k) ≤ servVal k.2 (n k) S 1 := fun k =>
    (binS_mono (hkeys.2 k) (hg k)).trans (binS_served k.2 (n k) S zero_le_one)
  have h1 : ∀ k, 1 ≤ binS (cntOf (keyOf s Pt) B k) (g k) := fun k => one_le_binS _ (hg k) (by rw [hg0])
  have hPf : ∏ k ∈ Kf, binS (cntOf (keyOf s Pt) B k) (g k) ≤ ∏ k ∈ Kf, servVal k.2 (n k) S 1 :=
    Finset.prod_le_prod₀ (fun k _ => le_trans zero_le_one (h1 k)) fun k _ => hle k
  have hPi : ∏ k ∈ Ki, binS (cntOf (keyOf s Pt) B k) (g k) ≤ ∏ k ∈ Ki, servVal k.2 (n k) S 1 :=
    Finset.prod_le_prod₀ (fun k _ => le_trans zero_le_one (h1 k)) fun k _ => hle k
  have hPi1 : 1 ≤ ∏ k ∈ Ki, binS (cntOf (keyOf s Pt) B k) (g k) := Finset.one_le_prod₀ fun k _ => h1 k
  have hPf0 : 0 ≤ ∏ k ∈ Kf, binS (cntOf (keyOf s Pt) B k) (g k) :=
    Finset.prod_nonneg fun k _ => le_trans zero_le_one (h1 k)
  rw [← List.prod_toFinset _ (finKeys_nodup cs reads), ← List.prod_toFinset _ (infKeys_nodup cs reads)]
  calc _ ≤ ∑ U ∈ B.powerset.filter (fun U => ¬ U ⊆ Bf), PK U := Finset.sum_le_sum hT
    _ = (∏ k ∈ Kf, binS (cntOf (keyOf s Pt) B k) (g k)) * (∏ k ∈ Ki, binS (cntOf (keyOf s Pt) B k) (g k) - 1) := by
        rw [hsplit, hall, hfin]; ring
    _ ≤ _ := mul_le_mul hPf (by linarith) (by linarith) (le_trans hPf0 hPf)

/-- **The capped sets meeting a cell sampled whole, counted**:
`Σ_{U ⊄ F} 2^{min(b(U), S)} ≤ 2^S 2^{|F|} (2^{|I|} − 1)`. -/
theorem whole_count_le (S : ℕ) :
    ∑ U ∈ (blocksIn Pt d.part ρ).powerset.filter (fun U => ¬ U ⊆ finBlocks Pt d ρ cs), (2 : ℝ) ^ min (bitsOf s Pt U) S ≤
      (2 : ℝ) ^ S * 2 ^ (finBlocks Pt d ρ cs).card * (2 ^ (infBlocks Pt d ρ cs).card - 1) := by
  set B := blocksIn Pt d.part ρ
  set Bf := finBlocks Pt d ρ cs
  set Bi := infBlocks Pt d ρ cs
  have hBf : Bf ⊆ B := Finset.filter_subset _ _
  have hcard : (B.powerset.filter (fun U => ¬ U ⊆ Bf)).card + Bf.powerset.card = B.powerset.card := by
    rw [← powerset_filter_sub B Bf hBf, add_comm]
    exact Finset.card_filter_add_card_filter_not _
  rw [Finset.card_powerset, Finset.card_powerset] at hcard
  have hB : B.card = Bf.card + Bi.card := (Finset.card_filter_add_card_filter_not _).symm
  rw [hB, pow_add] at hcard
  have hcR : ((B.powerset.filter (fun U => ¬ U ⊆ Bf)).card : ℝ) = 2 ^ Bf.card * (2 ^ Bi.card - 1) := by
    have : ((B.powerset.filter (fun U => ¬ U ⊆ Bf)).card : ℝ) + 2 ^ Bf.card = 2 ^ Bf.card * 2 ^ Bi.card := by
      exact_mod_cast hcard
    linarith
  calc _ ≤ ∑ U ∈ B.powerset.filter (fun U => ¬ U ⊆ Bf), (2 : ℝ) ^ S :=
        Finset.sum_le_sum fun U _ => pow_le_pow_right₀ (by norm_num) (min_le_right _ _)
    _ = _ := by rw [Finset.sum_const, nsmul_eq_mul, hcR]; ring

/-- **The sets meeting a cell sampled whole, capped**: `(1 − q)^θ` times the lesser of `∏_F G_g (∏_I G_g − 1)` and
`2^S 2^{n_F} (2^{n_I} − 1)`. -/
theorem wholeCapped_sound (hc : Properties.Spot.Covers s Pt d ρ cs) {q θ : ℚ} (hq1 : (q : ℝ) ≤ 1)
    (hθ : (0 : ℝ) ≤ θ) (S p : ℕ) :
    LogUB (if ((infKeys cs (cs.map Cell.parts)).map (grp cs (cs.map Cell.parts) S p)).isEmpty then none
      else oadd (lnPassWhole q θ p) (omin
        ((Fac.prod (((infKeys cs (cs.map Cell.parts)).map (grp cs (cs.map Cell.parts) S p)).map
            (servedFac (Ln2.of p) · 0 S p)) p).l.map
          ((Fac.prod (((finKeys cs (cs.map Cell.parts)).map (grp cs (cs.map Cell.parts) S p)).map
            (servedFac (Ln2.of p) · 0 S p)) p).f + ·))
        ((Fac.prod (((infKeys cs (cs.map Cell.parts)).map (grp cs (cs.map Cell.parts) S p)).map
            fun g => (Fac.one 0 p).pow g.n g.lnN p) p).l.map
          ((S : ℚ) * (Ln2.of p).up + (((finKeys cs (cs.map Cell.parts)).map (grp cs (cs.map Cell.parts) S p)).map
            fun g => (g.n : ℚ)).sum * (Ln2.of p).up + ·))))
      ((1 - (q : ℝ)) ^ (θ : ℝ) * ∑ U ∈ (blocksIn Pt d.part ρ).powerset.filter (fun U => ¬ U ⊆ finBlocks Pt d ρ cs),
        (2 : ℝ) ^ min (bitsOf s Pt U) S) := by
  set reads := cs.map Cell.parts
  set Kf := finKeys cs reads
  set Ki := infKeys cs reads
  set G := grp cs reads S p
  set L := Ln2.of p
  set B := blocksIn Pt d.part ρ
  set Bf := finBlocks Pt d ρ cs
  set Bi := infBlocks Pt d ρ cs
  set X := ∑ U ∈ B.powerset.filter (fun U => ¬ U ⊆ Bf), (2 : ℝ) ^ min (bitsOf s Pt U) S
  have hL : ln2Ok L := ln2Ok_of p
  have hpq : (0 : ℝ) ≤ (1 - (q : ℝ)) ^ (θ : ℝ) := Real.rpow_nonneg (by linarith) _
  have hX0 : 0 ≤ X := Finset.sum_nonneg fun _ _ => by positivity
  by_cases hKe : Ki = []
  · have hBie : Bi = ∅ := Finset.eq_empty_of_forall_notMem fun w hw => by
      have := (inf_keys hc).1 w hw
      rw [show infKeys cs (cs.map Cell.parts) = Ki from rfl, hKe] at this
      simp at this
    have : (Ki.map G).isEmpty = true := by rw [hKe]; rfl
    rw [if_pos this]
    show _ = (0 : ℝ)
    have hXz : X = 0 := by
      refine Finset.sum_eq_zero fun U hU => absurd ?_ (Finset.mem_filter.1 hU).2
      intro v hv
      have hvB := Finset.mem_powerset.1 (Finset.mem_filter.1 hU).1 hv
      by_contra hvf
      have : v ∈ Bi := Finset.mem_filter.2 ⟨hvB, fun h => hvf (Finset.mem_filter.2 ⟨hvB, h⟩)⟩
      rw [hBie] at this
      exact Finset.notMem_empty v this
    rw [hXz, mul_zero]
  have hne : (Ki.map G).isEmpty = false := by
    cases hK : Ki with
    | nil => exact absurd hK hKe
    | cons _ _ => rfl
  rw [if_neg (by rw [hne]; simp)]
  refine LogUB.oadd (lnPassWhole_sound hq1 hθ p) (LogUB.omin ?_ ?_) hX0
  · -- the keys' served values at `w = 1`
    have hgk : ∀ l : List (ℕ × ℕ), (∀ k ∈ l, k ∈ keysOf cs reads) →
        ∀ g ∈ l.map G, FacBounds (servedFac L g 0 S p) (servVal g.b g.n S (Real.exp ((0 : ℚ) : ℝ)) - 1) := by
      intro l hl g hg
      obtain ⟨k, hk, rfl⟩ := List.mem_map.1 hg
      exact servedFac_bounds hL (grp_ok (hl k hk)) 0 p
    have hPf := prod_map_bounds' (Kf.map G) (servedFac L · 0 S p)
      (fun g => servVal g.b g.n S (Real.exp ((0 : ℚ) : ℝ))) (hgk Kf fun k hk => (List.mem_filter.1 hk).1) p
    have hPi := prod_map_bounds' (Ki.map G) (servedFac L · 0 S p)
      (fun g => servVal g.b g.n S (Real.exp ((0 : ℚ) : ℝ))) (hgk Ki fun k hk => (List.mem_filter.1 hk).1) p
    have heq : ∀ l : List (ℕ × ℕ), ((l.map G).map fun g => servVal g.b g.n S (Real.exp ((0 : ℚ) : ℝ))).prod =
        (l.map fun k => servVal k.2 (countIn (reads.getD k.1 []) k.2) S 1).prod := by
      intro l
      simp only [List.map_map, Function.comp_def, Rat.cast_zero, Real.exp_zero]
      rfl
    rw [heq] at hPf hPi
    have hPfe : (Kf.map fun k => servVal k.2 (countIn (reads.getD k.1 []) k.2) S 1).prod ≤
        Real.exp (Fac.prod ((Kf.map G).map (servedFac L · 0 S p)) p).f := by
      have := le_exp_of_bounds hPf; linarith
    have hb := LogUB.scale hPi.2.2 hPi.1 hPfe
    exact hb.mono hX0 (whole_capped_le hc S)
  · -- counted, at `2^S` each
    have hQ := prod_map_bounds' (Ki.map G) (fun g => (Fac.one 0 p).pow g.n g.lnN p)
      (fun g => (1 + Real.exp ((0 : ℚ) : ℝ)) ^ g.n)
      (fun g hg => by
        obtain ⟨k, hk, rfl⟩ := List.mem_map.1 hg
        exact pow_bounds p (one_bounds p (Real.exp_pos _).le le_rfl) fun _ =>
          (grp_ok (List.mem_filter.1 hk).1).2.1) p
    have hQeq : ((Ki.map G).map fun g => (1 + Real.exp ((0 : ℚ) : ℝ)) ^ g.n).prod =
        (Ki.map fun k => (2 : ℝ) ^ (G k).n).prod := by
      simp only [List.map_map, Function.comp_def, Rat.cast_zero, Real.exp_zero]
      norm_num
    rw [hQeq] at hQ
    have hup : (0 : ℝ) ≤ L.up := le_trans (Real.log_pos one_lt_two).le hL.2
    have hBfn : ((Bf.card : ℕ) : ℝ) ≤ ((((Kf.map G).map fun g => (g.n : ℚ)).sum : ℚ) : ℝ) := by
      have := sum_le_keys Kf (finKeys_nodup cs reads) Bf (keyOf s Pt) (fin_keys hc).1 (fun k => (G k).n)
        (fun k _ => (fin_keys hc).2 k) (fun _ => (1 : ℝ)) (fun _ => zero_le_one)
      rw [Finset.sum_const, nsmul_eq_mul, mul_one] at this
      refine this.trans (le_of_eq ?_)
      rw [Rat.cast_list_sum]
      simp only [List.map_map, Function.comp_def, mul_one, Rat.cast_natCast]
    have hk : (2 : ℝ) ^ S * 2 ^ Bf.card ≤
        Real.exp (((S : ℚ) * L.up + (((Kf.map G).map fun g => (g.n : ℚ)).sum) * L.up : ℚ) : ℝ) := by
      rw [← pow_add]
      refine (two_pow_le_exp hL (S + Bf.card)).trans (Real.exp_le_exp.2 ?_)
      push_cast
      push_cast at hBfn
      nlinarith
    have hb := LogUB.scale hQ.2.2 hQ.1 hk
    refine hb.mono hX0 ((whole_count_le S).trans ?_)
    refine mul_le_mul_of_nonneg_left (sub_le_sub_right ?_ 1) (by positivity)
    have := prod_le_keys Ki (infKeys_nodup cs reads) Bi (keyOf s Pt) (inf_keys hc).1 (fun k => (G k).n)
      (fun k _ => (inf_keys hc).2 k) (fun _ => (2 : ℝ)) (fun _ => one_le_two)
    rw [Finset.prod_const] at this
    exact this

/-- **Rankin's capped factor, split**: the empty set, the nonempty sets in cells sampled at a rate, and the sets meeting
a cell sampled whole, each passing with probability `1 − q`. -/
theorem rankinFactorCapped_le (hc : Properties.Spot.Covers s Pt d ρ cs) (S : ℕ) (q θ : ℝ) :
    rankinFactorCapped s Pt d S q θ ρ ≤
      1 + ∑ U ∈ (finBlocks Pt d ρ cs).powerset.erase ∅,
          (2 : ℝ) ^ min (bitsOf s Pt U) S * d.passes Pt q (counts Pt d.part U) ρ ^ θ +
        (1 - q) ^ θ * ∑ U ∈ (blocksIn Pt d.part ρ).powerset.filter (fun U => ¬ U ⊆ finBlocks Pt d ρ cs),
          (2 : ℝ) ^ min (bitsOf s Pt U) S := by
  set B := blocksIn Pt d.part ρ
  set Bf := finBlocks Pt d ρ cs
  set T : Finset ℕ → ℝ := fun U => (2 : ℝ) ^ min (bitsOf s Pt U) S * d.passes Pt q (counts Pt d.part U) ρ ^ θ
  have hBf : Bf ⊆ B := Finset.filter_subset _ _
  have hT0 : T ∅ = 1 := by simp [T, bitsOf, passes_empty]
  have hrw : ∀ U ∈ B.powerset.filter (fun U => ¬ U ⊆ Bf), T U = (1 - q) ^ θ * (2 : ℝ) ^ min (bitsOf s Pt U) S := by
    intro U hU
    obtain ⟨hUB, hUf⟩ := Finset.mem_filter.1 hU
    rw [Finset.mem_powerset] at hUB
    obtain ⟨w, hwU, hwf⟩ := Finset.not_subset.1 hUf
    have hr : (cs.getD (Pt.stratumOf w) default).rate = none := by
      by_contra hne
      exact hwf (Finset.mem_filter.2 ⟨hUB hwU, Option.isSome_iff_ne_none.2 hne⟩)
    simp only [T]
    rw [passes_whole hc q hUB hwU hr, mul_comm]
  calc rankinFactorCapped s Pt d S q θ ρ = ∑ U ∈ B.powerset, T U := rfl
    _ = (T ∅ + ∑ U ∈ Bf.powerset.erase ∅, T U) + ∑ U ∈ B.powerset.filter (fun U => ¬ U ⊆ Bf), T U := by
      rw [← Finset.sum_filter_add_sum_filter_not B.powerset (· ⊆ Bf), powerset_filter_sub B Bf hBf,
        Finset.add_sum_erase _ _ (Finset.empty_mem_powerset Bf)]
    _ = _ := by rw [hT0, Finset.sum_congr rfl hrw, ← Finset.mul_sum]
    _ ≤ _ := le_rfl

end Capped

/-- **The capped bound is above Rankin's capped factor.** -/
theorem factorUpperCapped_sound : Properties.Spot.FactorUpperCappedSound := by
  intro cs S P v hv s Pt d ρ hc
  by_cases hP : P.valid = true
  swap
  · simp [lnFactorUpperCapped, hP] at hv
  obtain ⟨hq0, hq1, hθ, -⟩ := Params.valid_parts hP
  unfold lnFactorUpperCapped at hv
  rw [if_pos hP] at hv
  simp only [groupsOf, split_eq] at hv
  refine (rankinFactorCapped_le hc S _ _).trans ?_
  refine bound_sound hP hv ((finBlocks Pt d ρ cs).powerset.erase ∅) (fun U => (2 : ℝ) ^ min (bitsOf s Pt U) S)
    (yOf Pt cs) (fun U => d.passes Pt (P.q : ℝ) (counts Pt d.part U) ρ ^ (P.θ : ℝ)) (fun _ _ => by positivity)
    (fun U _ => yOf_nonneg U) (fun U hU => ?_) (fun σ => served_sound hc S P.θ σ P.p)
    (wholeCapped_sound hc hq1 hθ S P.p)
  have hUB : U ⊆ blocksIn Pt d.part ρ :=
    (Finset.mem_powerset.1 (Finset.mem_of_mem_erase hU)).trans (Finset.filter_subset _ _)
  exact Real.rpow_le_rpow (passes_nonneg hq1 _) (passes_le_phi hc hq0 hUB) hθ

end Properties.Spot.SecurityProofs.FactorUpperProof
