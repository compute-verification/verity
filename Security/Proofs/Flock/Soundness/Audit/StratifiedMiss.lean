import Proofs.Flock.Soundness.Audit.Stratified

/-!
# The stratified law's count curve

`miss K` is the largest escape of a set of at least `K` wrong units (`Law.miss`). Under the stratified law a set's escape
depends only on its count `c_s` in each stratum (`stratified_escape`). Stratum `s`'s escape falls by the factor
`r_s(c) = (n_s − c − k_s)/(n_s − c)` when its `(c + 1)`-th unit is wrong, and that factor falls as `c` grows. So the
likeliest set to escape fills the strata greedily, taking the largest factors first, as
`verity.proofs.profile.Stratified.miss` computes it.

* `stratified_miss_eq_greedy`: a count vector `a` that some `τ ∈ (0, 1]` separates attains `miss (∑ a)`, which is then
  `∏ s C(n_s − a_s, k_s)/C(n_s, k_s)`. Every factor `a` takes must be at least `τ`, and every factor it leaves at most
  `τ`. A greedy fill has such a `τ`, the smallest factor it took, so `audit_exfiltration`'s `hk : miss (K + 1) < δ`
  holds for a stratified law once the greedy fill at `K + 1` escapes with less than `δ`.
* `stratum_exchange`, the step: within one stratum, `C(n_s − c, k_s) · τ^a ≤ C(n_s − a, k_s) · τ^c` for every
  `c ≤ n_s`.
-/

namespace FlockSoundness.Audit

open Finset
open scoped ENNReal

namespace Law

variable {n m : ℕ}

/-- Stratum `s`'s escape with `c` of its units wrong: `C(n_s − c, k_s) / C(n_s, k_s)`. -/
noncomputable def stratumEscape (σ : Fin n → Fin m) (k : Fin m → ℕ) (s : Fin m) (c : ℕ) : ℝ≥0∞ :=
  (((stratum σ s).card - c).choose (k s) : ℝ≥0∞) / ((stratum σ s).card.choose (k s) : ℝ≥0∞)

/-- The factor by which stratum `s`'s escape falls when its `(c + 1)`-th unit is wrong: `(n_s − c − k_s)/(n_s − c)`. -/
noncomputable def stratumRatio (σ : Fin n → Fin m) (k : Fin m → ℕ) (s : Fin m) (c : ℕ) : ℝ≥0∞ :=
  (((stratum σ s).card - c - k s : ℕ) : ℝ≥0∞) / (((stratum σ s).card - c : ℕ) : ℝ≥0∞)

/-- The factors fall, cross-multiplied: `(N − j − k)(N − i) ≤ (N − i − k)(N − j)` for `i ≤ j ≤ N`. -/
theorem count_ratio_anti {N k i j : ℕ} (hij : i ≤ j) (hj : j ≤ N) :
    (N - j - k) * (N - i) ≤ (N - i - k) * (N - j) := by
  obtain ⟨x, hx⟩ : ∃ x, x = N - j := ⟨_, rfl⟩
  obtain ⟨y, hy⟩ : ∃ y, y = N - i := ⟨_, rfl⟩
  have hxy : x ≤ y := by omega
  rw [← hx, ← hy]
  rcases le_or_gt x k with h | h
  · rw [Nat.sub_eq_zero_of_le h, zero_mul]
    exact Nat.zero_le _
  · obtain ⟨x', rfl⟩ : ∃ x', x = k + x' := ⟨x - k, by omega⟩
    obtain ⟨d, rfl⟩ : ∃ d, y = k + (x' + d) := ⟨y - (k + x'), by omega⟩
    rw [Nat.add_sub_cancel_left, Nat.add_sub_cancel_left]
    nlinarith [Nat.zero_le (d * k)]

/-- One more wrong unit: `C(N − (c + 1), k) · (N − c) = C(N − c, k) · (N − c − k)`. -/
theorem choose_sub_succ_mul {N k c : ℕ} (hc : c < N) :
    ((N - (c + 1)).choose k : ℝ≥0∞) * ((N - c : ℕ) : ℝ≥0∞) = ((N - c).choose k : ℝ≥0∞) * ((N - c - k : ℕ) : ℝ≥0∞) := by
  have h := Nat.choose_mul_succ_eq (N - (c + 1)) k
  rw [show N - (c + 1) + 1 = N - c by omega] at h
  exact_mod_cast h

theorem le_of_mul_natCast_le {x y : ℝ≥0∞} {z : ℕ} (hz : 0 < z) (h : x * (z : ℝ≥0∞) ≤ y * (z : ℝ≥0∞)) : x ≤ y :=
  (ENNReal.mul_le_mul_iff_left (by exact_mod_cast hz.ne') (ENNReal.natCast_ne_top z)).1 h

/-- **The exchange, in one stratum of `N` units drawing `k`**: `C(N − c, k) · τ^a ≤ C(N − a, k) · τ^c` for every
`c ≤ N`, when `τ` is at most the factors `a` takes (`htake`, at the last one) and at least those it leaves (`hleave`, at
the next one), cross-multiplied. -/
theorem stratum_exchange {N k a c : ℕ} {τ : ℝ≥0∞} (ha : a ≤ N) (hc : c ≤ N)
    (htake : 0 < a → τ * ((N - (a - 1) : ℕ) : ℝ≥0∞) ≤ ((N - (a - 1) - k : ℕ) : ℝ≥0∞))
    (hleave : a < N → ((N - a - k : ℕ) : ℝ≥0∞) ≤ τ * ((N - a : ℕ) : ℝ≥0∞)) :
    ((N - c).choose k : ℝ≥0∞) * τ ^ a ≤ ((N - a).choose k : ℝ≥0∞) * τ ^ c := by
  set E : ℕ → ℝ≥0∞ := fun c => ((N - c).choose k : ℝ≥0∞) with hE
  have hup : ∀ i, a ≤ i → i < N → ((N - i - k : ℕ) : ℝ≥0∞) ≤ τ * ((N - i : ℕ) : ℝ≥0∞) := by
    intro i hai hi
    refine le_of_mul_natCast_le (z := N - a) (by omega) ?_
    calc ((N - i - k : ℕ) : ℝ≥0∞) * ((N - a : ℕ) : ℝ≥0∞) ≤ ((N - a - k : ℕ) : ℝ≥0∞) * ((N - i : ℕ) : ℝ≥0∞) := by
          exact_mod_cast count_ratio_anti hai hi.le
      _ ≤ τ * ((N - a : ℕ) : ℝ≥0∞) * ((N - i : ℕ) : ℝ≥0∞) := by gcongr; exact hleave (by omega)
      _ = τ * ((N - i : ℕ) : ℝ≥0∞) * ((N - a : ℕ) : ℝ≥0∞) := by ring
  have hdown : ∀ i, i < a → τ * ((N - i : ℕ) : ℝ≥0∞) ≤ ((N - i - k : ℕ) : ℝ≥0∞) := by
    intro i hia
    refine le_of_mul_natCast_le (z := N - (a - 1)) (by omega) ?_
    calc τ * ((N - i : ℕ) : ℝ≥0∞) * ((N - (a - 1) : ℕ) : ℝ≥0∞)
        = τ * ((N - (a - 1) : ℕ) : ℝ≥0∞) * ((N - i : ℕ) : ℝ≥0∞) := by ring
      _ ≤ ((N - (a - 1) - k : ℕ) : ℝ≥0∞) * ((N - i : ℕ) : ℝ≥0∞) := by gcongr; exact htake (by omega)
      _ ≤ ((N - i - k : ℕ) : ℝ≥0∞) * ((N - (a - 1) : ℕ) : ℝ≥0∞) := by
          exact_mod_cast count_ratio_anti (by omega) (by omega)
  have up : ∀ d, a + d ≤ N → E (a + d) ≤ E a * τ ^ d := by
    intro d
    induction d with
    | zero => intro _; simp
    | succ d ih =>
      intro hd
      have hlt : a + d < N := by omega
      have hstep : E (a + d + 1) ≤ E (a + d) * τ := by
        refine le_of_mul_natCast_le (z := N - (a + d)) (by omega) ?_
        rw [hE]
        simp only
        rw [choose_sub_succ_mul hlt, mul_assoc]
        gcongr
        exact hup _ (by omega) hlt
      calc E (a + (d + 1)) = E (a + d + 1) := by rw [Nat.add_assoc]
        _ ≤ E (a + d) * τ := hstep
        _ ≤ E a * τ ^ d * τ := by gcongr; exact ih (by omega)
        _ = E a * τ ^ (d + 1) := by rw [pow_succ, mul_assoc]
  have down : ∀ d, c + d ≤ a → E c * τ ^ d ≤ E (c + d) := by
    intro d
    induction d with
    | zero => intro _; simp
    | succ d ih =>
      intro hd
      have hlt : c + d < N := by omega
      have hstep : E (c + d) * τ ≤ E (c + d + 1) := by
        refine le_of_mul_natCast_le (z := N - (c + d)) (by omega) ?_
        rw [hE]
        simp only
        rw [choose_sub_succ_mul hlt, mul_assoc]
        gcongr
        exact hdown _ (by omega)
      calc E c * τ ^ (d + 1) = E c * τ ^ d * τ := by rw [pow_succ, mul_assoc]
        _ ≤ E (c + d) * τ := by gcongr; exact ih (by omega)
        _ ≤ E (c + d + 1) := hstep
        _ = E (c + (d + 1)) := by rw [Nat.add_assoc]
  show E c * τ ^ a ≤ E a * τ ^ c
  rcases le_or_gt a c with hac | hca
  · obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hac
    calc E (a + d) * τ ^ a ≤ E a * τ ^ d * τ ^ a := by gcongr; exact up d hc
      _ = E a * τ ^ (a + d) := by rw [mul_assoc, ← pow_add, Nat.add_comm d a]
  · obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hca.le
    calc E c * τ ^ (c + d) = E c * τ ^ d * τ ^ c := by rw [mul_assoc, ← pow_add, Nat.add_comm d c]
      _ ≤ E (c + d) * τ ^ c := by gcongr; exact down d le_rfl

/-- **The stratified count curve.** A count vector `a` that some `τ ∈ (0, 1]` separates, every factor it takes at least
`τ` and every factor it leaves at most `τ`, attains the stratified law's `miss` at `∑ a`: no set of at least `∑ a` wrong
units escapes more often than the sets with counts `a`, which escape with `∏ s C(n_s − a_s, k_s)/C(n_s, k_s)`. A greedy
fill, taking the largest factor left one unit at a time, has such a `τ`. -/
theorem stratified_miss_eq_greedy (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (a : Fin m → ℕ) (ha : ∀ s, a s ≤ (stratum σ s).card) {τ : ℝ≥0∞} (hτ0 : 0 < τ) (hτ1 : τ ≤ 1)
    (htake : ∀ s, 0 < a s → τ ≤ stratumRatio σ k s (a s - 1))
    (hleave : ∀ s, a s < (stratum σ s).card → stratumRatio σ k s (a s) ≤ τ) :
    (stratified σ k hk).miss (∑ s, a s) = ∏ s, stratumEscape σ k s (a s) := by
  classical
  have hfib : ∀ B : Finset (Fin n), B.card = ∑ s, (B.filter (σ · = s)).card := fun B =>
    Finset.card_eq_sum_card_fiberwise (fun u _ => Finset.mem_univ (σ u))
  refine le_antisymm (iSup₂_le fun B hB => ?_) ?_
  · rw [stratified_escape]
    set c : Fin m → ℕ := fun s => (B.filter (σ · = s)).card with hc
    have hcN : ∀ s, c s ≤ (stratum σ s).card := fun s => by
      rw [hc]; simp only
      rw [filter_eq_inter_stratum]; exact card_le_card inter_subset_right
    have hsum : ∑ s, a s ≤ ∑ s, c s := (hfib B) ▸ hB
    have hex : ∀ s, stratumEscape σ k s (c s) * τ ^ a s ≤ stratumEscape σ k s (a s) * τ ^ c s := by
      intro s
      have hN := ha s
      have e := stratum_exchange (N := (stratum σ s).card) (k := k s) (τ := τ) hN (hcN s)
        (fun h => by
          have := htake s h
          unfold stratumRatio at this
          have hpos : ((stratum σ s).card - (a s - 1) : ℕ) ≠ 0 := by omega
          rwa [ENNReal.le_div_iff_mul_le (Or.inl (by exact_mod_cast hpos)) (Or.inl (ENNReal.natCast_ne_top _))]
            at this)
        (fun h => by
          have := hleave s h
          unfold stratumRatio at this
          have hpos : ((stratum σ s).card - a s : ℕ) ≠ 0 := by omega
          rwa [ENNReal.div_le_iff_le_mul (Or.inl (by exact_mod_cast hpos)) (Or.inl (ENNReal.natCast_ne_top _))]
            at this)
      unfold stratumEscape
      calc _ = ((((stratum σ s).card - c s).choose (k s) : ℝ≥0∞) * τ ^ a s) /
              ((stratum σ s).card.choose (k s) : ℝ≥0∞) := by rw [div_eq_mul_inv, div_eq_mul_inv]; ring
        _ ≤ ((((stratum σ s).card - a s).choose (k s) : ℝ≥0∞) * τ ^ c s) /
              ((stratum σ s).card.choose (k s) : ℝ≥0∞) := ENNReal.div_le_div_right e _
        _ = _ := by rw [div_eq_mul_inv, div_eq_mul_inv]; ring
    have hτt : τ ^ (∑ s, a s) ≠ ⊤ := ENNReal.pow_ne_top (ne_top_of_le_ne_top ENNReal.one_ne_top hτ1)
    refine (ENNReal.mul_le_mul_iff_left (pow_ne_zero _ hτ0.ne') hτt).1 ?_
    calc (∏ s, stratumEscape σ k s (c s)) * τ ^ (∑ s, a s) = ∏ s, stratumEscape σ k s (c s) * τ ^ a s := by
          rw [Finset.prod_mul_distrib, Finset.prod_pow_eq_pow_sum]
      _ ≤ ∏ s, stratumEscape σ k s (a s) * τ ^ c s := Finset.prod_le_prod fun s _ => hex s
      _ = (∏ s, stratumEscape σ k s (a s)) * τ ^ (∑ s, c s) := by
          rw [Finset.prod_mul_distrib, Finset.prod_pow_eq_pow_sum]
      _ ≤ (∏ s, stratumEscape σ k s (a s)) * τ ^ (∑ s, a s) :=
          mul_le_mul_right (pow_le_pow_right_of_le_one' hτ1 hsum) _
  · choose T hTs hTc using fun s => Finset.exists_subset_card_eq (ha s)
    set B := univ.biUnion T with hBdef
    have hfil : ∀ s, B.filter (σ · = s) = T s := by
      intro s
      ext u
      simp only [hBdef, mem_filter, mem_biUnion, mem_univ, true_and]
      constructor
      · rintro ⟨⟨t, ht⟩, rfl⟩
        have : σ u = t := mem_stratum.1 (hTs t ht)
        rwa [this]
      · intro hu
        exact ⟨⟨s, hu⟩, mem_stratum.1 (hTs s hu)⟩
    have hcard : ∑ s, a s ≤ B.card := by
      rw [hfib B]
      exact le_of_eq (Finset.sum_congr rfl fun s _ => by rw [hfil, hTc])
    refine le_trans (le_of_eq ?_) ((stratified σ k hk).escape_le_miss hcard)
    rw [stratified_escape]
    exact Finset.prod_congr rfl fun s _ => by rw [hfil, hTc]; rfl

end Law

end FlockSoundness.Audit
