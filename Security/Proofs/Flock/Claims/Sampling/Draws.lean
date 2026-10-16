import Proofs.Flock.Claims.Prob
import Proofs.Flock.Soundness.Audit.Work

/-!
# The draw bounds, as the claims read probabilities

The sampling compiler's knowledge error carries the draw's term: if at least `t` regions are wrong, the verifier's draw
misses all of them with probability at most its law's bound. Each registered law (`Flock.Draw.Law`: `subset`,
`bernoulli`, `stratified`, `work`) has that bound proved already, in `ℝ≥0∞` over the audit layer's `prCoin`
(`FlockSoundness.Audit.Law`). This file states each in `Claims.Pr`, the uniform average the claims use, through
`Pr_eq_toReal_prCoin`:

* `subset_miss_eq`, `subset_miss_le`: a uniform `k`-subset misses `B` with probability `C(n − |B|, k)/C(n, k)`, at most
  `C(n − t, k)/C(n, k)` when `t ≤ |B|` (`Law.subset_escape`);
* `bernoulli_miss_eq`, `bernoulli_miss_le`: each unit drawn when its coin in `[0, den)` is below `num` misses `B` with
  probability `((den − num)/den)^|B|`, at most `((den − num)/den)^t` (`Law.bernoulli_escape`);
* `stratified_miss_eq`: independent uniform `k_s`-subsets of the strata miss `B` with probability
  `∏ s C(n_s − m_s, k_s)/C(n_s, k_s)` (`Law.stratified_escape`), and `stratified_miss_le`: at most `(1 − π)^t` when every
  non-empty stratum draws at least a fraction `π` of its units;
* `work_miss_le`: the work law misses units doing at least `T` of the work `W` with probability at most
  `((W − T)/W)^K` (`Law.work_escape_le`);
* `miss_le_count`: any law misses `B` with at most its count curve at `t ≤ |B|` (`Law.miss`), which
  `Law.stratified_miss_eq_greedy` computes for the stratified and work laws.

`miss_le_of` turns each into its form at `δ`: a bound of at most `δ` at `t` is a miss probability of at most `δ`.
-/

namespace Claims.Sampling

open FlockSoundness.Audit FlockSoundness.Game Finset
open scoped ENNReal

/-- `Claims.Pr` is the audit layer's `prCoin`, read in `ℝ`. -/
theorem Pr_eq_toReal_prCoin {α : Type} [Fintype α] (p : α → Prop) : Claims.Pr p = (prCoin p).toReal := by
  classical
  rw [prCoin_eq_card, ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]
  unfold Claims.Pr
  congr 2

/-- A law's miss probability, in the claims' terms, is its escape. -/
theorem Pr_miss_eq {n : ℕ} (L : Law n) (B : Finset (Fin n)) :
    Claims.Pr (fun ω : L.Ω => Disjoint (L.draw ω) B) = (L.escape B).toReal :=
  Pr_eq_toReal_prCoin _

/-- A miss probability below an escape bound `e` that is finite. -/
theorem Pr_miss_le {n : ℕ} (L : Law n) (B : Finset (Fin n)) {e : ℝ≥0∞} (he : e ≠ ⊤) (h : L.escape B ≤ e) :
    Claims.Pr (fun ω : L.Ω => Disjoint (L.draw ω) B) ≤ e.toReal := by
  rw [Pr_miss_eq]
  exact ENNReal.toReal_mono he h

/-- **The form at `δ`**: a miss bound `b t` of at most `δ` at the `t` wrong regions is a miss probability of at most
`δ`. -/
theorem miss_le_of {p b δ : ℝ} (h : p ≤ b) (hb : b ≤ δ) : p ≤ δ := h.trans hb

/-! ## The uniform `k`-subset -/

/-- **The uniform `k`-subset misses `B`** with probability `C(n − |B|, k)/C(n, k)`. -/
theorem subset_miss_eq {n k : ℕ} (hk : k ≤ n) (B : Finset (Fin n)) :
    Claims.Pr (fun S : {S : Finset (Fin n) // S.card = k} => Disjoint S.1 B) =
      ((n - B.card).choose k : ℝ) / (n.choose k : ℝ) := by
  have := Pr_miss_eq (Law.subset n k hk) B
  rw [Law.subset_escape, ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast] at this
  exact this

/-- **The uniform `k`-subset misses `t` wrong regions** with probability at most `C(n − t, k)/C(n, k)`. -/
theorem subset_miss_le {n k t : ℕ} (hk : k ≤ n) {B : Finset (Fin n)} (ht : t ≤ B.card) :
    Claims.Pr (fun S : {S : Finset (Fin n) // S.card = k} => Disjoint S.1 B) ≤
      ((n - t).choose k : ℝ) / (n.choose k : ℝ) := by
  rw [subset_miss_eq hk]
  gcongr

/-! ## Bernoulli -/

theorem bernFactor_zero (num den : ℕ) [NeZero den] :
    (Law.bernFactor num den 0).toReal = ((den - num : ℕ) : ℝ) / den := by
  unfold Law.bernFactor
  rw [mul_zero, zero_add, ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]

/-- **The Bernoulli draw misses `B`** with probability `((den − num)/den)^|B|`: unit `u` is drawn when its coin
`ω u ∈ [0, den)` is below `num`. -/
theorem bernoulli_miss_eq {n num den : ℕ} [NeZero den] (h : num ≤ den) (B : Finset (Fin n)) :
    Claims.Pr (fun ω : Fin n → Fin den => Disjoint (univ.filter fun u => (ω u).val < num) B) =
      (((den - num : ℕ) : ℝ) / den) ^ B.card := by
  have := Pr_miss_eq (Law.bernoulli n num den) B
  rw [Law.bernoulli_escape h, ENNReal.toReal_pow, bernFactor_zero num den] at this
  exact this

/-- **The Bernoulli draw misses `t` wrong regions** with probability at most `((den − num)/den)^t`. -/
theorem bernoulli_miss_le {n num den t : ℕ} [NeZero den] (h : num ≤ den) {B : Finset (Fin n)} (ht : t ≤ B.card) :
    Claims.Pr (fun ω : Fin n → Fin den => Disjoint (univ.filter fun u => (ω u).val < num) B) ≤
      (((den - num : ℕ) : ℝ) / den) ^ t := by
  rw [bernoulli_miss_eq h]
  have hd : (0 : ℝ) < den := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne den)
  refine pow_le_pow_of_le_one (by positivity) ?_ ht
  rw [div_le_one hd]
  exact_mod_cast Nat.sub_le den num

/-! ## Stratified -/

/-- **The stratified draw misses `B`** with probability `∏ s C(n_s − m_s, k_s)/C(n_s, k_s)`, with `n_s` the stratum's
units and `m_s` those of `B`. -/
theorem stratified_miss_eq {n m : ℕ} (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (Law.stratum σ s).card)
    (B : Finset (Fin n)) :
    Claims.Pr (fun ω : (Law.stratified σ k hk).Ω => Disjoint ((Law.stratified σ k hk).draw ω) B) =
      ∏ s, (((Law.stratum σ s).card - (B.filter (σ · = s)).card).choose (k s) : ℝ) /
        ((Law.stratum σ s).card.choose (k s) : ℝ) := by
  rw [Pr_miss_eq, Law.stratified_escape, ENNReal.toReal_prod]
  refine Finset.prod_congr rfl fun s _ => ?_
  rw [ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]

/-- One hypergeometric factor, at most the per-unit miss to the power of the wrong units: `C(N − b, k)/C(N, k) ≤
(1 − π)^b` when `π·N ≤ k` and `b ≤ N`. -/
theorem choose_ratio_le_pow {N k b : ℕ} {π : ℝ} (hk : k ≤ N) (hb : b ≤ N)
    (hπ : 0 < N → π * N ≤ k) : ((N - b).choose k : ℝ) / (N.choose k : ℝ) ≤ (1 - π) ^ b := by
  rcases Nat.eq_zero_or_pos N with h0 | hN
  · obtain rfl : b = 0 := by omega
    obtain rfl : k = 0 := by omega
    simp [h0]
  have hC : (0 : ℝ) < N.choose k := by exact_mod_cast Nat.choose_pos hk
  have hNr : (0 : ℝ) < N := by exact_mod_cast hN
  have key := Law.choose_sub_mul_pow_le N k b hb
  have h1 : ((N - b).choose k : ℝ) / (N.choose k : ℝ) ≤ (((N - k : ℕ) : ℝ) / N) ^ b := by
    rw [div_pow, div_le_div_iff₀ hC (by positivity)]
    exact_mod_cast key.trans_eq (mul_comm _ _)
  refine h1.trans (pow_le_pow_left₀ (by positivity) ?_ b)
  rw [div_le_iff₀ hNr, Nat.cast_sub hk]
  nlinarith [hπ hN]

/-- **The stratified draw misses `t` wrong regions** with probability at most `(1 − π)^t`, when every non-empty
stratum draws at least a fraction `π ≤ 1` of its units: each unit is drawn with probability at least `π`. -/
theorem stratified_miss_le {n m t : ℕ} (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (Law.stratum σ s).card)
    {π : ℝ} (hπ0 : 0 ≤ π) (hπ1 : π ≤ 1) (hπ : ∀ s, 0 < (Law.stratum σ s).card → π * (Law.stratum σ s).card ≤ k s)
    {B : Finset (Fin n)} (ht : t ≤ B.card) :
    Claims.Pr (fun ω : (Law.stratified σ k hk).Ω => Disjoint ((Law.stratified σ k hk).draw ω) B) ≤ (1 - π) ^ t := by
  classical
  rw [stratified_miss_eq]
  have hb : ∀ s, (B.filter (σ · = s)).card ≤ (Law.stratum σ s).card := fun s =>
    card_le_card (filter_subset_filter _ (subset_univ B))
  have hsum : ∑ s, (B.filter (σ · = s)).card = B.card := by
    rw [← card_biUnion]
    · congr 1
      ext u
      simp
    · intro s _ s' _ hss'
      exact disjoint_filter.2 fun u _ h h' => hss' (h.symm.trans h')
  calc ∏ s, (((Law.stratum σ s).card - (B.filter (σ · = s)).card).choose (k s) : ℝ) /
        ((Law.stratum σ s).card.choose (k s) : ℝ)
      ≤ ∏ s, (1 - π) ^ (B.filter (σ · = s)).card :=
        prod_le_prod₀ (fun s _ => by positivity) fun s _ => choose_ratio_le_pow (hk s) (hb s) (hπ s)
    _ = (1 - π) ^ B.card := by rw [prod_pow_eq_pow_sum, hsum]
    _ ≤ (1 - π) ^ t := pow_le_pow_of_le_one (by linarith) (by linarith) ht

/-! ## Work -/

/-- **The work draw misses units doing at least `T` of the work** `W` with probability at most `((W − T)/W)^K`. -/
theorem work_miss_le {n m : ℕ} (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) (hf : ∀ s, 1 ≤ f s)
    (hW : 0 < Law.totalWork σ w) {B : Finset (Fin n)} {T : ℕ} (hT : T ≤ Law.workOf σ w B) :
    Claims.Pr (fun ω : (Law.work σ w f K).Ω => Disjoint ((Law.work σ w f K).draw ω) B) ≤
      (((Law.totalWork σ w - T : ℕ) : ℝ) / (Law.totalWork σ w : ℝ)) ^ K := by
  have hW0 : (Law.totalWork σ w : ℝ≥0∞) ≠ 0 := by exact_mod_cast hW.ne'
  refine (Pr_miss_le _ B (ENNReal.pow_ne_top (ENNReal.div_ne_top (ENNReal.natCast_ne_top _) hW0))
    (Law.work_escape_le_of_le σ w f K hf hW hT)).trans_eq ?_
  rw [ENNReal.toReal_pow, ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]

/-! ## Any law: its count curve -/

/-- **Any law misses `t` wrong regions** with probability at most its count curve at `t`, the largest escape of a set
of at least `t` units. -/
theorem miss_le_count {n t : ℕ} (L : Law n) {B : Finset (Fin n)} (ht : t ≤ B.card) :
    Claims.Pr (fun ω : L.Ω => Disjoint (L.draw ω) B) ≤ (L.miss t).toReal :=
  Pr_miss_le L B (ne_top_of_le_ne_top ENNReal.one_ne_top (L.miss_le_one t)) (L.escape_le_miss ht)

end Claims.Sampling
