import Models.Accounting.Guarantees.RemoteDraw
import Proofs.Accounting.RemoteHeld.OverTime
import Proofs.Accounting.RateAudit.Horizon
import Proofs.Accounting.Prob
import Proofs.Flock.Soundness.Audit.Work

/-!
# The openings' pooled bound: the proof

Window `t`'s size `U_t` and correct receipts are fixed before its openings coin (`Live`), so given the earlier windows
its draw of `s'_t` distinct receipts lands on its `g_t` correct ones with probability `C(g_t, s'_t)/C(U_t, s'_t)`, at
most `(g_t/U_t)^(s'_t)` and `0` when `g_t < s'_t` (`Draw.pr_subset_le`, from the uniform subset's hypergeometric law
`FlockSoundness.Audit.Law.card_disjoint_subsets` and `choose_div_le_pow'`). Such per-round bounds, each read off the
rounds before, multiply: every round's event holds while their product is at most `c` with probability at most `c`
(`Draw.pr_live_le`, peeling round 0 as `Rate.Horizon.pr_causal_le` does). When the wrong receipts pass the pooled bound,
the wrong shares `f_t = 1 − g_t/U_t` are no point of its knapsack, so a window has fewer correct receipts than draws, or
`Σ_t s'_t·f_t` passes the budget `S·(1 − δ^(1/S))` and weighted AM–GM puts the product at most `δ`
(`Draw.prod_le_of_pooled_lt`). `PooledDraw` is the two at `c = δ`; `LiveCoinsDrawn` and the memory line at the record's
pooled bound (`HeldDrawn` and its time-held forms) follow from it as `Held` does from `LiveCoins`.
-/

namespace Accounting.SecurityProofs.Remote

open Finset Accounting.Rate Accounting.Rate.Horizon
open Pous (pr)
open scoped ENNReal
open Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Draw
  Accounting.Remote.Draw.Assumptions

namespace Draw

/-! ## Adaptive rounds -/

/-- Rounds `0, …, n − 1`, each with coins `Y i` and then `Z i`. Round `i`'s event `A i` reads the coins of rounds up to
`i`, and `h i`, read off the earlier rounds and round `i`'s first coins, bounds the probability of `A i` over its second
coins. Then every event holds while the bounds multiply to at most `c` with probability at most `c`. -/
theorem pr_live_le : ∀ (n : ℕ) (Y Z : Fin n → Type) [∀ i, Fintype (Y i)] [∀ i, Fintype (Z i)]
    (A : (i : Fin n) → ((j : Fin n) → Y j × Z j) → Prop) (h : (i : Fin n) → ((j : Fin n) → Y j × Z j) → ℝ≥0∞),
    (∀ i ω ω', (∀ j, j ≤ i → ω j = ω' j) → (A i ω ↔ A i ω')) →
    (∀ i ω ω', (∀ j, j < i → ω j = ω' j) → (ω i).1 = (ω' i).1 → h i ω = h i ω') →
    (∀ i ω, pr (fun z : Z i => A i (Function.update ω i ((ω i).1, z))) ≤ h i ω) →
    ∀ c, pr (fun ω => (∀ i, A i ω) ∧ ∏ i, h i ω ≤ c) ≤ c := by
  intro n
  induction n with
  | zero =>
    intro Y Z _ _ A h _ _ _ c
    by_cases hc : 1 ≤ c
    · exact (pr_le_one _).trans hc
    · refine (Bandwidth.pr_mono (fun ω hω => (hc (by simpa using hω.2) : False))).trans ?_
      rw [pr_false]
      exact zero_le
  | succ n ih =>
    intro Y Z _ _ A h hA hh hb c
    classical
    let T : Type := (i : Fin n) → Y i.succ × Z i.succ
    let e : ((j : Fin (n + 1)) → Y j × Z j) ≃ (Z 0 × T) × Y 0 :=
      { toFun := fun ω => (((ω 0).2, Fin.tail ω), (ω 0).1)
        invFun := fun q => Fin.cons (q.2, q.1.1) q.1.2
        left_inv := fun ω => by simp
        right_inv := fun q => by simp }
    rw [pr_equiv e]
    refine pr_le_of_fiber _ _ (fun y₀ => ?_)
    simp only [e, Equiv.coe_fn_symm_mk]
    rcases isEmpty_or_nonempty (Z 0 × T) with hE | ⟨⟨z₀, t₀⟩⟩
    · exact (le_of_eq (Bandwidth.pr_eq_zero_of_card _ Fintype.card_eq_zero)).trans zero_le
    set H := h 0 (Fin.cons (y₀, z₀) t₀ : (j : Fin (n + 1)) → Y j × Z j) with hHdef
    have hH : ∀ z t, h 0 (Fin.cons (y₀, z) t : (j : Fin (n + 1)) → Y j × Z j) = H :=
      fun z t => hh 0 _ _ (fun j hj => absurd hj (Fin.not_lt_zero j)) rfl
    let A₀ : Z 0 → Prop := fun z => A 0 (Fin.cons (y₀, z) t₀ : (j : Fin (n + 1)) → Y j × Z j)
    have hA₀ : ∀ z t, A 0 (Fin.cons (y₀, z) t : (j : Fin (n + 1)) → Y j × Z j) ↔ A₀ z := fun z t =>
      hA 0 _ _ (fun j hj => by
        obtain rfl : j = 0 := Fin.le_zero_iff.1 hj
        rfl)
    have hpA : pr A₀ ≤ H := by
      have := hb 0 (Fin.cons (y₀, z₀) t₀)
      simpa [Fin.update_cons_zero, A₀] using this
    let A' : Z 0 → (i : Fin n) → T → Prop := fun z i t => A i.succ (Fin.cons (y₀, z) t)
    let h' : Z 0 → (i : Fin n) → T → ℝ≥0∞ := fun z i t => h i.succ (Fin.cons (y₀, z) t)
    have hrest : ∀ z c', pr (fun t : T => (∀ i, A' z i t) ∧ ∏ i, h' z i t ≤ c') ≤ c' := by
      intro z
      refine ih (fun i => Y i.succ) (fun i => Z i.succ) (A' z) (h' z) ?_ ?_ ?_
      · intro i ω ω' hω
        refine hA i.succ _ _ (fun j hj => ?_)
        cases j using Fin.cases with
        | zero => rfl
        | succ j => simpa using hω j (Fin.succ_le_succ_iff.1 hj)
      · intro i ω ω' hω h1
        refine hh i.succ _ _ (fun j hj => ?_) (by simpa using h1)
        cases j using Fin.cases with
        | zero => rfl
        | succ j => simpa using hω j (Fin.succ_lt_succ_iff.1 hj)
      · intro i ω
        have := hb i.succ (Fin.cons (y₀, z) ω)
        simpa [A', h', Fin.cons_update] using this
    let B : Z 0 × T → Prop := fun p => (∀ i, A' p.1 i p.2) ∧ (∏ i, h' p.1 i p.2) * H ≤ c
    refine (Bandwidth.pr_mono (fun p hp => (?_ : A₀ p.1 ∧ B p))).trans ?_
    · obtain ⟨hall, hprod⟩ := hp
      refine ⟨(hA₀ p.1 p.2).1 (hall 0), fun i => hall i.succ, ?_⟩
      rw [Fin.prod_univ_succ, hH, mul_comm] at hprod
      exact hprod
    by_cases hc : c = ⊤
    · rw [hc]; exact le_top
    by_cases hH0 : H = 0
    · have hA0 : pr A₀ = 0 := le_antisymm (hpA.trans hH0.le) zero_le
      refine (pr_and_le_mul A₀ B ⊤ (fun _ _ => le_top)).trans ?_
      rw [hA0, zero_mul]
      exact zero_le
    refine (pr_and_le_mul A₀ B (c / H) (fun z _ => ?_)).trans ?_
    · refine (Bandwidth.pr_mono (fun t ht => ⟨ht.1, ?_⟩)).trans (hrest z (c / H))
      exact (ENNReal.le_div_iff_mul_le (Or.inl hH0) (Or.inr hc)).2 ht.2
    calc pr A₀ * (c / H) ≤ H * (c / H) := by gcongr
      _ ≤ c := ENNReal.mul_div_le

/-! ## One window's draw -/

/-- An event that reads one coordinate of a product has at most that coordinate's probability. -/
theorem pr_coord_le {ι : Type} [Fintype ι] [DecidableEq ι] {β : ι → Type} [∀ i, Fintype (β i)] (i : ι)
    (P : β i → Prop) : pr (fun z : (j : ι) → β j => P (z i)) ≤ pr P := by
  rw [pr_equiv (Equiv.piSplitAt i β)]
  refine pr_le_of_fiber _ _ (fun y => ?_)
  simp [Equiv.piSplitAt_symm_apply]

/-- A uniform `k`-subset of `U` lands inside `C` with probability at most `(|C|/U)^k`, and never when `|C| < k`. -/
theorem pr_subset_le {U k : ℕ} (hk : k ≤ U) (C : Finset (Fin U)) :
    pr (fun S : {S : Finset (Fin U) // S.card = k} => S.1 ⊆ C) ≤
      ENNReal.ofReal (if k ≤ C.card then ((C.card : ℝ) / U) ^ k else 0) := by
  classical
  have hCU : C.card ≤ U := (card_le_univ C).trans_eq (Fintype.card_fin U)
  have hcard : pr (fun S : {S : Finset (Fin U) // S.card = k} => S.1 ⊆ C) =
      ((C.card.choose k : ℕ) : ℝ≥0∞) / (U.choose k : ℕ) := by
    unfold pr
    have h1 := FlockSoundness.Audit.Law.card_disjoint_subsets k Cᶜ
    rw [card_compl, Fintype.card_fin, Nat.sub_sub_self hCU] at h1
    rw [Fintype.card_finset_len, Fintype.card_fin, ← h1]
    congr 3
    ext S
    simpa only [mem_filter, mem_univ, true_and] using (disjoint_compl_right_iff (x := S.1) (y := C)).symm
  rw [hcard]
  by_cases hkC : k ≤ C.card
  swap
  · rw [Nat.choose_eq_zero_of_lt (not_le.1 hkC), Nat.cast_zero, ENNReal.zero_div]
    exact zero_le
  simp only [hkC, ↓reduceIte]
  rcases Nat.eq_zero_or_pos U with rfl | hU
  · obtain rfl : k = 0 := Nat.le_zero.1 hk
    simp
  have h := FlockSoundness.Audit.Law.choose_div_le_pow' (m := U - C.card) hk hU
  rw [Nat.sub_sub_self hCU] at h
  refine h.trans (le_of_eq ?_)
  rw [ENNReal.ofReal_pow (by positivity), ENNReal.ofReal_div_of_pos (by exact_mod_cast hU),
    ENNReal.ofReal_natCast, ENNReal.ofReal_natCast]

/-! ## The pooled bound -/

/-- Wrong receipts beyond the pooled bound leave the windows' draws, all landing on correct receipts, a probability
of at most `δ`: `∏_t (g_t/U_t)^(d_t) ≤ δ`, and `0` when a window has fewer correct receipts than draws. -/
theorem prod_le_of_pooled_lt {n : ℕ} (U g d : Fin n → ℕ) {δ : ℝ} (hδ : 0 ≤ δ) (hg : ∀ t, g t ≤ U t)
    (h : pooled n U d δ < ∑ t, ((U t - g t : ℕ) : ℝ)) :
    ∏ t, (if d t ≤ g t then ((g t : ℝ) / U t) ^ d t else 0) ≤ δ := by
  by_cases hdg : ∀ t, d t ≤ g t
  swap
  · push Not at hdg
    obtain ⟨t, ht⟩ := hdg
    rw [Finset.prod_eq_zero (Finset.mem_univ t) (by simp [not_le.2 ht])]
    exact hδ
  simp only [hdg, ↓reduceIte]
  set f : Fin n → ℝ := fun t => ((U t - g t : ℕ) : ℝ) / U t with hf
  have hfU : ∀ t, f t * U t = ((U t - g t : ℕ) : ℝ) := by
    intro t
    rcases Nat.eq_zero_or_pos (U t) with h0 | hpos
    · have : g t = 0 := by have := hg t; omega
      simp [hf, h0, this]
    · have hU : (U t : ℝ) ≠ 0 := by positivity
      simp only [hf]
      field_simp
  have hbox : ∀ t, 0 ≤ f t ∧ f t ≤ 1 - (d t : ℝ) / U t := by
    intro t
    refine ⟨by positivity, ?_⟩
    rcases Nat.eq_zero_or_pos (U t) with h0 | hpos
    · simp [hf, h0]
    · have hU : (0 : ℝ) < U t := by exact_mod_cast hpos
      have hd : (d t : ℝ) ≤ g t := by exact_mod_cast hdg t
      simp only [hf]
      rw [Nat.cast_sub (hg t), sub_div, div_self hU.ne']
      gcongr
  have hbdd : BddAbove ((fun f : Fin n → ℝ => ∑ t, f t * U t) ''
      {f | (∀ t, 0 ≤ f t ∧ f t ≤ 1 - (d t : ℝ) / U t) ∧ ∑ t, (d t : ℝ) * f t ≤ budget (∑ t, d t) δ}) := by
    refine ⟨∑ t, (U t : ℝ), ?_⟩
    rintro _ ⟨f', ⟨hf', -⟩, rfl⟩
    refine Finset.sum_le_sum (fun t _ => ?_)
    have h1 : f' t ≤ 1 := (hf' t).2.trans (by have : 0 ≤ (d t : ℝ) / U t := by positivity
                                              linarith)
    nlinarith [(hf' t).1, (Nat.cast_nonneg (U t) : (0 : ℝ) ≤ U t)]
  have hover : budget (∑ t, d t) δ < ∑ t, (d t : ℝ) * f t := by
    by_contra hle
    push Not at hle
    have := le_csSup hbdd ⟨f, ⟨hbox, hle⟩, rfl⟩
    simp only [hfU] at this
    exact absurd h (not_lt.2 this)
  set S := ∑ t, d t with hS
  have hd0 : S = 0 → ∀ t, d t = 0 := fun h0 t => by
    have := Finset.single_le_sum (fun i _ => Nat.zero_le (d i)) (Finset.mem_univ t)
    omega
  have hSpos : 0 < S := by
    rcases Nat.eq_zero_or_pos S with h0 | h0
    · have hz : ∑ t, (d t : ℝ) * f t = 0 := Finset.sum_eq_zero (fun t _ => by simp [hd0 h0 t])
      rw [hz, budget, h0] at hover
      simp at hover
    · exact h0
  set x : Fin n → ℝ := fun t => (g t : ℝ) / U t with hx
  have hx0 : ∀ t, 0 ≤ x t := fun t => by positivity
  have hdx : ∀ t, (d t : ℝ) * x t = d t - d t * f t := by
    intro t
    rcases Nat.eq_zero_or_pos (U t) with h0 | hpos
    · have : d t = 0 := by have := hdg t; have := hg t; omega
      simp [this]
    · have hU : (U t : ℝ) ≠ 0 := by positivity
      simp only [hx, hf]
      rw [Nat.cast_sub (hg t)]
      field_simp
      ring
  have hSR0 : (0 : ℝ) < S := by exact_mod_cast hSpos
  have hsum : ∑ t, (d t : ℝ) * x t < S * δ ^ ((S : ℝ)⁻¹) := by
    rw [Finset.sum_congr rfl (fun t _ => hdx t), Finset.sum_sub_distrib, ← Nat.cast_sum, ← hS]
    unfold budget at hover
    linarith
  have hamgm := Real.geom_mean_le_arith_mean Finset.univ (fun t => (d t : ℝ)) x (fun t _ => Nat.cast_nonneg _)
    (by rw [← Nat.cast_sum, ← hS]; exact hSR0) (fun t _ => hx0 t)
  have hSR : (∑ t, (d t : ℝ)) = S := by rw [hS, Nat.cast_sum]
  have hP : ∏ t, x t ^ (d t : ℝ) = ∏ t, x t ^ d t :=
    Finset.prod_congr rfl (fun t _ => Real.rpow_natCast _ _)
  rw [hSR, hP] at hamgm
  have hP0 : 0 ≤ ∏ t, x t ^ d t := Finset.prod_nonneg (fun t _ => pow_nonneg (hx0 t) _)
  have hroot : (∏ t, x t ^ d t) ^ ((S : ℝ)⁻¹) ≤ δ ^ ((S : ℝ)⁻¹) := by
    refine hamgm.trans ?_
    rw [div_le_iff₀ hSR0]
    linarith
  calc ∏ t, x t ^ d t = ((∏ t, x t ^ d t) ^ ((S : ℝ)⁻¹)) ^ S :=
        (Real.rpow_inv_natCast_pow hP0 hSpos.ne').symm
    _ ≤ (δ ^ ((S : ℝ)⁻¹)) ^ S := pow_le_pow_left₀ (Real.rpow_nonneg hP0 _) hroot S
    _ = δ := Real.rpow_inv_natCast_pow hδ hSpos.ne'

/-- `PooledDraw` at one session. -/
theorem pooledDraw {n m : ℕ} {s : Fin n → ℕ} {X : Fin n → Type} [∀ t, Fintype (X t)] (D : Session n m s X)
    (T : Trace (Coins n m s X)) {δ : ℝ} (hδ : 0 ≤ δ) (hL : Live D) (hR : Refused D T) (hC : Counted D T) :
    T.Holds δ (fun ω => (T.wrong ω : ℝ) ≤ D.pooled δ ω) := by
  classical
  let r : Fin n → Coins n m s X → ℝ := fun t ω =>
    if D.draws t ω ≤ D.good t ω then ((D.good t ω : ℝ) / (D.size t ω : ℕ)) ^ D.draws t ω else 0
  have hr0 : ∀ t ω, 0 ≤ r t ω := fun t ω => by
    simp only [r]
    split_ifs
    · positivity
    · exact le_rfl
  have hgood : ∀ t ω, D.good t ω ≤ D.size t ω := fun t ω =>
    (card_le_univ _).trans_eq (Fintype.card_fin _)
  let A : (t : Fin n) → Coins n m s X → Prop := fun t ω => D.drawn t ω ⊆ D.correct t ω
  have hA : ∀ t (ω ω' : Coins n m s X), (∀ j, j ≤ t → ω j = ω' j) → (A t ω ↔ A t ω') := by
    intro t ω ω' hω
    have ht : ω t = ω' t := hω t le_rfl
    obtain ⟨h1, h2⟩ := hL t ω ω' (fun j hj => hω j hj.le) (by rw [ht])
    simp only [A, Session.drawn]
    rw [ht, h1, h2]
  have hh : ∀ t (ω ω' : Coins n m s X), (∀ j, j < t → ω j = ω' j) → (ω t).1 = (ω' t).1 →
      ENNReal.ofReal (r t ω) = ENNReal.ofReal (r t ω') := by
    intro t ω ω' hω h1
    obtain ⟨hs, hc⟩ := hL t ω ω' hω h1
    have e1 : D.draws t ω = D.draws t ω' := by simp only [Session.draws, hs]
    have e2 : D.good t ω = D.good t ω' := by
      unfold Session.good
      rw [hc, hs]
    have e3 : ((D.size t ω : ℕ) : ℝ) = (D.size t ω' : ℕ) := by rw [hs]
    simp only [r, e1, e2, e3]
  have hb : ∀ t (ω : Coins n m s X),
      pr (fun z : Openings m (s t) => A t (Function.update ω t ((ω t).1, z))) ≤ ENNReal.ofReal (r t ω) := by
    intro t ω
    have hLz : ∀ z : Openings m (s t), D.size t (Function.update ω t ((ω t).1, z)) = D.size t ω ∧
        D.correct t (Function.update ω t ((ω t).1, z)) = D.correct t ω := fun z =>
      hL t _ ω (fun j hj => Function.update_of_ne hj.ne _ _) (by simp)
    let C' : Finset (Fin (D.size t ω)) := univ.filter fun i : Fin (D.size t ω) => (i : ℕ) ∈ D.correct t ω
    refine (Bandwidth.pr_mono (fun z hz => (?_ : (z (D.size t ω)).1 ⊆ C'))).trans ?_
    · simp only [A, Session.drawn, Function.update_self] at hz
      rw [(hLz z).1, (hLz z).2] at hz
      intro i hi
      simpa [C'] using hz (Finset.mem_map_of_mem _ hi)
    refine (pr_coord_le (β := fun U : Fin (m + 1) => {S : Finset (Fin U) // S.card = min (s t) U})
      (D.size t ω) (fun S => S.1 ⊆ C')).trans ?_
    exact pr_subset_le (min_le_right _ _) C'
  unfold Trace.Holds
  rw [Accounting.pr_eq_toReal]
  refine ENNReal.toReal_le_of_le_ofReal hδ ?_
  refine (Bandwidth.pr_mono (fun ω hω => (?_ : (∀ t, A t ω) ∧
      ∏ t, ENNReal.ofReal (r t ω) ≤ ENNReal.ofReal δ))).trans
    (pr_live_le n X (fun t => Openings m (s t)) A (fun t ω => ENNReal.ofReal (r t ω)) hA hh hb _)
  obtain ⟨hacc, hw⟩ := hω
  refine ⟨hR ω hacc, ?_⟩
  rw [← ENNReal.ofReal_prod_of_nonneg (fun t _ => hr0 t ω)]
  refine ENNReal.ofReal_le_ofReal (prod_le_of_pooled_lt (fun t => D.size t ω) (fun t => D.good t ω)
    (fun t => D.draws t ω) hδ (fun t => hgood t ω) ?_)
  have hw' : D.pooled δ ω < T.wrong ω := not_le.1 hw
  have hc : (T.wrong ω : ℝ) ≤ D.wrong ω := by exact_mod_cast hC ω hacc
  calc pooled n (fun t => D.size t ω) (fun t => D.draws t ω) δ = D.pooled δ ω := rfl
    _ < D.wrong ω := hw'.trans_le hc
    _ = ∑ t, ((D.size t ω - D.good t ω : ℕ) : ℝ) := by rw [Session.wrong, Nat.cast_sum]

end Draw


/-- **The openings' pooled bound** (`Accounting.Guarantees.Remote.PooledDraw`). -/
theorem PooledDraw : Accounting.Guarantees.Remote.PooledDraw := by
  intro n m k X _ D T δ hδ hL hR hC
  exact Draw.pooledDraw D T hδ hL hR hC

/-- **Live coins, derived** (`Accounting.Guarantees.Remote.LiveCoinsDrawn`). -/
theorem LiveCoinsDrawn : Accounting.Guarantees.Remote.LiveCoinsDrawn := by
  intro n m k X _ D T W δ hδ hL hR hC hW
  exact holdsMono (fun ω ha h => h.trans (hW ω ha)) (Draw.pooledDraw D T hδ hL hR hC)

/-- **The snapshot held at the record's pooled bound** (`Accounting.Guarantees.Remote.HeldDrawn`). -/
theorem HeldDrawn : Accounting.Guarantees.Remote.HeldDrawn := by
  intro n m k X _ D T S s₀ e s E β Br Bd δs δc hE hs hβ hδ hI hsq hd hS hr hc hk hL hR hC
  refine holdsMono (fun ω ha h => ?_) (holdsAnd hk (Draw.pooledDraw D T hδ hL hR hC))
  obtain ⟨hm, h0, hP⟩ := hc ω ha
  exact held_of ω hE hs hβ hI hsq hd hS hr hm h0 hP h.1 h.2

/-- **Held over time with the links charged, at the record's pooled bound**
(`Accounting.Guarantees.Remote.HeldOverTimeLinksDrawn`). -/
theorem HeldOverTimeLinksDrawn : Accounting.Guarantees.Remote.HeldOverTimeLinksDrawn := by
  intro n m k X _ D T K S s E β βl Br Bd Bl δs δc hE hs hβ hβl hδ hl hsq hd hS hr hp hcov hk hL hR hC
  refine holdsMono (fun ω ha h => ?_) (holdsAnd hk (Draw.pooledDraw D T hδ hL hR hC))
  exact heldOverLinks_of K ω hE hs hβ hβl hl hsq hd hS hr hp hcov ha h.1 h.2

/-- **Held over time at the record's pooled bound, under Identity** (`Accounting.Guarantees.Remote.HeldOverTimeDrawn`):
the link-charged form at links of rate 0 and burst 0. -/
theorem HeldOverTimeDrawn : Accounting.Guarantees.Remote.HeldOverTimeDrawn := by
  intro n m k X _ D T K S s E β Br Bd δs δc hE hs hβ hδ hI hsq hd hS hr hp hcov hk hL hR hC
  simpa only [add_zero] using HeldOverTimeLinksDrawn n m k X D T K S s E β 0 Br Bd 0 δs δc hE hs hβ le_rfl hδ
    (links_of_identity hI) hsq hd hS hr hp hcov hk hL hR hC

end Accounting.SecurityProofs.Remote
