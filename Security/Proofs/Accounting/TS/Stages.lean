import Definitions.Accounting.Guarantees.TS.Stages
import Proofs.Accounting.TS

/-!
# Sampled proofs under per-stage coins: proofs

The round's step (`Stages.budget_point`): for `s ≥ 2`, `p ∈ [1/e, 1)`, `L = ln(1/p)` and `0 ≤ y ≤ m ≤ 1`,
`e^{mL}·(1 − m + y)^s ≤ 1 − m + y/p`. With `t = m − y` and `λ = t/m`, Jensen on `x ↦ x^s` between `1` and `1 − m` gives
`(1 − t)^s ≤ (1 − λ) + λ·(1 − m)^s`; the chord of `exp` on `[0, L]` gives `e^{mL} ≤ 1 − m + m/p`; and
`e^{mL}·(1 − m)^s ≤ (1 − m)·e^{mL}(1 − m) ≤ 1 − m`, since `(1 − m)^s ≤ (1 − m)^2` and `1 − m ≤ e^{−m} ≤ e^{−mL}` (this is
where `s ≥ 2` and `L ≤ 1`, i.e. `p ≥ 1/e`, enter).

On coins `a` before the draws, `m = Pr[hit]` and `y = Pr[hit ∧ ok]` over one draw, and a round's `s` independent draws
all pass with probability `Pr[ok]^s ≤ (1 − m + y)^s` (`Stages.prob_pi_all`). A share `d ≤ r·m` gives
`e^{d/Λ} ≤ e^{mL}` at `Λ = r/L`, so `E[e^{d/Λ}·1{pass} | a] ≤ 1 − m + y/p` (`Stages.sampled_step`), and averaging over `a`,
`1 − E[m] + E[y]/p ≤ 1` exactly when the draws are lucky at `p` (`Stages.sampled_budget`).

Under per-stage coins the draws' luck is the per-unit luck averaged over the draw law: a unit's skip and check are its
own stage's (`Stages.skipped_of_le`, `Stages.passes_of_le`), so conditioning on the stages before it, `StageLuck` bounds
`Pr[skipped_u ∧ passes_u]` by `p·Pr[skipped_u]` (`Stages.unit_luck`); and a unit's work weight at most `r` times its
draw probability prices the share at `r` (`Stages.prob_comp`).
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.Assumptions.TS Accounting.Guarantees.TS Pouw.RelativeSecurity

namespace Stages

theorem budget_point {s : ℕ} (hs : 2 ≤ s) {p m y : ℝ} (hp0 : Real.exp (-1) ≤ p)
    (hy0 : 0 ≤ y) (hym : y ≤ m) (hm1 : m ≤ 1) :
    Real.exp (m * Real.log (1 / p)) * (1 - m + y) ^ s ≤ 1 - m + y / p := by
  have hp : 0 < p := (Real.exp_pos _).trans_le hp0
  set L := Real.log (1 / p) with hLdef
  have hL1 : L ≤ 1 := by
    rw [hLdef, one_div, Real.log_inv, neg_le]
    calc (-1 : ℝ) = Real.log (Real.exp (-1)) := (Real.log_exp _).symm
      _ ≤ Real.log p := Real.log_le_log (Real.exp_pos _) hp0
  have heL : Real.exp L = 1 / p := Real.exp_log (by positivity)
  have hm0 : 0 ≤ m := hy0.trans hym
  rcases hm0.eq_or_lt with hm | hm
  · subst hm
    have hy : y = 0 := le_antisymm hym hy0
    subst hy
    simp
  have hchord : Real.exp (m * L) ≤ 1 - m + m / p := by
    have h := (convexOn_exp).2 (Set.mem_univ 0) (Set.mem_univ L) (sub_nonneg.mpr hm1) hm.le
      (by ring : (1 - m) + m = 1)
    simp only [smul_eq_mul, mul_zero, zero_add, Real.exp_zero, mul_one, heL] at h
    calc Real.exp (m * L) ≤ 1 - m + m * (1 / p) := h
      _ = 1 - m + m / p := by ring
  have hend : Real.exp (m * L) * (1 - m) ^ s ≤ 1 - m := by
    have h1m : 0 ≤ 1 - m := sub_nonneg.mpr hm1
    have hpow : (1 - m) ^ s ≤ (1 - m) ^ 2 := pow_le_pow_of_le_one h1m (by linarith) hs
    have hexp : Real.exp (m * L) * (1 - m) ≤ 1 := by
      have := one_sub_mul_exp_le m
      have hle : Real.exp (m * L) ≤ Real.exp m := Real.exp_le_exp.mpr (by nlinarith)
      nlinarith [Real.exp_pos (m * L)]
    calc Real.exp (m * L) * (1 - m) ^ s ≤ Real.exp (m * L) * (1 - m) ^ 2 :=
          mul_le_mul_of_nonneg_left hpow (Real.exp_pos _).le
      _ = (Real.exp (m * L) * (1 - m)) * (1 - m) := by ring
      _ ≤ 1 * (1 - m) := mul_le_mul_of_nonneg_right hexp h1m
      _ = 1 - m := one_mul _
  set lam := (m - y) / m with hlam
  have hlam0 : 0 ≤ lam := div_nonneg (by linarith) hm.le
  have hlam1 : lam ≤ 1 := by rw [hlam, div_le_one hm]; linarith
  have hmid : 1 - m + y = (1 - lam) * 1 + lam * (1 - m) := by
    rw [hlam]; field_simp; ring
  have hjen : (1 - m + y) ^ s ≤ (1 - lam) * 1 ^ s + lam * (1 - m) ^ s := by
    have h := (convexOn_pow s).2 (Set.mem_Ici.mpr zero_le_one) (Set.mem_Ici.mpr (sub_nonneg.mpr hm1))
      (sub_nonneg.mpr hlam1) hlam0 (by ring : (1 - lam) + lam = 1)
    simp only [smul_eq_mul] at h
    rw [hmid]
    exact h
  have hy' : (1 - lam) * m = y := by rw [hlam]; field_simp; ring
  calc Real.exp (m * L) * (1 - m + y) ^ s
      ≤ Real.exp (m * L) * ((1 - lam) * 1 ^ s + lam * (1 - m) ^ s) :=
        mul_le_mul_of_nonneg_left hjen (Real.exp_pos _).le
    _ = (1 - lam) * Real.exp (m * L) + lam * (Real.exp (m * L) * (1 - m) ^ s) := by ring
    _ ≤ (1 - lam) * (1 - m + m / p) + lam * (1 - m) := by gcongr
    _ = 1 - m + (1 - lam) * m / p := by ring
    _ = 1 - m + y / p := by rw [hy']

/-- `s` independent draws all land in `q` with probability `Pr[q]^s`. -/
theorem prob_pi_all {D : Type} [Fintype D] (s : ℕ) (q : D → Prop) :
    prob (fun ds : Fin s → D => ∀ i, q (ds i)) = prob q ^ s := by
  classical
  unfold prob
  have key : ∀ (inst : DecidablePred fun ds : Fin s → D => ∀ i, q (ds i)) (inst' : DecidablePred q),
      (@Finset.filter _ (fun ds : Fin s → D => ∀ i, q (ds i)) inst Finset.univ).card =
        (@Finset.filter _ q inst' Finset.univ).card ^ s := by
    intro inst inst'
    have h : @Finset.filter _ (fun ds : Fin s → D => ∀ i, q (ds i)) inst Finset.univ =
        Fintype.piFinset fun _ => @Finset.filter _ q inst' Finset.univ := by
      ext ds; simp [Fintype.mem_piFinset]
    rw [h, Fintype.card_piFinset, Finset.prod_const, Finset.card_univ, Fintype.card_fin]
  rw [key, Fintype.card_fun, Fintype.card_fin, Nat.cast_pow, Nat.cast_pow, div_pow]

theorem sum_le_card_of_expect {Ω : Type} [Fintype Ω] (f : Ω → ℝ) (h : expect f ≤ 1) :
    ∑ ω, f ω ≤ Fintype.card Ω := by
  unfold expect at h
  rcases Nat.eq_zero_or_pos (Fintype.card Ω) with h0 | h0
  · have : IsEmpty Ω := Fintype.card_eq_zero_iff.mp h0
    simp
  · rwa [div_le_one (by exact_mod_cast h0)] at h

theorem prob_prod {A D : Type} [Fintype A] [Fintype D] (E : A × D → Prop) :
    prob E = expect fun a => prob fun t => E (a, t) := by
  classical
  rw [prob_eq_expect, expect_prod]
  congr 1
  funext a
  rw [prob_eq_expect]

theorem prob_prod_swap {A D : Type} [Fintype A] [Fintype D] (E : A × D → Prop) :
    prob E = expect fun t => prob fun a => E (a, t) := by
  classical
  rw [prob_eq_expect, expect_prod_swap]
  congr 1
  funext t
  rw [prob_eq_expect]

theorem expect_empty {Ω : Type} [Fintype Ω] [IsEmpty Ω] (f : Ω → ℝ) : expect f = 0 := by
  unfold expect
  simp

theorem prob_empty {Ω : Type} [Fintype Ω] [IsEmpty Ω] (q : Ω → Prop) : prob q = 0 := by
  unfold prob
  simp

theorem expect_lin {Ω : Type} [Fintype Ω] [Nonempty Ω] (f g : Ω → ℝ) (p : ℝ) :
    expect (fun a => 1 - f a + g a / p) = 1 - expect f + expect g / p := by
  unfold expect
  have hC : (Fintype.card Ω : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.sum_div, Finset.sum_const, Finset.card_univ,
    nsmul_eq_mul, mul_one]
  field_simp

/-- One path of coins before the draws: the drawn checks all pass with probability `Pr[ok]^s`, and the share is
priced at `r` by the draws' mass on saved work. -/
theorem sampled_step {D : Type} [Fintype D] [Nonempty D] {s : ℕ} (hs : 2 ≤ s) {p r : ℝ}
    (hp0 : Real.exp (-1) ≤ p) (hp1 : p < 1) (hr : 0 < r) (hit ok : D → Prop) (sh : ℝ)
    (hd : sh ≤ r * prob hit) :
    expect (fun ds : Fin s → D =>
      (by classical exact if (∀ i, ok (ds i)) then Real.exp (sh / (r / Real.log (1 / p))) else 0)) ≤
      1 - prob hit + prob (fun t => hit t ∧ ok t) / p := by
  classical
  have hp : 0 < p := (Real.exp_pos _).trans_le hp0
  have hL0 : 0 ≤ Real.log (1 / p) := Real.log_nonneg (by rw [le_div_iff₀ hp]; linarith)
  have hE : expect (fun ds : Fin s → D =>
      if (∀ i, ok (ds i)) then Real.exp (sh / (r / Real.log (1 / p))) else 0) =
      Real.exp (sh / (r / Real.log (1 / p))) * prob ok ^ s := by
    rw [← prob_pi_all, prob_eq_expect, ← expect_mul_left]
    congr 1; funext ds
    split_ifs <;> simp
  rw [hE]
  have hok : prob ok ≤ 1 - prob hit + prob (fun t => hit t ∧ ok t) := by
    have h1 := prob_add_not hit
    have h2 : prob ok ≤ prob (fun t => ¬hit t) + prob (fun t => hit t ∧ ok t) :=
      (prob_mono fun t ht => by by_cases h : hit t; exacts [Or.inr ⟨h, ht⟩, Or.inl h]).trans
        (prob_or_le _ _)
    linarith
  have hexp : Real.exp (sh / (r / Real.log (1 / p))) ≤ Real.exp (prob hit * Real.log (1 / p)) := by
    apply Real.exp_le_exp.mpr
    rw [div_div_eq_mul_div, div_le_iff₀ hr]
    have := mul_le_mul_of_nonneg_right hd hL0
    linarith
  calc Real.exp (sh / (r / Real.log (1 / p))) * prob ok ^ s
      ≤ Real.exp (prob hit * Real.log (1 / p)) * (1 - prob hit + prob (fun t => hit t ∧ ok t)) ^ s :=
        mul_le_mul hexp (pow_le_pow_left₀ (prob_nonneg ok) hok s) (pow_nonneg (prob_nonneg ok) s)
          (Real.exp_pos _).le
    _ ≤ 1 - prob hit + prob (fun t => hit t ∧ ok t) / p :=
        budget_point hs hp0 (prob_nonneg _) (prob_mono fun t ht => ht.1) (prob_le_one _)

/-- **A sampled round's budget**: `E[e^{d/Λ}·1{pass}] ≤ 1` at `Λ = r/ln(1/p)`, in `BudgetAt`'s form, for a check
`pass` that is every draw's `ok` and a share `d` fixed before the draws. -/
theorem sampled_budget {A D : Type} [Fintype A] [Fintype D] {s : ℕ} (hs : 2 ≤ s) {p r : ℝ}
    (hp0 : Real.exp (-1) ≤ p) (hp1 : p < 1) (hr : 0 < r) (hit ok : A → D → Prop) (d₀ : A → ℝ)
    (hd : ∀ a, d₀ a ≤ r * prob (fun t => hit a t))
    (hluck : prob (fun z : A × D => hit z.1 z.2 ∧ ok z.1 z.2) ≤ p * prob (fun z : A × D => hit z.1 z.2))
    (pass : A × (Fin s → D) → Prop) (hpass : ∀ ω, pass ω ↔ ∀ i, ok ω.1 (ω.2 i))
    (d : A × (Fin s → D) → ℝ) (hdd : ∀ ω, d ω = d₀ ω.1) :
    (∑ ω, (by classical exact if pass ω then Real.exp (d ω / (r / Real.log (1 / p))) else 0)) ≤
      (Fintype.card (A × (Fin s → D)) : ℝ) := by
  classical
  apply sum_le_card_of_expect
  rcases isEmpty_or_nonempty A with hA | hA
  · rw [expect_empty]; exact zero_le_one
  rcases isEmpty_or_nonempty D with hD | hD
  · have : IsEmpty (Fin s → D) := isEmpty_fun.mpr ⟨⟨⟨0, by omega⟩⟩, hD⟩
    rw [expect_empty]; exact zero_le_one
  have hp : 0 < p := (Real.exp_pos _).trans_le hp0
  rw [expect_prod]
  refine (expect_mono (g := fun a => 1 - prob (hit a) + prob (fun t => hit a t ∧ ok a t) / p)
    fun a => ?_).trans ?_
  · refine le_of_eq_of_le ?_ (sampled_step hs hp0 hp1 hr (hit a) (ok a) (d₀ a) (hd a))
    congr 1
    funext ds
    by_cases h : ∀ i, ok a (ds i)
    · rw [ite_eq_left ((hpass (a, ds)).2 h), ite_eq_left h, hdd]
    · rw [ite_eq_right (fun h' => h ((hpass (a, ds)).1 h')), ite_eq_right h]
  rw [expect_lin, ← prob_prod (fun z : A × D => hit z.1 z.2),
    ← prob_prod (fun z : A × D => hit z.1 z.2 ∧ ok z.1 z.2)]
  have : (prob fun z : A × D => hit z.1 z.2 ∧ ok z.1 z.2) / p ≤ prob fun z : A × D => hit z.1 z.2 := by
    rw [div_le_iff₀ hp]; linarith
  linarith

/-- The probability that a ticket names a unit with `P` is the sum of those units' draw probabilities. -/
theorem prob_comp {D V : Type} [Fintype D] [Fintype V] (tk : D → V) (P : V → Prop) :
    prob (fun t => P (tk t)) = ∑ u, (by classical exact if P u then prob (fun t => tk t = u) else 0) := by
  classical
  rw [prob_eq_expect]
  have h : (fun t => if P (tk t) then (1 : ℝ) else 0) =
      fun t => ∑ u, if tk t = u then (if P u then (1 : ℝ) else 0) else 0 := by
    funext t
    rw [Finset.sum_ite_eq]
    simp
  rw [h, expect_sum]
  refine Finset.sum_congr rfl fun u _ => ?_
  by_cases hu : P u
  · simp only [hu, ite_true]
    rw [prob_eq_expect]
  · simp only [hu, ite_false, ite_self]
    unfold expect
    simp

theorem paths_nonempty (C : Type) [Nonempty C] : ∀ m, Nonempty (Paths C m)
  | 0 => ⟨()⟩
  | m + 1 => by
    have := paths_nonempty C m
    exact inferInstanceAs (Nonempty (C × Paths C m))

variable {C V : Type}

theorem skipped_of_le (stg : V → ℕ) : ∀ {m : ℕ} (T : CoinStages C V m) (x : Paths C m) (u : V), m ≤ stg u →
    ¬ T.skipped stg x u
  | _, .done, _, _, _ => fun h => h
  | _, .stage (m := m) _ _ next, x, u, h => by
    simp only [CoinStages.skipped]
    rintro (⟨h1, -⟩ | h2)
    · omega
    · exact skipped_of_le stg (next x.1) x.2 u (by omega) h2

theorem passes_of_le (stg : V → ℕ) : ∀ {m : ℕ} (T : CoinStages C V m) (x : Paths C m) (u : V), m ≤ stg u →
    T.passes stg x u
  | _, .done, _, _, _ => trivial
  | _, .stage (m := m) _ _ next, x, u, h => by
    simp only [CoinStages.passes]
    exact ⟨fun h1 => absurd h1 (by omega), passes_of_le stg (next x.1) x.2 u (by omega)⟩

/-- **The per-unit luck along the round**: under (L) per unit at its stage's coin, a unit skipped along the round's
coins passes with probability at most `p` times that of its being skipped. -/
theorem unit_luck [Fintype C] {p : ℝ} (hp : 0 ≤ p) (stg : V → ℕ) : ∀ {m : ℕ} (T : CoinStages C V m),
    T.LuckAt p stg → ∀ u, prob (fun x : Paths C m => T.skipped stg x u ∧ T.passes stg x u) ≤
      p * prob (fun x : Paths C m => T.skipped stg x u)
  | _, .done, _, u => by
    have h : prob (fun x : Paths C 0 => (CoinStages.done : CoinStages C V 0).skipped stg x u ∧
        (CoinStages.done : CoinStages C V 0).passes stg x u) = 0 := by
      refine le_antisymm ((prob_mono fun x hx => hx.1).trans (Rounds.prob_false.le)) (prob_nonneg _)
    rw [h]
    exact mul_nonneg hp (prob_nonneg _)
  | _, .stage (m := m) skip ok next, hL, u => by
    obtain ⟨hL, hnext⟩ := hL
    rcases isEmpty_or_nonempty C with hC | hC
    · rw [prob_empty, prob_empty, mul_zero]
    have := paths_nonempty C m
    by_cases hu : stg u = m
    · have e1 : prob (fun x : Paths C (m + 1) => (CoinStages.stage skip ok next).skipped stg x u ∧
            (CoinStages.stage skip ok next).passes stg x u) = prob (fun c => skip c u ∧ ok c u) := by
        rw [← prob_fst (β := Paths C m) (fun c => skip c u ∧ ok c u)]
        congr 1
        funext x
        simp only [CoinStages.skipped, CoinStages.passes, hu, true_and, forall_const]
        have hs := skipped_of_le stg (next x.1) x.2 u hu.ge
        have hp' := passes_of_le stg (next x.1) x.2 u hu.ge
        apply propext
        constructor
        · rintro ⟨h1 | h1, h2, -⟩
          · exact ⟨h1, h2⟩
          · exact absurd h1 hs
        · rintro ⟨h1, h2⟩
          exact ⟨Or.inl h1, h2, hp'⟩
      have e2 : prob (fun x : Paths C (m + 1) => (CoinStages.stage skip ok next).skipped stg x u) =
          prob (fun c => skip c u) := by
        rw [← prob_fst (β := Paths C m) (fun c => skip c u)]
        congr 1
        funext x
        simp only [CoinStages.skipped, hu, true_and]
        have hs := skipped_of_le stg (next x.1) x.2 u hu.ge
        apply propext
        exact ⟨fun h => h.resolve_right hs, Or.inl⟩
      rw [e1, e2]
      exact hL u hu
    · rw [prob_prod (fun x : C × Paths C m => (CoinStages.stage skip ok next).skipped stg x u ∧
          (CoinStages.stage skip ok next).passes stg x u),
        prob_prod (fun x : C × Paths C m => (CoinStages.stage skip ok next).skipped stg x u),
        ← expect_mul_left]
      refine expect_mono fun c => ?_
      simp only [CoinStages.skipped, CoinStages.passes, hu, false_and, false_or, false_implies, true_and]
      exact unit_luck hp stg (next c) (hnext c) u

/-- A round's share under per-stage coins is priced at `r` by the draws' mass on skipped units. -/
theorem share_le [Fintype V] {D : Type} [Fintype D] (stg : V → ℕ) (tk : D → V) (w : V → ℝ) {r : ℝ}
    (hw : ∀ u, w u ≤ r * prob (fun t => tk t = u)) {M : ℕ} (T : CoinStages C V M) (x : Paths C M) :
    (∑ u, ((by classical exact if T.skipped stg x u then w u else 0) : ℝ)) ≤
      r * prob (fun t => T.skipped stg x (tk t)) := by
  classical
  rw [prob_comp tk (fun u => T.skipped stg x u), Finset.mul_sum]
  refine Finset.sum_le_sum fun u _ => ?_
  split_ifs
  · exact hw u
  · simp

theorem share_nonneg [Fintype V] (stg : V → ℕ) (w : V → ℝ) (hw0 : ∀ u, 0 ≤ w u) {M : ℕ} (T : CoinStages C V M)
    (x : Paths C M) : 0 ≤ ∑ u, ((by classical exact if T.skipped stg x u then w u else 0) : ℝ) := by
  classical
  refine Finset.sum_nonneg fun u _ => ?_
  split_ifs
  · exact hw0 u
  · exact le_rfl

end Stages

open Stages in
theorem SampledBudget : Accounting.Guarantees.TS.SampledBudget := by
  intro A D _ _ s n W p r hs hp0 hp1 hr hP hL
  induction W with
  | done => trivial
  | round R next ih =>
    obtain ⟨hPR, hPn⟩ := hP
    obtain ⟨hLR, hLn⟩ := hL
    refine ⟨fun a => (hPR a.1).1, ?_, fun a => ih a (hPn a) (hLn a)⟩
    exact sampled_budget hs hp0 hp1 hr R.hit R.ok R.share (fun a => (hPR a).2) hLR
      (fun a => ∀ i, R.ok a.1 (a.2 i)) (fun _ => Iff.rfl) (fun a => R.share a.1) (fun _ => rfl)

open Stages in
theorem StageBudget : Accounting.Guarantees.TS.StageBudget := by
  intro C V D _ _ _ M s n W stg tk w p r hs hp0 hp1 hr hw0 hw hL
  have hp : 0 ≤ p := ((Real.exp_pos _).trans_le hp0).le
  induction W with
  | done => trivial
  | round T next ih =>
    obtain ⟨hLT, hLn⟩ := hL
    have hluck : prob (fun z : Paths C M × D => T.skipped stg z.1 (tk z.2) ∧ T.passes stg z.1 (tk z.2)) ≤
        p * prob (fun z : Paths C M × D => T.skipped stg z.1 (tk z.2)) := by
      rw [prob_prod_swap (fun z : Paths C M × D => T.skipped stg z.1 (tk z.2) ∧ T.passes stg z.1 (tk z.2)),
        prob_prod_swap (fun z : Paths C M × D => T.skipped stg z.1 (tk z.2)), ← expect_mul_left]
      exact expect_mono fun t => unit_luck hp stg T hLT (tk t)
    have key := sampled_budget (s := s) hs hp0 hp1 hr (fun a t => T.skipped stg a (tk t))
      (fun a t => T.passes stg a (tk t))
      (fun a => ∑ u, ((by classical exact if T.skipped stg a u then w u else 0) : ℝ))
      (fun a => share_le stg tk w hw T a) hluck (fun a => ∀ i, T.passes stg a.1 (tk (a.2 i))) (fun _ => Iff.rfl)
      (fun a => ∑ u, ((by classical exact if T.skipped stg a.1 u then w u else 0) : ℝ)) (fun _ => rfl)
    rw [StageWindow.game]
    exact ⟨fun a => share_nonneg stg w hw0 T a.1, key, fun a => ih a (hLn a)⟩

theorem CompilerStage : Accounting.Guarantees.TS.CompilerStage := by
  intro K R T L U F X _ _ _ _ _ _ _ _ _ _ C V D _ _ _ M s Z H S P W stg tk w lam p r δ δbind δzk nπ hcert hwf hCH hfH
    hS hs hp0 hp1 hr hδ hsplit hfc hww hsr hsu hns hfo hfi hG hw0 hw hL hPS
  have hp : 0 < p := (Real.exp_pos _).trans_le hp0
  have hL0 : 0 < Real.log (1 / p) := Real.log_pos (by rw [lt_div_iff₀ hp]; linarith)
  have hB : RoundGame.BudgetAt (r / Real.log (1 / p)) P.game := by
    rw [hG]
    exact StageBudget W stg tk w p r hs hp0 hp1 hr hw0 hw hL
  exact CompilerC Z H (Paths C M × (Fin s → D)) S P lam (r / Real.log (1 / p)) p δ δbind δzk nπ hcert hwf hCH hfH
    hS hp hp1 (div_pos hr hL0) hδ hsplit hfc hww hsr hsu hns hfo hfi hB hPS

theorem StageIdeal : Accounting.Guarantees.TS.StageIdeal := by
  intro K R T L U F X _ _ _ _ _ _ _ _ _ _ C V D _ _ _ M s Z H S P W stg tk w lam p δ δbind δzk nπ hcert hwf hCH hfH
    hS hs hp0 hp1 hδ hsplit hfc hww hsr hsu hns hfo hfi hG hw0 hw hL hPS
  have h := CompilerStage Z H S P W stg tk w lam p 1 δ δbind δzk nπ hcert hwf hCH hfH hS hs hp0 hp1 one_pos hδ
    hsplit hfc hww hsr hsu hns hfo hfi hG hw0 (fun u => by rw [one_mul]; exact hw u) hL hPS
  have h0 : Z.epsSample H.prog lam (1 / Real.log (1 / p)) p (δ - S * (δbind + nπ * δzk)) S = 0 := by
    unfold Model.epsSample
    rw [sub_self, zero_mul, zero_mul]
  rwa [h0, sub_zero] at h

end Accounting.SecurityProofs.TS
