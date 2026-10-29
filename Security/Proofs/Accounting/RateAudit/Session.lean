import Mathlib
import Definitions.Accounting.Guarantees.RateAuditHorizon
import Definitions.Accounting.Guarantees.RateAuditNamed
import Proofs.Accounting.RateAudit.Horizon
import Proofs.Accounting.RateAudit.Named

/-!
# The pooled draw: windows and sessions, the proof

**A window.** The window's draw of `d` openings is uniform over its `P·N` receipts and independent of its passes'
coins: at each store and each value of those coins, the window's correct receipts are one fixed set, and the draw lands
in it with probability `(#correct/(P·N))^d` (`Horizon.pr_sample'`, `Chain.pr_draw_le`).

**The monitoring period.** A session's windows are causal rounds over `M = P·N` items (`Rate.Causal`): window `w`'s
correct receipts read the store, the earlier windows' coins, their draws among them, and its own passes' coins, never
its own draw. `HorizonSampling` bounds the session's shortfall over its windows (`Session.pr_draws_le`).

**The floor in every window.** Keep the first window whose correct receipts number at most `K_w` as it is and count
every other window's receipts all good: still causal, since whether window `w` is the first short one reads only the
windows up to `w`, and the good sets then hold at most `(n − 1)·M + K_w` items in all. `HorizonSampling` bounds it,
with no union over the windows (`Horizon.pr_floor_le`).

**Paid.** A session with more than `K` correct receipts in all, or more than `K_w` in every window, whose reads fall
short of those receipts' words leaves some window short (`Session.pr_unpaid_le`), and that window some pass, which
`RatePaidNamedRO` bounds at the devices the earlier coins pick (`Session.pr_window_le`, then `Chain.pr_unpaid_le`).
-/

namespace Accounting.Rate

open Pous (Bits pr)
open scoped ENNReal
open Bandwidth (pr_mono pr_or_le pr_exists_le)

namespace Horizon

/-- `d` uniform items of a finite type all land in a set `G`: `(|G|/|α|)^d`. -/
theorem pr_sample' {α : Type} [Fintype α] (G : Finset α) (d : ℕ) :
    pr (fun I : Fin d → α => ∀ t, I t ∈ G) = ((G.card : ℝ≥0∞) / Fintype.card α) ^ d := by
  classical
  unfold pr
  have hG : Finset.univ.filter (fun I : Fin d → α => ∀ t, I t ∈ G) = Fintype.piFinset (fun _ => G) := by
    ext I
    simp [Fintype.mem_piFinset]
  rw [Finset.filter_congr_decidable, hG, Fintype.card_piFinset]
  simp [div_eq_mul_inv, mul_pow, ENNReal.inv_pow]

/-- **The floor in every round**: every round's sample of `d` lands in its good set while some round's good set holds
at most `k` of the `M` items, at most `(((n − 1)·M + k)/(n·M))^(d·n)`. -/
theorem pr_floor_le (n M k d : ℕ) (X : Fin n → Type) [∀ i, Fintype (X i)]
    (G : (i : Fin n) → ((j : Fin n) → X j × (Fin d → Fin M)) → Finset (Fin M)) (hG : Causal (s := fun _ => d) G) :
    pr (fun ω : (j : Fin n) → X j × (Fin d → Fin M) => (∀ i t, (ω i).2 t ∈ G i ω) ∧ ∃ i, (G i ω).card ≤ k)
      ≤ ((((n - 1) * M + k : ℕ) : ℝ≥0∞) / (n * M)) ^ (d * n) := by
  classical
  let G' : (i : Fin n) → ((j : Fin n) → X j × (Fin d → Fin M)) → Finset (Fin M) := fun i ω =>
    if (G i ω).card ≤ k ∧ ∀ u < i, k < (G u ω).card then G i ω else Finset.univ
  have hG' : Causal (s := fun _ => d) G' := by
    intro i ω ω' hlt hi
    have hu : ∀ u, u < i → G u ω = G u ω' := fun u hu =>
      hG u ω ω' (fun j hj => hlt j (hj.trans hu)) (congrArg Prod.fst (hlt u hu))
    have hc : ((G i ω).card ≤ k ∧ ∀ u < i, k < (G u ω).card) ↔
        ((G i ω').card ≤ k ∧ ∀ u < i, k < (G u ω').card) := by
      rw [hG i ω ω' hlt hi]
      exact and_congr_right fun _ => forall₂_congr fun u hu' => by rw [hu u hu']
    simp only [G']
    by_cases h : (G i ω).card ≤ k ∧ ∀ u < i, k < (G u ω).card
    · rw [ite_eq_left h, ite_eq_left (hc.1 h), hG i ω ω' hlt hi]
    · rw [ite_eq_right h, ite_eq_right (mt hc.2 h)]
  refine (pr_mono (fun ω hω => ?_)).trans
    ((pr_causal_le' n M (d * ((n - 1) * M + k)) X (fun _ => d) G' hG').trans (le_of_eq ?_))
  · obtain ⟨hall, i₀, hi₀⟩ := hω
    refine ⟨fun i t => ?_, ?_⟩
    · simp only [G']
      split_ifs
      · exact hall i t
      · exact Finset.mem_univ _
    obtain ⟨w, hws, hwmin⟩ := Finset.exists_min_image (Finset.univ.filter (fun i => (G i ω).card ≤ k)) id
      ⟨i₀, Finset.mem_filter.2 ⟨Finset.mem_univ _, hi₀⟩⟩
    have hcond : (G w ω).card ≤ k ∧ ∀ u < w, k < (G u ω).card := by
      refine ⟨(Finset.mem_filter.1 hws).2, fun u hu => ?_⟩
      by_contra hk
      exact absurd (hwmin u (Finset.mem_filter.2 ⟨Finset.mem_univ _, not_lt.1 hk⟩)) (not_le.2 hu)
    have hw : (G' w ω).card ≤ k := by
      simp only [G', ite_eq_left hcond]
      exact hcond.1
    have h2 := Finset.sum_le_card_nsmul (Finset.univ.erase w) (fun i => (G' i ω).card) M
      (fun i _ => (Finset.card_le_univ _).trans (by simp))
    rw [smul_eq_mul, Finset.card_erase_of_mem (Finset.mem_univ w), Finset.card_univ, Fintype.card_fin] at h2
    have hsum : ∑ i, (G' i ω).card ≤ (n - 1) * M + k := by
      rw [← Finset.add_sum_erase _ _ (Finset.mem_univ w)]
      omega
    show ∑ i, d * (G' i ω).card ≤ d * ((n - 1) * M + k)
    rw [← Finset.mul_sum]
    exact Nat.mul_le_mul_left d hsum
  · simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
    rcases Nat.eq_zero_or_pos d with rfl | hd
    · simp
    rw [mul_comm n d]
    congr 1
    have hd' : (d : ℝ≥0∞) ≠ 0 := by exact_mod_cast hd.ne'
    rw [Nat.cast_mul d, Nat.cast_mul d n, show (M : ℝ≥0∞) * (d * n) = d * (n * M) by ring]
    exact ENNReal.mul_div_mul_left _ _ hd' (ENNReal.natCast_ne_top d)

end Horizon

namespace Chain

variable {P N b h ρ S L q r s : ℕ}

/-- A window's correct receipts are its passes' correct receipts, counted together. -/
theorem card_correctAll (A : Chain P N b h ρ S L q r s) (E : Store N b) (ω : Fin P → PassCoins N b h ρ r s) :
    (A.correctAll E ω).card = ∑ j, ((A.dev j ω).correct E (ω j).1 (A.oracle j E ω)).card := by
  classical
  unfold correctAll
  rw [Finset.card_filter, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  rw [Finset.sum_boole]
  simp

/-- **A window's pooled draw**: every opening lands on a correct receipt while the window holds at most `K`,
at most `(K/(P·N))^d`. -/
theorem pr_draw_le (A : Chain P N b h ρ S L q r s) (d K : ℕ) :
    pr (fun x : Store N b × WindowCoins P N b h ρ r s d =>
        (∀ t, x.2.2 t ∈ A.correctAll x.1 x.2.1) ∧ (A.correctAll x.1 x.2.1).card ≤ K)
      ≤ ((K : ℝ≥0∞) / (P * N)) ^ d := by
  classical
  let e : Store N b × WindowCoins P N b h ρ r s d ≃
      (Fin d → Fin P × Fin N) × (Store N b × (Fin P → PassCoins N b h ρ r s)) :=
    (Equiv.prodAssoc _ _ _).symm.trans (Equiv.prodComm _ _)
  rw [pr_equiv e]
  refine pr_le_of_fiber _ _ (fun y => ?_)
  by_cases hK : (A.correctAll y.1 y.2).card ≤ K
  · refine (pr_mono (Q := fun I : Fin d → Fin P × Fin N => ∀ t, I t ∈ A.correctAll y.1 y.2)
      (fun I hI => hI.1)).trans (le_of_eq_of_le (Horizon.pr_sample' _ d) ?_)
    simp only [Fintype.card_prod, Fintype.card_fin, Nat.cast_mul]
    exact pow_le_pow_left₀ zero_le (ENNReal.div_le_div_right (by exact_mod_cast hK) _) _
  · refine (pr_mono (Q := fun _ : Fin d → Fin P × Fin N => False) (fun I hI => hK hI.2)).trans ?_
    rw [Horizon.pr_false]
    exact zero_le

/-- A window's unpaid words from a per-pass bound, with its draw beside: at most `P·ε`. -/
theorem pr_unpaid_draw_le {τ : ℕ} (A : Chain P N b h ρ S L q r s) (tier : ℕ → Fin τ) (slack d : ℕ) (ε : ℝ≥0∞)
    (hε : ∀ j η, pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        ∑ t, (A.device j η).tierReads tier t x.1 x.2.1 x.2.2 + (A.device j η).rebuilt x.1 x.2.1 x.2.2 * b + slack <
          ((A.device j η).correct x.1 x.2.1 x.2.2).card * b) ≤ ε) :
    pr (fun y : Store N b × WindowCoins P N b h ρ r s d =>
        A.spent tier y.1 y.2.1 + P * slack < (A.correctAll y.1 y.2.1).card * b) ≤ (P : ℝ≥0∞) * ε := by
  classical
  let e : Store N b × WindowCoins P N b h ρ r s d ≃
      (Store N b × (Fin P → PassCoins N b h ρ r s)) × (Fin d → Fin P × Fin N) :=
    (Equiv.prodAssoc _ _ _).symm
  rw [pr_equiv e]
  refine pr_le_of_fiber _ _ (fun D => ?_)
  refine (pr_mono (fun x hx => ?_)).trans (A.pr_unpaid_le tier slack ε hε)
  have hx' : A.spent tier x.1 x.2 + P * slack < (A.correctAll x.1 x.2).card * b := hx
  rw [card_correctAll, Finset.sum_mul] at hx'
  exact hx'

end Chain

namespace Session

variable {n P N b h ρ S L q r s d : ℕ}

/-- **Sequencing over windows**: an event of window `w`'s chained prover and its coins has at most its bound for the
chained provers the earlier windows' coins pick. -/
theorem pr_window_le (A : Session n P N b h ρ S L q r s d) (w : Fin n)
    (Q : Chain P N b h ρ S L q r s → Store N b × WindowCoins P N b h ρ r s d → Prop) (ε : ℝ≥0∞)
    (hQ : ∀ η, pr (Q (A.window w η)) ≤ ε) :
    pr (fun x : Store N b × (Fin n → WindowCoins P N b h ρ r s d) => Q (A.chain w x.2) (x.1, x.2 w)) ≤ ε := by
  classical
  let e : Store N b × (Fin n → WindowCoins P N b h ρ r s d) ≃
      (Store N b × WindowCoins P N b h ρ r s d) × ({v : Fin n // v ≠ w} → WindowCoins P N b h ρ r s d) :=
    ((Equiv.refl _).prodCongr (Equiv.funSplitAt w _)).trans (Equiv.prodAssoc _ _ _).symm
  rw [pr_equiv e]
  refine pr_le_of_fiber _ _ (fun ω' => ?_)
  let η : {v : Fin n // v < w} → WindowCoins P N b h ρ r s d := fun v => ω' ⟨v.1, v.2.ne⟩
  refine (pr_mono (fun y hy => ?_)).trans (hQ η)
  have hω : ((Equiv.funSplitAt w (WindowCoins P N b h ρ r s d)).symm (y.2, ω')) w = y.2 := by simp
  have hη : (fun v : {v : Fin n // v < w} =>
      ((Equiv.funSplitAt w (WindowCoins P N b h ρ r s d)).symm (y.2, ω')) v.1) = η := by
    funext v
    simp [η, v.2.ne]
  have hc : A.chain w ((Equiv.funSplitAt w (WindowCoins P N b h ρ r s d)).symm (y.2, ω')) = A.window w η := by
    simp only [Session.chain]
    rw [hη]
  simpa [e, hω, hc] using hy

/-- **A session's pooled draws**: every window's openings land on correct receipts while the session holds at most
`K` in all or some window at most `K_w`. -/
theorem pr_draws_le (A : Session n P N b h ρ S L q r s d) (K Kw : ℕ) :
    pr (fun x : Store N b × (Fin n → WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.chain w x.2).correctAll x.1 (x.2 w).1) ∧
          (∑ w, ((A.chain w x.2).correctAll x.1 (x.2 w).1).card ≤ K ∨
            ∃ w, ((A.chain w x.2).correctAll x.1 (x.2 w).1).card ≤ Kw))
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) +
        ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n) := by
  classical
  rw [pr_equiv (Equiv.prodComm _ _)]
  refine pr_le_of_fiber _ _ (fun E => ?_)
  let X : Fin n → Type := fun _ => Fin P → PassCoins N b h ρ r s
  let e : (Fin n → WindowCoins P N b h ρ r s d) ≃ ((w : Fin n) → X w × (Fin d → Fin (P * N))) :=
    Equiv.piCongrRight (fun _ => (Equiv.refl _).prodCongr (Equiv.arrowCongr (Equiv.refl _) finProdFinEquiv))
  rw [pr_equiv e]
  let G : (w : Fin n) → ((v : Fin n) → X v × (Fin d → Fin (P * N))) → Finset (Fin (P * N)) :=
    fun w ω => ((A.chain w (e.symm ω)).correctAll E (e.symm ω w).1).map finProdFinEquiv.toEmbedding
  have hG : Causal (s := fun _ => d) G := by
    intro w ω ω' hlt hw
    have hη : (fun v : {v : Fin n // v < w} => e.symm ω v.1) = (fun v => e.symm ω' v.1) :=
      funext fun v => by simp [e, hlt v.1 v.2]
    have h₁ : (e.symm ω w).1 = (e.symm ω' w).1 := by simp [e, hw]
    simp only [G, Session.chain]
    rw [hη, h₁]
  have hmem : ∀ ω w t, (e.symm ω w).2 t ∈ (A.chain w (e.symm ω)).correctAll E (e.symm ω w).1 →
      (ω w).2 t ∈ G w ω := by
    intro ω w t h
    simp only [G, Finset.mem_map_equiv]
    exact h
  have hcard : ∀ ω w, (G w ω).card = ((A.chain w (e.symm ω)).correctAll E (e.symm ω w).1).card :=
    fun ω w => Finset.card_map _
  refine (pr_mono (fun ω hω => ?_)).trans ((pr_or_le _ _).trans (add_le_add
    ((Horizon.pr_causal_le' n (P * N) (d * K) X (fun _ => d) G hG).trans (le_of_eq ?_))
    ((Horizon.pr_floor_le n (P * N) Kw d X G hG).trans (le_of_eq ?_))))
  · obtain ⟨hall, hK | ⟨w, hw⟩⟩ := hω
    · refine Or.inl ⟨fun w t => hmem ω w t (hall w t), ?_⟩
      show ∑ i, d * (G i ω).card ≤ d * K
      simp only [hcard]
      rw [← Finset.mul_sum]
      exact Nat.mul_le_mul_left d hK
    · exact Or.inr ⟨fun w t => hmem ω w t (hall w t), w, (hcard ω w).trans_le hw⟩
  · simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
    rcases Nat.eq_zero_or_pos d with rfl | hd
    · simp
    rw [mul_comm n d]
    congr 1
    have hd' : (d : ℝ≥0∞) ≠ 0 := by exact_mod_cast hd.ne'
    push_cast
    rw [show (P : ℝ≥0∞) * N * (d * n) = d * (n * P * N) by ring]
    exact ENNReal.mul_div_mul_left _ _ hd' (ENNReal.natCast_ne_top d)
  · simp only [Nat.cast_mul, mul_assoc]

/-- A session's unpaid words from a per-pass bound: the probability that some window's reads, rebuilt words and `P`
slacks fall short of its correct receipts' words is at most `n·P·ε`. -/
theorem pr_unpaid_le {τ : ℕ} (A : Session n P N b h ρ S L q r s d) (tier : ℕ → Fin τ) (slack : ℕ) (ε : ℝ≥0∞)
    (hε : ∀ w η j ι, pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        ∑ t, ((A.window w η).device j ι).tierReads tier t x.1 x.2.1 x.2.2 +
            ((A.window w η).device j ι).rebuilt x.1 x.2.1 x.2.2 * b + slack <
          (((A.window w η).device j ι).correct x.1 x.2.1 x.2.2).card * b) ≤ ε) :
    pr (fun x : Store N b × (Fin n → WindowCoins P N b h ρ r s d) =>
        ∃ w, (A.chain w x.2).spent tier x.1 (x.2 w).1 + P * slack <
          ((A.chain w x.2).correctAll x.1 (x.2 w).1).card * b)
      ≤ (n : ℝ≥0∞) * (P * ε) := by
  classical
  calc _ ≤ ∑ w, pr (fun x : Store N b × (Fin n → WindowCoins P N b h ρ r s d) =>
          (A.chain w x.2).spent tier x.1 (x.2 w).1 + P * slack <
            ((A.chain w x.2).correctAll x.1 (x.2 w).1).card * b) := pr_exists_le _
    _ ≤ ∑ _w : Fin n, (P : ℝ≥0∞) * ε := Finset.sum_le_sum (fun w _ => pr_window_le A w
          (fun C y => C.spent tier y.1 y.2.1 + P * slack < (C.correctAll y.1 y.2.1).card * b) _
          (fun η => (A.window w η).pr_unpaid_draw_le tier slack d ε (hε w η)))
    _ = (n : ℝ≥0∞) * (P * ε) := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end Session

end Accounting.Rate

namespace Accounting.SecurityProofs

open Accounting.Rate
open Pous (pr)
open scoped ENNReal
open Bandwidth (pr_mono pr_or_le)

/-- **A window's pooled draw** (`Accounting.Guarantees.RateWindowDraw`). -/
theorem RateWindowDraw : Accounting.Guarantees.RateWindowDraw := by
  intro P N b h ρ S L q r s d K A
  exact Chain.pr_draw_le A d K

/-- **A session's pooled draws: the monitoring period and every window** (`Accounting.Guarantees.RateSessionDraw`). -/
theorem RateSessionDraw : Accounting.Guarantees.RateSessionDraw := by
  intro n P N b h ρ S L q r s d K Kw A
  exact Session.pr_draws_le A K Kw

/-- **The route `Sampled` for a session, with named words, with side sources that ignore the oracle**
(`Accounting.Guarantees.RateSessionSampledNamedRO`). -/
theorem RateSessionSampledNamedRO : Accounting.Guarantees.RateSessionSampledNamedRO := by
  intro n P N b h ρ S L q r s d K Kw lam τ tier A hA
  have h₁ := Session.pr_draws_le A K Kw
  have h₂ := Session.pr_unpaid_le A tier (S + (L - ρ) + lam) _ (fun w η j ι =>
    (pr_mono (fun _ hx => by omega)).trans
      (RatePaidNamedRO N b h ρ S L q lam τ tier ((A.window w η).device j ι) (hA w η j ι)))
  refine ((pr_mono (fun x hx => ?_)).trans ((pr_or_le _ _).trans (add_le_add h₁ h₂))).trans
    (le_of_eq (by
      rw [mul_assoc (n : ℝ≥0∞) (P : ℝ≥0∞) ((N : ℝ≥0∞) / 2 ^ h + (q.choose 2 : ℝ≥0∞) / 2 ^ h + (2 ^ N : ℝ≥0∞) / 2 ^ lam)]))
  obtain ⟨hall, hT | ⟨w, hw⟩⟩ := hx
  · by_cases hD : ∑ w, ((A.chain w x.2).correctAll x.1 (x.2 w).1).card ≤ K ∨
        ∃ w, ((A.chain w x.2).correctAll x.1 (x.2 w).1).card ≤ Kw
    · exact Or.inl ⟨hall, hD⟩
    refine Or.inr ?_
    by_contra hno
    push Not at hno hD
    have hle := Finset.sum_le_sum (fun w (_ : w ∈ Finset.univ) => hno w)
    rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul,
      ← Finset.sum_mul] at hle
    have hK : (K + 1) * b ≤ (∑ w, ((A.chain w x.2).correctAll x.1 (x.2 w).1).card) * b :=
      Nat.mul_le_mul_right _ hD.1
    omega
  · by_cases hD : ∑ w, ((A.chain w x.2).correctAll x.1 (x.2 w).1).card ≤ K ∨
        ∃ w, ((A.chain w x.2).correctAll x.1 (x.2 w).1).card ≤ Kw
    · exact Or.inl ⟨hall, hD⟩
    push Not at hD
    exact Or.inr ⟨w, hw.trans_le (Nat.mul_le_mul_right _ (hD.2 w))⟩

end Accounting.SecurityProofs
