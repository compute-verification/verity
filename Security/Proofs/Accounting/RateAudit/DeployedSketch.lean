import Mathlib
import Definitions.Accounting.Guarantees.RateAuditDeployed
import Proofs.Accounting.RateAudit.Deployed
import Proofs.Accounting.RateAudit.SketchRank

/-!
# The deployed verbatim-sketch session

For fixed full-rank signs, one pass's correct receipts are paid for except when it guessed an unqueried tag or made a
correct digest query without obtaining a sector. `card_guess_le` prices the former by `N/2^h`;
`card_bad_le` and `SketchRankFiber` price the latter by `q/2^(8r)`. A union over a pass's ground candidates and passes,
then the existing causal pooled-draw argument over windows, gives `RemoteSessionConservationSketch`.
-/

namespace Accounting.RateSketch

open Pous (pr)
open scoped ENNReal
open Bandwidth (pr_exists_le pr_le_of_card pr_mono pr_or_le)

namespace Device

variable {N m ρ h S L q ks : ℕ}

/-- Outside the low-rank event, a verbatim pass's correct receipts are paid for by its held sectors, memory reads,
side-link sectors and rebuilt sectors. -/
theorem pr_unpaid_rank_le
    (A : Device N m ρ h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)))
    (σ : SketchSigns ks m) (r X : ℕ)
    (hr32 : r ≤ 32)
    (hr : ∀ u, r ≤ ((Accounting.Tag.signMatrix (σ.block u)).map (Int.cast : ℤ → ℚ)).rank)
    (hX : ∀ O D, (A.prog σ D).rebuilds O D * 256 ≤ X) :
    pr (fun x : Data N m ρ Sector × Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32)) =>
      A.footprint σ x.2 x.1 + (S + (L - ρ * 256) + X) <
        (A.correct sketch σ x.2 x.1).card * (m * 256))
      ≤ (N : ℝ≥0∞) / 2 ^ h + (q : ℝ≥0∞) / 2 ^ (8 * r) := by
  classical
  let g := 256 ^ (32 - r)
  have hfib : fib (sketch (k := ks) (m := m)) σ ≤ g :=
    Accounting.SecurityProofs.SketchRankFiber ks m r σ hr
  have hsplit : ∀ x : Data N m ρ Sector × Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32)),
      A.footprint σ x.2 x.1 + (S + (L - ρ * 256) + X) <
          (A.correct sketch σ x.2 x.1).card * (m * 256) →
        (∃ t, t ∈ A.correct sketch σ x.2 x.1 ∧ t ∉ queried A sketch σ x.2 x.1) ∨
          ∃ i : Fin q, Bad A sketch σ x.2 x.1 i g := by
    rintro ⟨D, O⟩ hunpaid
    dsimp only at hunpaid
    by_cases hall : ∀ t ∈ A.correct sketch σ O D, t ∈ queried A sketch σ O D
    swap
    · push Not at hall
      exact Or.inl hall
    by_cases hcomp : ∀ t ∈ A.correct sketch σ O D, ∀ p, relevant t p → p ∈ A.obtained σ O D
    swap
    · push Not at hcomp
      obtain ⟨t, ht, p, hrel, hp⟩ := hcomp
      have hqmem : (D.stream, t, sketch σ (D.tile t)) ∈ (A.prog σ D).queries O D := by
        simpa [queried] using hall t ht
      have hlt := List.idxOf_lt_length_of_mem hqmem
      have hget : ((A.prog σ D).queries O D)[
          ((A.prog σ D).queries O D).idxOf (D.stream, t, sketch σ (D.tile t))]? =
          some (D.stream, t, sketch σ (D.tile t)) := by
        rw [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
      refine Or.inr ⟨⟨_, hlt.trans_le (A.bounded σ _ O D).2⟩, hfib, ⟨p, ?_⟩, _, hget, rfl⟩
      simp only [unobtained, Finset.mem_filter, Finset.mem_univ, true_and]
      exact ⟨hp, _, hget, hrel⟩
    have hne : (A.correct sketch σ O D).Nonempty := by
      by_contra hn
      rw [Finset.not_nonempty_iff_eq_empty.mp hn, Finset.card_empty, zero_mul] at hunpaid
      omega
    obtain ⟨t₀, ht₀⟩ := hne
    have hsub :
        ((A.correct sketch σ O D) ×ˢ (Finset.univ : Finset (Fin m))).disjSum
            (Finset.univ : Finset (Fin ρ)) ⊆ A.obtained σ O D := by
      rintro (⟨t, u⟩ | i) hp'
      · simp only [Finset.inl_mem_disjSum, Finset.mem_product, Finset.mem_univ, and_true] at hp'
        exact hcomp t hp' _ rfl
      · exact hcomp t₀ ht₀ _ trivial
    have hc := Finset.card_le_card hsub
    rw [Finset.card_disjSum, Finset.card_product, Finset.card_univ, Fintype.card_fin, Finset.card_univ,
      Fintype.card_fin] at hc
    have hob := A.card_obtained σ O D
    have hrebuild := hX O D
    have hpay :
        (A.correct sketch σ O D).card * (m * 256) + ρ * 256 ≤
          S + A.footprint σ O D + L + X := by
      calc
        _ = ((A.correct sketch σ O D).card * m + ρ) * 256 := by ring
        _ ≤ (A.obtained σ O D).card * 256 := Nat.mul_le_mul_right 256 hc
        _ ≤ S + A.footprint σ O D + L + (A.prog σ D).rebuilds O D * 256 := hob
        _ ≤ S + A.footprint σ O D + L + X := by omega
    omega
  refine (pr_mono hsplit).trans ((pr_or_le _ _).trans (add_le_add ?_ ?_))
  · refine (pr_exists_le _).trans ?_
    have ht : ∀ t : Fin N,
        pr (fun x : Data N m ρ Sector × Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32)) =>
          t ∈ A.correct sketch σ x.2 x.1 ∧ t ∉ queried A sketch σ x.2 x.1)
          ≤ (1 : ℝ≥0∞) / 2 ^ h := by
      intro t
      rw [Rate.pr_equiv (Equiv.prodComm _ _)]
      change pr (fun x : Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32)) × Data N m ρ Sector =>
        t ∈ A.correct sketch σ x.1 x.2 ∧ t ∉ queried A sketch σ x.1 x.2) ≤ (1 : ℝ≥0∞) / 2 ^ h
      refine Rate.pr_le_of_fiber _ _ (fun D => ?_)
      change pr (fun O : Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32)) =>
        t ∈ A.correct sketch σ O D ∧ t ∉ queried A sketch σ O D) ≤ (1 : ℝ≥0∞) / 2 ^ h
      have hbound := Bandwidth.pr_le_of_card
        (Ω := Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32)))
        (fun O => t ∈ A.correct sketch σ O D ∧ t ∉ queried A sketch σ O D)
        (2 ^ h) 1 (by positivity)
        (by simpa [Device.correct, and_comm] using card_guess_le A sketch σ D t)
      push_cast at hbound
      simpa using hbound
    refine (Finset.sum_le_sum (fun t _ => ht t)).trans (le_of_eq ?_)
    simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, div_eq_mul_inv]
  · refine (pr_exists_le _).trans ?_
    have hi : ∀ i : Fin q,
        pr (fun x : Data N m ρ Sector × Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32)) =>
          Bad A sketch σ x.2 x.1 i g) ≤ (g : ℝ≥0∞) / 2 ^ 256 := by
      intro i
      refine Rate.pr_le_of_fiber _ _ (fun O => ?_)
      convert Bandwidth.pr_le_of_card (fun D : Data N m ρ Sector => Bad A sketch σ O D i g)
        (Fintype.card Sector) g (by simp [Sector])
        (card_bad_le A sketch σ O i g) using 1
      all_goals norm_num [Sector]
    refine (Finset.sum_le_sum (fun i _ => hi i)).trans ?_
    have hg : (g : ℝ≥0∞) = 2 ^ (256 - 8 * r) := by
      dsimp [g]
      push_cast
      rw [show (256 : ℝ≥0∞) = 2 ^ 8 by norm_num, ← pow_mul,
        show 8 * (32 - r) = 256 - 8 * r by omega]
    have hsplitPow : (2 : ℝ≥0∞) ^ 256 = 2 ^ (256 - 8 * r) * 2 ^ (8 * r) := by
      rw [← pow_add, Nat.sub_add_cancel (by omega)]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, hg, hsplitPow]
    have hc : (2 : ℝ≥0∞) ^ (256 - 8 * r) ≠ 0 := pow_ne_zero _ (by norm_num)
    have ht : (2 : ℝ≥0∞) ^ (256 - 8 * r) ≠ ⊤ := ENNReal.pow_ne_top ENNReal.ofNat_ne_top
    have hcancel :
        (2 : ℝ≥0∞) ^ (256 - 8 * r) /
            ((2 : ℝ≥0∞) ^ (256 - 8 * r) * 2 ^ (8 * r)) =
          1 / 2 ^ (8 * r) := by
      simpa using (ENNReal.mul_div_mul_left (c := (2 : ℝ≥0∞) ^ (256 - 8 * r))
        1 (2 ^ (8 * r)) hc ht)
    rw [hcancel]
    simp [ENNReal.div_eq_inv_mul, mul_comm]

end Device

namespace SketchDeployedWindow

variable {P N m ρ h S L q ks G : ℕ}

/-- Correct receipts counted pass by pass. -/
theorem card_correctAll (A : SketchDeployedWindow P N m ρ h S L q ks G) (E : SketchStore N m)
    (σ : SketchSigns ks m) (ω : Fin P → SketchPassCoins N ρ h ks G) :
    (A.correctAll E σ ω).card = ∑ j,
      let D := Data.ofParts E (ω j).1
      ((A.dev j E σ ω).correct sketch σ (A.oracle j E σ ω) D).card := by
  classical
  unfold correctAll
  rw [Finset.card_filter, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  rw [Finset.sum_boole]
  simp

/-- Candidate `c` of pass `j` is unpaid. -/
def UnpaidAt (A : SketchDeployedWindow P N m ρ h S L q ks G) (σ : SketchSigns ks m) (slack : ℕ)
    (j : Fin P) (c : Fin G) (x : SketchStore N m × (Fin P → SketchPassCoins N ρ h ks G)) : Prop :=
  let η : {i : Fin P // i < j} → SketchPassCoins N ρ h ks G := fun i => x.2 i.1
  let B := A.device j η c
  let D := Data.ofParts x.1 (x.2 j).1
  B.footprint σ ((x.2 j).2 c) D + slack < (B.correct sketch σ ((x.2 j).2 c) D).card * (m * 256)

/-- The pass candidate actually kept is unpaid. -/
def Unpaid (A : SketchDeployedWindow P N m ρ h S L q ks G) (σ : SketchSigns ks m) (slack : ℕ)
    (j : Fin P) (x : SketchStore N m × (Fin P → SketchPassCoins N ρ h ks G)) : Prop :=
  let D := Data.ofParts x.1 (x.2 j).1
  (A.dev j x.1 σ x.2).footprint σ (A.oracle j x.1 σ x.2) D + slack <
    ((A.dev j x.1 σ x.2).correct sketch σ (A.oracle j x.1 σ x.2) D).card * (m * 256)

/-- If a window's total reads plus one slack per pass do not pay for its correct receipts, some kept candidate is
unpaid. -/
theorem exists_unpaid (A : SketchDeployedWindow P N m ρ h S L q ks G) (σ : SketchSigns ks m) (slack : ℕ)
    (x : SketchStore N m × (Fin P → SketchPassCoins N ρ h ks G))
    (hx : A.reads x.1 σ x.2 + P * slack < (A.correctAll x.1 σ x.2).card * (m * 256)) :
    ∃ j c, UnpaidAt A σ slack j c x := by
  by_contra hno
  push Not at hno
  have hle := Finset.sum_le_sum (fun j (_ : j ∈ Finset.univ) =>
    (not_lt.1 (hno j (A.kept j x.1 σ x.2)) :
      let D := Data.ofParts x.1 (x.2 j).1
      ((A.dev j x.1 σ x.2).correct sketch σ (A.oracle j x.1 σ x.2) D).card * (m * 256) ≤
        (A.dev j x.1 σ x.2).footprint σ (A.oracle j x.1 σ x.2) D + slack))
  rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul] at hle
  rw [card_correctAll, Finset.sum_mul] at hx
  exact (not_lt_of_ge (by simpa [reads] using hle)) hx

/-- Sequencing: candidate `c` of pass `j` has the one-pass bound for every device selected by earlier pass coins. -/
theorem pr_unpaidAt_le (A : SketchDeployedWindow P N m ρ h S L q ks G) (σ : SketchSigns ks m) (slack : ℕ)
    (ε : ℝ≥0∞) (j : Fin P) (c : Fin G)
    (hε : ∀ η, pr (fun x : Data N m ρ Sector × Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32)) =>
      let B := A.device j η c
      B.footprint σ x.2 x.1 + slack < (B.correct sketch σ x.2 x.1).card * (m * 256)) ≤ ε) :
    pr (UnpaidAt A σ slack j c) ≤ ε := by
  classical
  let ePass : SketchStore N m × (Fin P → SketchPassCoins N ρ h ks G) ≃
      (SketchStore N m × SketchPassCoins N ρ h ks G) ×
        ({i : Fin P // i ≠ j} → SketchPassCoins N ρ h ks G) :=
    ((Equiv.refl _).prodCongr (Equiv.funSplitAt j _)).trans (Equiv.prodAssoc _ _ _).symm
  rw [Rate.pr_equiv ePass]
  refine Rate.pr_le_of_fiber _ _ (fun ω' => ?_)
  let η : {i : Fin P // i < j} → SketchPassCoins N ρ h ks G := fun i => ω' ⟨i.1, i.2.ne⟩
  let dataEquiv : Data N m ρ Sector ≃ SketchStore N m × SketchStream ρ :=
    Equiv.sumArrowEquivProdArrow _ _ _
  let eCand : SketchStore N m × SketchPassCoins N ρ h ks G ≃
      (Data N m ρ Sector × Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) ×
        ({z : Fin G // z ≠ c} → Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) :=
    (Equiv.prodAssoc _ _ _).symm |>.trans
      (dataEquiv.symm.prodCongr (Equiv.funSplitAt c _)) |>.trans (Equiv.prodAssoc _ _ _).symm
  letI : Fintype (Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) := inferInstance
  rw [Rate.pr_equiv eCand]
  refine Rate.pr_le_of_fiber _ _ (fun Os => ?_)
  refine (pr_mono (fun y hy => ?_)).trans (hε η)
  have hpass : ((Equiv.funSplitAt j (SketchPassCoins N ρ h ks G)).symm
      ((eCand.symm (y, Os)).2, ω')) j = (eCand.symm (y, Os)).2 := by simp
  have hη : (fun i : {i : Fin P // i < j} =>
      ((Equiv.funSplitAt j (SketchPassCoins N ρ h ks G)).symm ((eCand.symm (y, Os)).2, ω')) i.1) = η := by
    funext i
    simp [η, i.2.ne]
  have hηsplit :
      (fun i : {i : Fin P // i < j} =>
        if h : i.1 = j then (eCand.symm (y, Os)).2 else ω' ⟨i.1, h⟩) = η := by
    funext i
    simp [η, i.2.ne]
  have hdata :
      Data.ofParts (eCand.symm (y, Os)).1 (eCand.symm (y, Os)).2.1 = y.1 := by
    funext p
    rcases p with p | p <;> simp [eCand, dataEquiv, Data.ofParts]
  have horacle : (eCand.symm (y, Os)).2.2 c = y.2 := by
    simp [eCand]
  set_option maxRecDepth 2000 in
    simpa [ePass, UnpaidAt, hpass, hη, hηsplit, hdata, horacle] using hy

/-- At fixed signs whose sector blocks have rank at least `r`, grinding contributes one union over the `P·G`
candidate passes. -/
theorem pr_unpaid_good_le (A : SketchDeployedWindow P N m ρ h S L q ks G) (σ : SketchSigns ks m)
    (r X : ℕ) (hr32 : r ≤ 32)
    (hr : ∀ u, r ≤ ((Accounting.Tag.signMatrix (σ.block u)).map (Int.cast : ℤ → ℚ)).rank)
    (hX : ∀ j η c O D, ((A.device j η c).prog σ D).rebuilds O D * 256 ≤ X) :
    pr (fun x : SketchStore N m × (Fin P → SketchPassCoins N ρ h ks G) =>
      A.reads x.1 σ x.2 + P * (S + (L - ρ * 256) + X) <
        (A.correctAll x.1 σ x.2).card * (m * 256))
      ≤ (P : ℝ≥0∞) * (G * ((N : ℝ≥0∞) / 2 ^ h + (q : ℝ≥0∞) / 2 ^ (8 * r))) := by
  let ε := (N : ℝ≥0∞) / 2 ^ h + (q : ℝ≥0∞) / 2 ^ (8 * r)
  calc
    _ ≤ pr (fun x : SketchStore N m × (Fin P → SketchPassCoins N ρ h ks G) =>
        ∃ j c, UnpaidAt A σ (S + (L - ρ * 256) + X) j c x) :=
      pr_mono (exists_unpaid A σ _)
    _ ≤ ∑ j, pr (fun x : SketchStore N m × (Fin P → SketchPassCoins N ρ h ks G) =>
        ∃ c, UnpaidAt A σ (S + (L - ρ * 256) + X) j c x) := pr_exists_le _
    _ ≤ ∑ j, ∑ c, pr (UnpaidAt A σ (S + (L - ρ * 256) + X) j c) :=
      Finset.sum_le_sum fun _ _ => pr_exists_le _
    _ ≤ ∑ _j : Fin P, ∑ _c : Fin G, ε := Finset.sum_le_sum fun j _ =>
      Finset.sum_le_sum fun c _ => pr_unpaidAt_le A σ _ ε j c fun η =>
        Device.pr_unpaid_rank_le (A.device j η c) σ r X hr32 hr (hX j η c)
    _ = (P : ℝ≥0∞) * (G * ε) := by
      simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]

/-- One window's unpaid receipts: one rank-tail event, plus the candidate-pass query terms. -/
theorem pr_unpaid_le {d : ℕ} (A : SketchDeployedWindow P N m ρ h S L q ks G) (r X : ℕ) (εrank : ℝ≥0∞)
    (hr32 : r ≤ 32) (hRank : Accounting.Assumptions.SketchRank ks m r εrank)
    (hX : ∀ j η c σ O D, ((A.device j η c).prog σ D).rebuilds O D * 256 ≤ X) :
    pr (fun x : SketchStore N m × SketchWindowCoins P N m ρ h ks G d =>
      A.reads x.1 x.2.1 x.2.2.1 + P * (S + (L - ρ * 256) + X) <
        (A.correctAll x.1 x.2.1 x.2.2.1).card * (m * 256))
      ≤ εrank + (P : ℝ≥0∞) * (G * ((N : ℝ≥0∞) / 2 ^ h + (q : ℝ≥0∞) / 2 ^ (8 * r))) := by
  classical
  let Passes := Fin P → SketchPassCoins N ρ h ks G
  let Draw := Fin d → Fin P × Fin N
  let eDraw : SketchStore N m × (SketchSigns ks m × Passes × Draw) ≃
      (SketchStore N m × (SketchSigns ks m × Passes)) × Draw :=
    ((Equiv.refl _).prodCongr (Equiv.prodAssoc _ _ _).symm).trans
      (Equiv.prodAssoc _ _ _).symm
  rw [Rate.pr_equiv eDraw]
  refine Rate.pr_le_of_fiber _ _ (fun _draw => ?_)
  let badRank : SketchSigns ks m → Prop := fun σ =>
    ∃ u, ((Accounting.Tag.signMatrix (σ.block u)).map (Int.cast : ℤ → ℚ)).rank < r
  let unpaid : SketchStore N m × (SketchSigns ks m × Passes) → Prop := fun x =>
    A.reads x.1 x.2.1 x.2.2 + P * (S + (L - ρ * 256) + X) <
      (A.correctAll x.1 x.2.1 x.2.2).card * (m * 256)
  have hsubset : ∀ x, unpaid x → badRank x.2.1 ∨ (¬ badRank x.2.1 ∧ unpaid x) := fun x hx => by
    by_cases hb : badRank x.2.1
    · exact Or.inl hb
    · exact Or.inr ⟨hb, hx⟩
  refine (pr_mono hsubset).trans ((pr_or_le _ _).trans (add_le_add ?_ ?_))
  · let eRank : SketchStore N m × (SketchSigns ks m × Passes) ≃
        SketchSigns ks m × (SketchStore N m × Passes) :=
      (Equiv.prodAssoc _ _ _).symm |>.trans
        ((Equiv.prodComm _ _).prodCongr (Equiv.refl _)) |>.trans (Equiv.prodAssoc _ _ _)
    rw [Rate.pr_equiv eRank]
    exact Rate.pr_le_of_fiber _ _ fun _ => hRank
  · let eGood : SketchStore N m × (SketchSigns ks m × Passes) ≃
        (SketchStore N m × Passes) × SketchSigns ks m :=
      ((Equiv.refl _).prodCongr (Equiv.prodComm _ _)).trans (Equiv.prodAssoc _ _ _).symm
    rw [Rate.pr_equiv eGood]
    refine Rate.pr_le_of_fiber _ _ fun σ => ?_
    by_cases hr : ∀ u, r ≤ ((Accounting.Tag.signMatrix (σ.block u)).map (Int.cast : ℤ → ℚ)).rank
    · exact (pr_mono fun _ hx => hx.2).trans
        (pr_unpaid_good_le A σ r X hr32 hr (hX := fun j η c O D => hX j η c σ O D))
    · push Not at hr
      calc
        pr (fun x : SketchStore N m × Passes => ¬badRank σ ∧ unpaid (x.1, σ, x.2)) ≤
            pr (fun _ : SketchStore N m × Passes => False) :=
          pr_mono fun _ hx => (hx.1 hr).elim
        _ ≤ (P : ℝ≥0∞) * (G * ((N : ℝ≥0∞) / 2 ^ h + (q : ℝ≥0∞) / 2 ^ (8 * r))) := by
          rw [Rate.Horizon.pr_false]
          exact bot_le

end SketchDeployedWindow

namespace SketchDeployedSession

variable {n P N m ρ h S L q ks G d : ℕ}

/-- Sequencing over windows. -/
theorem pr_window_le (A : SketchDeployedSession n P N m ρ h S L q ks G d) (w : Fin n)
    (Q : SketchDeployedWindow P N m ρ h S L q ks G →
      SketchStore N m × SketchWindowCoins P N m ρ h ks G d → Prop) (ε : ℝ≥0∞)
    (hQ : ∀ η, pr (Q (A.window w η)) ≤ ε) :
    pr (fun x : SketchStore N m × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      Q (A.win w x.2) (x.1, x.2 w)) ≤ ε := by
  classical
  let e : SketchStore N m × (Fin n → SketchWindowCoins P N m ρ h ks G d) ≃
      (SketchStore N m × SketchWindowCoins P N m ρ h ks G d) ×
        ({v : Fin n // v ≠ w} → SketchWindowCoins P N m ρ h ks G d) :=
    ((Equiv.refl _).prodCongr (Equiv.funSplitAt w _)).trans (Equiv.prodAssoc _ _ _).symm
  rw [Rate.pr_equiv e]
  refine Rate.pr_le_of_fiber _ _ (fun ω' => ?_)
  let η : {v : Fin n // v < w} → SketchWindowCoins P N m ρ h ks G d := fun v => ω' ⟨v.1, v.2.ne⟩
  refine (pr_mono (fun y hy => ?_)).trans (hQ η)
  have hω : ((Equiv.funSplitAt w (SketchWindowCoins P N m ρ h ks G d)).symm (y.2, ω')) w = y.2 := by simp
  have hη : (fun v : {v : Fin n // v < w} =>
      ((Equiv.funSplitAt w (SketchWindowCoins P N m ρ h ks G d)).symm (y.2, ω')) v.1) = η := by
    funext v
    simp [η, v.2.ne]
  have hc : A.win w ((Equiv.funSplitAt w (SketchWindowCoins P N m ρ h ks G d)).symm (y.2, ω')) =
      A.window w η := by
    simp only [SketchDeployedSession.win]
    rw [hη]
  simpa [e, hω, hc] using hy

/-- A session's pooled draws as a causal family over the monitoring period, on any store `store ω` drawn before the
session's coins. -/
theorem pr_draws_map_of_causal {Ω : Type} [Fintype Ω] (store : Ω → SketchStore N m)
    (A : SketchDeployedSession n P N m ρ h S L q ks G d)
    (Ev : SketchStore N m → (Fin n → SketchWindowCoins P N m ρ h ks G d) → Prop) (B : ℝ≥0∞)
    (hB : ∀ (X : Fin n → Type) [∀ w, Fintype (X w)]
      (Gs : (w : Fin n) → ((v : Fin n) → X v × (Fin d → Fin (P * N))) → Finset (Fin (P * N))),
      Rate.Causal (s := fun _ => d) Gs → ∀
        (f : ((v : Fin n) → X v × (Fin d → Fin (P * N))) → Fin n →
          SketchWindowCoins P N m ρ h ks G d) (E : SketchStore N m),
        (∀ ω w t, (f ω w).2.2 t ∈
            (A.win w (f ω)).correctAll E (f ω w).1 (f ω w).2.1 → (ω w).2 t ∈ Gs w ω) →
        (∀ ω w, (Gs w ω).card =
          ((A.win w (f ω)).correctAll E (f ω w).1 (f ω w).2.1).card) →
        pr (fun ω => Ev E (f ω)) ≤ B) :
    pr (fun x : Ω × (Fin n → SketchWindowCoins P N m ρ h ks G d) => Ev (store x.1) x.2) ≤ B := by
  classical
  rw [Rate.pr_equiv (Equiv.prodComm _ _)]
  refine Rate.pr_le_of_fiber _ _ (fun ω => ?_)
  show pr (fun y => Ev (store ω) y) ≤ B
  generalize store ω = E
  let X : Fin n → Type := fun _ => SketchSigns ks m × (Fin P → SketchPassCoins N ρ h ks G)
  let e : (Fin n → SketchWindowCoins P N m ρ h ks G d) ≃
      ((w : Fin n) → X w × (Fin d → Fin (P * N))) :=
    Equiv.piCongrRight fun _ =>
      (Equiv.prodAssoc _ _ _).symm |>.trans
        ((Equiv.refl _).prodCongr (Equiv.arrowCongr (Equiv.refl _) finProdFinEquiv))
  rw [Rate.pr_equiv e]
  let Gs : (w : Fin n) → ((v : Fin n) → X v × (Fin d → Fin (P * N))) → Finset (Fin (P * N)) :=
    fun w ω => ((A.win w (e.symm ω)).correctAll E (e.symm ω w).1 (e.symm ω w).2.1).map
      finProdFinEquiv.toEmbedding
  have hG : Rate.Causal (s := fun _ => d) Gs := by
    intro w ω ω' hlt hw
    have hη : (fun v : {v : Fin n // v < w} => e.symm ω v.1) =
        (fun v => e.symm ω' v.1) := funext fun v => by simp [e, hlt v.1 v.2]
    have hσ : (e.symm ω w).1 = (e.symm ω' w).1 := by simp [e, hw]
    have hp : (e.symm ω w).2.1 = (e.symm ω' w).2.1 := by simp [e, hw]
    simp only [Gs, SketchDeployedSession.win]
    rw [hη, hσ, hp]
  have hmem : ∀ ω w t, (e.symm ω w).2.2 t ∈
      (A.win w (e.symm ω)).correctAll E (e.symm ω w).1 (e.symm ω w).2.1 →
      (ω w).2 t ∈ Gs w ω := by
    intro ω w t ht
    simp only [Gs, Finset.mem_map_equiv]
    exact ht
  have hcard : ∀ ω w, (Gs w ω).card =
      ((A.win w (e.symm ω)).correctAll E (e.symm ω w).1 (e.symm ω w).2.1).card :=
    fun ω w => Finset.card_map _
  exact hB X Gs hG e.symm E hmem hcard

/-- A session's pooled draws as a causal family over the monitoring period. -/
theorem pr_draws_of_causal (A : SketchDeployedSession n P N m ρ h S L q ks G d)
    (Ev : SketchStore N m → (Fin n → SketchWindowCoins P N m ρ h ks G d) → Prop) (B : ℝ≥0∞)
    (hB : ∀ (X : Fin n → Type) [∀ w, Fintype (X w)]
      (Gs : (w : Fin n) → ((v : Fin n) → X v × (Fin d → Fin (P * N))) → Finset (Fin (P * N))),
      Rate.Causal (s := fun _ => d) Gs → ∀
        (f : ((v : Fin n) → X v × (Fin d → Fin (P * N))) → Fin n →
          SketchWindowCoins P N m ρ h ks G d) (E : SketchStore N m),
        (∀ ω w t, (f ω w).2.2 t ∈
            (A.win w (f ω)).correctAll E (f ω w).1 (f ω w).2.1 → (ω w).2 t ∈ Gs w ω) →
        (∀ ω w, (Gs w ω).card =
          ((A.win w (f ω)).correctAll E (f ω w).1 (f ω w).2.1).card) →
        pr (fun ω => Ev E (f ω)) ≤ B) :
    pr (fun x : SketchStore N m × (Fin n → SketchWindowCoins P N m ρ h ks G d) => Ev x.1 x.2) ≤ B :=
  pr_draws_map_of_causal id A Ev B hB

/-- The pooled draw over the whole monitoring period, on any store drawn before the session's coins. -/
theorem pr_draws_horizon_le_map {Ω : Type} [Fintype Ω] (store : Ω → SketchStore N m)
    (A : SketchDeployedSession n P N m ρ h S L q ks G d) (K : ℕ) :
    pr (fun x : Ω × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      (∀ w t, (x.2 w).2.2 t ∈ (A.win w x.2).correctAll (store x.1) (x.2 w).1 (x.2 w).2.1) ∧
        ∑ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1 (x.2 w).2.1).card ≤ K)
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) := by
  refine pr_draws_map_of_causal store A (fun E y =>
    (∀ w t, (y w).2.2 t ∈ (A.win w y).correctAll E (y w).1 (y w).2.1) ∧
      ∑ w, ((A.win w y).correctAll E (y w).1 (y w).2.1).card ≤ K) _
    (fun X _ Gs hG f E hmem hcard => ?_)
  refine (pr_mono (fun ω hω => ?_)).trans
    ((Rate.Horizon.pr_causal_le' n (P * N) (d * K) X (fun _ => d) Gs hG).trans (le_of_eq ?_))
  · obtain ⟨hall, hK⟩ := hω
    refine ⟨fun w t => hmem ω w t (hall w t), ?_⟩
    show ∑ i, d * (Gs i ω).card ≤ d * K
    simp only [hcard]
    rw [← Finset.mul_sum]
    exact Nat.mul_le_mul_left d hK
  · simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
    rcases Nat.eq_zero_or_pos d with rfl | hd
    · simp
    rw [mul_comm n d]
    congr 1
    have hd' : (d : ℝ≥0∞) ≠ 0 := by exact_mod_cast hd.ne'
    push_cast
    rw [show (P : ℝ≥0∞) * N * (d * n) = d * (n * P * N) by ring]
    exact ENNReal.mul_div_mul_left _ _ hd' (ENNReal.natCast_ne_top d)

/-- The pooled draw over the whole monitoring period. -/
theorem pr_draws_horizon_le (A : SketchDeployedSession n P N m ρ h S L q ks G d) (K : ℕ) :
    pr (fun x : SketchStore N m × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      (∀ w t, (x.2 w).2.2 t ∈ (A.win w x.2).correctAll x.1 (x.2 w).1 (x.2 w).2.1) ∧
        ∑ w, ((A.win w x.2).correctAll x.1 (x.2 w).1 (x.2 w).2.1).card ≤ K)
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) :=
  pr_draws_horizon_le_map id A K

/-- The pooled draw in every window, on any store drawn before the session's coins. -/
theorem pr_draws_window_le_map {Ω : Type} [Fintype Ω] (store : Ω → SketchStore N m)
    (A : SketchDeployedSession n P N m ρ h S L q ks G d) (Kw : ℕ) :
    pr (fun x : Ω × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      (∀ w t, (x.2 w).2.2 t ∈ (A.win w x.2).correctAll (store x.1) (x.2 w).1 (x.2 w).2.1) ∧
        ∃ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1 (x.2 w).2.1).card ≤ Kw)
      ≤ ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n) := by
  refine pr_draws_map_of_causal store A (fun E y =>
    (∀ w t, (y w).2.2 t ∈ (A.win w y).correctAll E (y w).1 (y w).2.1) ∧
      ∃ w, ((A.win w y).correctAll E (y w).1 (y w).2.1).card ≤ Kw) _
    (fun X _ Gs hG f E hmem hcard => ?_)
  refine (pr_mono (fun ω hω => ?_)).trans
    ((Rate.Horizon.pr_floor_le n (P * N) Kw d X Gs hG).trans (le_of_eq ?_))
  · obtain ⟨hall, w, hw⟩ := hω
    exact ⟨fun w t => hmem ω w t (hall w t), w, (hcard ω w).trans_le hw⟩
  · simp only [Nat.cast_mul, mul_assoc]

/-- The pooled draw in every window. -/
theorem pr_draws_window_le (A : SketchDeployedSession n P N m ρ h S L q ks G d) (Kw : ℕ) :
    pr (fun x : SketchStore N m × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      (∀ w t, (x.2 w).2.2 t ∈ (A.win w x.2).correctAll x.1 (x.2 w).1 (x.2 w).2.1) ∧
        ∃ w, ((A.win w x.2).correctAll x.1 (x.2 w).1 (x.2 w).2.1).card ≤ Kw)
      ≤ ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n) :=
  pr_draws_window_le_map id A Kw

/-- Both pooled-draw readings, on any store drawn before the session's coins. -/
theorem pr_draws_le_map {Ω : Type} [Fintype Ω] (store : Ω → SketchStore N m)
    (A : SketchDeployedSession n P N m ρ h S L q ks G d) (K Kw : ℕ) :
    pr (fun x : Ω × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      (∀ w t, (x.2 w).2.2 t ∈ (A.win w x.2).correctAll (store x.1) (x.2 w).1 (x.2 w).2.1) ∧
        (∑ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1 (x.2 w).2.1).card ≤ K ∨
          ∃ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1 (x.2 w).2.1).card ≤ Kw))
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) +
        ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n) :=
  (pr_mono fun _ h => h.2.imp (fun hK => ⟨h.1, hK⟩) (fun hw => ⟨h.1, hw⟩)).trans
    ((pr_or_le _ _).trans (add_le_add (pr_draws_horizon_le_map store A K)
      (pr_draws_window_le_map store A Kw)))

/-- Both pooled-draw readings. -/
theorem pr_draws_le (A : SketchDeployedSession n P N m ρ h S L q ks G d) (K Kw : ℕ) :
    pr (fun x : SketchStore N m × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      (∀ w t, (x.2 w).2.2 t ∈ (A.win w x.2).correctAll x.1 (x.2 w).1 (x.2 w).2.1) ∧
        (∑ w, ((A.win w x.2).correctAll x.1 (x.2 w).1 (x.2 w).2.1).card ≤ K ∨
          ∃ w, ((A.win w x.2).correctAll x.1 (x.2 w).1 (x.2 w).2.1).card ≤ Kw))
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) +
        ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n) :=
  pr_draws_le_map id A K Kw

/-- A session's unpaid windows from a per-window bound. -/
theorem pr_unpaid_le (A : SketchDeployedSession n P N m ρ h S L q ks G d) (slack : ℕ) (ε : ℝ≥0∞)
    (hε : ∀ w η, pr (fun y : SketchStore N m × SketchWindowCoins P N m ρ h ks G d =>
      (A.window w η).reads y.1 y.2.1 y.2.2.1 + P * slack <
        ((A.window w η).correctAll y.1 y.2.1 y.2.2.1).card * (m * 256)) ≤ ε) :
    pr (fun x : SketchStore N m × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      ∃ w, (A.win w x.2).reads x.1 (x.2 w).1 (x.2 w).2.1 + P * slack <
        ((A.win w x.2).correctAll x.1 (x.2 w).1 (x.2 w).2.1).card * (m * 256))
      ≤ (n : ℝ≥0∞) * ε := by
  calc
    _ ≤ ∑ w, pr (fun x : SketchStore N m × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
        (A.win w x.2).reads x.1 (x.2 w).1 (x.2 w).2.1 + P * slack <
          ((A.win w x.2).correctAll x.1 (x.2 w).1 (x.2 w).2.1).card * (m * 256)) := pr_exists_le _
    _ ≤ ∑ _w : Fin n, ε := Finset.sum_le_sum fun w _ =>
      pr_window_le A w (fun C y => C.reads y.1 y.2.1 y.2.2.1 + P * slack <
        (C.correctAll y.1 y.2.1 y.2.2.1).card * (m * 256)) ε (hε w)
    _ = (n : ℝ≥0∞) * ε := by
      simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]

/-- Session conservation in counts, from the pooled draws and an unpaid bound `ε_u` on some window's unpaid words, on
any store `store ω` drawn before the session's coins. -/
theorem pr_paid_of_unpaid_map {Ω : Type} [Fintype Ω] (store : Ω → SketchStore N m)
    (A : SketchDeployedSession n P N m ρ h S L q ks G d) (slack K Kw : ℕ) (εu : ℝ≥0∞)
    (hu : pr (fun x : Ω × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      ∃ w, (A.win w x.2).reads (store x.1) (x.2 w).1 (x.2 w).2.1 + P * slack <
        ((A.win w x.2).correctAll (store x.1) (x.2 w).1 (x.2 w).2.1).card * (m * 256)) ≤ εu) :
    pr (fun x : Ω × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      (∀ w t, (x.2 w).2.2 t ∈ (A.win w x.2).correctAll (store x.1) (x.2 w).1 (x.2 w).2.1) ∧
        (∑ w, (A.win w x.2).reads (store x.1) (x.2 w).1 (x.2 w).2.1 + n * (P * slack) <
            (K + 1) * (m * 256) ∨
          ∃ w, (A.win w x.2).reads (store x.1) (x.2 w).1 (x.2 w).2.1 + P * slack <
            (Kw + 1) * (m * 256)))
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) +
        ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n) + εu := by
  have h₁ := pr_draws_le_map store A K Kw
  refine (pr_mono (fun x hx => ?_)).trans ((pr_or_le _ _).trans (add_le_add h₁ hu))
  obtain ⟨hall, hT | ⟨w, hw⟩⟩ := hx
  · by_cases hD : ∑ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1 (x.2 w).2.1).card ≤ K ∨
        ∃ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1 (x.2 w).2.1).card ≤ Kw
    · exact Or.inl ⟨hall, hD⟩
    refine Or.inr ?_
    push Not at hD
    by_contra hno
    push Not at hno
    have hle := Finset.sum_le_sum fun w (_ : w ∈ Finset.univ) => hno w
    rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul,
      ← Finset.sum_mul] at hle
    have hK : (K + 1) * (m * 256) ≤
        (∑ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1 (x.2 w).2.1).card) * (m * 256) :=
      Nat.mul_le_mul_right _ hD.1
    omega
  · by_cases hD : ∑ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1 (x.2 w).2.1).card ≤ K ∨
        ∃ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1 (x.2 w).2.1).card ≤ Kw
    · exact Or.inl ⟨hall, hD⟩
    push Not at hD
    exact Or.inr ⟨w, hw.trans_le (Nat.mul_le_mul_right _ (hD.2 w))⟩

/-- Session conservation in counts, from the pooled draws and one per-window unpaid bound. -/
theorem pr_paid_of_window (A : SketchDeployedSession n P N m ρ h S L q ks G d) (slack K Kw : ℕ)
    (ε : ℝ≥0∞)
    (hε : ∀ w η, pr (fun y : SketchStore N m × SketchWindowCoins P N m ρ h ks G d =>
      (A.window w η).reads y.1 y.2.1 y.2.2.1 + P * slack <
        ((A.window w η).correctAll y.1 y.2.1 y.2.2.1).card * (m * 256)) ≤ ε) :
    pr (fun x : SketchStore N m × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      (∀ w t, (x.2 w).2.2 t ∈ (A.win w x.2).correctAll x.1 (x.2 w).1 (x.2 w).2.1) ∧
        (∑ w, (A.win w x.2).reads x.1 (x.2 w).1 (x.2 w).2.1 + n * (P * slack) <
            (K + 1) * (m * 256) ∨
          ∃ w, (A.win w x.2).reads x.1 (x.2 w).1 (x.2 w).2.1 + P * slack <
            (Kw + 1) * (m * 256)))
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) +
        ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n) +
          (n : ℝ≥0∞) * ε :=
  pr_paid_of_unpaid_map id A slack K Kw _ (pr_unpaid_le A slack ε hε)

end SketchDeployedSession

/-- The named non-sampling terms of a sketch pass sum to its slack in bits. -/
theorem gammaSketch_mul (N m ρ S L X : ℕ) (hN : 0 < N) (hm : 0 < m) :
    gammaSketch N m ρ S L X * (N * m * 256) = ((S + (L - ρ * 256) + X : ℕ) : ℝ) := by
  have hNm : (N : ℝ) * m * 256 ≠ 0 := by positivity
  unfold gammaSketch
  rw [div_mul_cancel₀ _ hNm]

end Accounting.RateSketch

namespace Accounting.SecurityProofs

open Accounting.Rate Accounting.RateSketch
open Pous (pr)
open scoped ENNReal
open Bandwidth (pr_mono)

/-- **The remote bandwidth audit with deployed tensor-sketch receipts, for the verbatim class**
(`Accounting.Guarantees.RemoteSessionConservationSketch`). -/
theorem RemoteSessionConservationSketch : Accounting.Guarantees.RemoteSessionConservationSketch := by
  intro n P N m ρ h S L q ks G d r X ε A dbar dw hn hP hN hm hr32 hdw hRank hX
  let slack := S + (L - ρ * 256) + X
  let term := (N : ℝ≥0∞) / 2 ^ h + (q : ℝ≥0∞) / 2 ^ (8 * r)
  have hX' : ∀ w η j ι c σ O D,
      (((A.window w η).device j ι c).prog σ D).rebuilds O D * 256 ≤ X := by
    intro w η j ι c σ O D
    let E : RateSketch.SketchStore N m := fun p => D (.inl p)
    let R : RateSketch.SketchStream ρ := fun i => D (.inr i)
    have hD : RateSketch.Data.ofParts E R = D := by
      funext p
      rcases p with p | p <;> rfl
    simpa [hD] using hX w η j ι c E σ R O
  have hε : ∀ w η, pr (fun y : SketchStore N m × SketchWindowCoins P N m ρ h ks G d =>
      (A.window w η).reads y.1 y.2.1 y.2.2.1 + P * slack <
        ((A.window w η).correctAll y.1 y.2.1 y.2.2.1).card * (m * 256)) ≤
      ε + (P : ℝ≥0∞) * (G * term) := fun w η =>
    SketchDeployedWindow.pr_unpaid_le (A.window w η) r X ε hr32 hRank (hX' w η)
  have main := SketchDeployedSession.pr_paid_of_window A slack
    ⌊(1 - dbar) * ((n : ℝ) * P * N)⌋₊ ⌊(1 - dw) * ((P : ℝ) * N)⌋₊
    (ε + (P : ℝ≥0∞) * (G * term)) hε
  have hγ := gammaSketch_mul N m ρ S L X hN hm
  have hγ' : gammaSketch N m ρ S L X * ((N : ℝ) * (m * 256)) = (slack : ℝ) := by
    simpa [slack, mul_assoc] using hγ
  have hb : (0 : ℝ) < m * 256 := by positivity
  refine (pr_mono (fun x hx => ?_)).trans (main.trans ?_)
  · obtain ⟨hall, hT | ⟨w, hw⟩⟩ := hx
    · refine ⟨hall, Or.inl ?_⟩
      have hc := Deployed.count_lt
        (R := ((∑ w, (A.win w x.2).reads x.1 (x.2 w).1 (x.2 w).2.1 : ℕ) : ℝ))
        (t := dbar) (g := gammaSketch N m ρ S L X) (c := (n : ℝ) * P)
        (Nn := N) (b := (m : ℝ) * 256) (sl := slack) hb hγ'
        (by simpa [mul_assoc] using hT)
      rw [← mul_assoc]
      exact_mod_cast hc
    · refine ⟨hall, Or.inr ⟨w, ?_⟩⟩
      have hc := Deployed.count_lt
        (R := (((A.win w x.2).reads x.1 (x.2 w).1 (x.2 w).2.1 : ℕ) : ℝ))
        (t := dw) (g := gammaSketch N m ρ S L X) (c := (P : ℝ))
        (Nn := N) (b := (m : ℝ) * 256) (sl := slack) hb hγ'
        (by simpa [mul_assoc] using hw)
      exact_mod_cast hc
  · have e1 : ((n : ℝ) * P * N) = ((n * P * N : ℕ) : ℝ) := by push_cast; ring
    have e2 : ((P : ℝ) * N) = ((P * N : ℕ) : ℝ) := by push_cast; ring
    rw [e1, e2]
    calc
      _ ≤ ENNReal.ofReal (1 - dbar) ^ (d * n) + ENNReal.ofReal (1 - dw / n) ^ (d * n) +
          (n : ℝ≥0∞) * (ε + (P : ℝ≥0∞) * (G * term)) := by
        refine add_le_add (add_le_add ?_ ?_) le_rfl
        · rw [show ((n : ℝ≥0∞) * P * N) = ((n * P * N : ℕ) : ℝ≥0∞) by push_cast; ring]
          exact Deployed.horizon_le dbar (n * P * N) (d * n) (by positivity)
        · rw [show ((n : ℝ≥0∞) * P * N) =
              (n : ℝ≥0∞) * ((P * N : ℕ) : ℝ≥0∞) by push_cast; ring]
          exact Deployed.window_le dw n (P * N) (d * n) hn (by positivity) hdw
      _ = ENNReal.ofReal (1 - dbar) ^ (d * n) + ENNReal.ofReal (1 - dw / n) ^ (d * n) +
          (n : ℝ≥0∞) * ε + (n * P * G : ℝ≥0∞) * term := by
        ring

/-- At the conservative deployed caps, the rank-query term is already `2^14` before multiplying by windows or
passes. This is why the 16-row worst-cap bound is vacuous. -/
theorem sketchQueryTermCaps :
    (2 ^ 55 : ℝ≥0∞) * ((2 ^ 55 : ℝ≥0∞) / 2 ^ 96) = 2 ^ 14 := by
  rw [← mul_div_assoc]
  rw [show (2 : ℝ≥0∞) ^ 55 * 2 ^ 55 = 2 ^ 96 * 2 ^ 14 by
    rw [← pow_add, ← pow_add]]
  simpa using (ENNReal.mul_div_mul_left (c := (2 : ℝ≥0∞) ^ 96) (2 ^ 14) 1
    (pow_ne_zero _ (by norm_num)) (ENNReal.pow_ne_top ENNReal.ofNat_ne_top))

/-- **The verbatim sketch session at the deployed receipt parameters, product form.** The multiplicative bound
`2·2^-40 + n·ε + n·P·G·(N/2^512 + q/2^96)`: each grinding candidate at its own query cap `q`, which is the special
case `qtot = G·q` of the deployed guarantee's shared budget (`Device.budget_of_bounded`). At the conservative caps
`q = G = 2^55` its query term alone is `2^14` before `n·P` (`sketchQueryTermCaps`), which is why the deployed
guarantee (`Accounting.Guarantees.RemoteSessionConservationSketchDeployed`, proved in `DeployedSketchSum.lean`)
states the sum form instead. -/
theorem RemoteSessionConservationSketchDeployedProduct :
    ∀ (n P N ρ S L q G X : ℕ) (ε : ℝ≥0∞)
      (A : RateSketch.SketchDeployedSession n P N 128 ρ 512 S L q 16 G 2773),
      7 ≤ n → 0 < P → 0 < N →
      Accounting.Assumptions.SketchRank 16 128 12 ε →
      (∀ w η j ι c E σ R O,
        let B := (A.window w η).device j ι c
        let D := RateSketch.Data.ofParts E R
        (B.prog σ D).rebuilds O D * 256 ≤ X) →
      pr (fun x : RateSketch.SketchStore N 128 ×
          (Fin n → RateSketch.SketchWindowCoins P N 128 ρ 512 16 G 2773) =>
        (∀ w t, (x.2 w).2.2 t ∈
          (A.win w x.2).correctAll x.1 (x.2 w).1 (x.2 w).2.1) ∧
          ((((∑ w, (A.win w x.2).reads x.1 (x.2 w).1 (x.2 w).2.1 : ℕ)) : ℝ) <
              (1 - (0.0015 + RateSketch.gammaSketch N 128 ρ S L X)) * (n * P * N * 32768) ∨
            ∃ w, (((A.win w x.2).reads x.1 (x.2 w).1 (x.2 w).2.1 : ℕ) : ℝ) <
              (1 - (0.01 + RateSketch.gammaSketch N 128 ρ S L X)) * (P * N * 32768)))
        ≤ 2 / 2 ^ 40 + (n : ℝ≥0∞) * ε +
          (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ 512 + (q : ℝ≥0∞) / 2 ^ 96) := by
  intro n P N ρ S L q G X ε A hn hP hN hRank hX
  have h := RemoteSessionConservationSketch n P N 128 ρ 512 S L q 16 G 2773 12 X ε A
    0.0015 0.01 (by omega) hP hN (by norm_num) (by norm_num) (by norm_num) hRank hX
  have h' :
      pr (fun x : RateSketch.SketchStore N 128 ×
          (Fin n → RateSketch.SketchWindowCoins P N 128 ρ 512 16 G 2773) =>
        (∀ w t, (x.2 w).2.2 t ∈
          (A.win w x.2).correctAll x.1 (x.2 w).1 (x.2 w).2.1) ∧
          ((((∑ w, (A.win w x.2).reads x.1 (x.2 w).1 (x.2 w).2.1 : ℕ)) : ℝ) <
              (1 - (0.0015 + RateSketch.gammaSketch N 128 ρ S L X)) * (n * P * N * 32768) ∨
            ∃ w, (((A.win w x.2).reads x.1 (x.2 w).1 (x.2 w).2.1 : ℕ) : ℝ) <
              (1 - (0.01 + RateSketch.gammaSketch N 128 ρ S L X)) * (P * N * 32768))) ≤
        ENNReal.ofReal (1 - 0.0015) ^ (2773 * n) + ENNReal.ofReal (1 - 0.01 / n) ^ (2773 * n) +
          (n : ℝ≥0∞) * ε +
            (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ 512 + (q : ℝ≥0∞) / 2 ^ 96) := by
    convert h using 1 <;> (norm_num; ring)
  refine h'.trans ?_
  rw [show (2 : ℝ≥0∞) / 2 ^ 40 = 1 / 2 ^ 40 + 1 / 2 ^ 40 by
    rw [ENNReal.div_add_div_same]; norm_num]
  gcongr
  · exact Deployed.horizon_deployed n hn
  · exact Deployed.window_deployed n (by omega)

end Accounting.SecurityProofs
