import Proofs.Flock.Soundness.Discharge.Exec.FlockBatchedL
import Proofs.Flock.Soundness.Discharge.Exec.SessionBatchAccL
import Proofs.Flock.Soundness.Audit.FlockBatchedAcc

/-!
# The batched Flock audit with the expected-time extractor (DESIGN.md §3)

`FlockBatched.lean` with the extractor of expected-time collision resistance: each drawn unit's
table is extracted from `k` reruns of everything after `Commit` in which the session accepts, so an
extraction costs `1 + k` runs in expectation whatever the acceptance probability `ε`. Its knowledge
bound is relative to `ε` (`joint_le_accBL`): `ε·Pr[fails] ≤ a + r·ε`, with `a = ε_c⁻ + √(k·adv₀L)`
and `r = N₀/(ek) < 1`. So at a wrong drawn unit
$$\varepsilon\le\varepsilon\Pr[\text{fails}]+\varepsilon\Pr[\text{link}]\le a+r\varepsilon+
\varepsilon\Pr[\text{link}],\qquad\varepsilon\le\frac{a+\varepsilon\Pr[\text{link}]}{1-r}.$$

**The analysis** (`analysisBEL`) carries that inequality as its `cover`:

* `ks`: `a/(1 − r)` for the unit's table, a bound rather than a probability;
* `link`: `ε·Pr[the extracted table has the link event]/(1 − r)`, with `LinkEvB` of
  `FlockBatched.lean` on the table extracted from accepted reruns.

At `k = cN₀/e`, `1/(1 − r) = c/(c−1)`, which is `2` at the record's `c = 2`. The knowledge term is
its own bound averaged over the draw (`flock_batched_knowledgeSoundEL`), with no named assumption.
The link term stays named (`LinkSound`), for the link theorem under expected-time collision
resistance (`Assumptions.SHA512CRExpected`).
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Audit Audit.Partition
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128


variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

variable {L : Law n}

/-- The probability that the session accepts, after `Commit`. -/
noncomputable abbrev accProbL {S : Finset (Fin n)} {R : Reg}
    (τ : Strategy (batchedSessionL H E Lf plan S R)) : ℝ :=
  expect (ind (acceptedBL (plan S R)))
    (sessAfterBL (gamesBL execArith H E Lf (plan S R) (batchStratL H E Lf plan τ).1)) (batchStratL H E Lf plan τ).2

theorem prob_batchedSession_eqL (S : Finset (Fin n)) (R : Reg)
    (τ : Strategy (batchedSessionL H E Lf plan S R)) :
    prob (· = true) (batchedSessionL H E Lf plan S R) τ = ENNReal.ofReal (accProbL H E Lf plan τ) := by
  rw [prob_batchedSessionL, ← ENNReal.ofReal_toReal
    (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)), prob_toReal]
  rfl

/-- **The batched audit's analysis, with the expected-time extractor.** `X` is the committed
transcript (the link theorem defines it), `decode` reads each drawn unit from its table's witness,
and `k` is the number of accepted reruns, with `N₀/(ek) < 1` for every table of every plan. -/
noncomputable def analysisBEL
    (decode : DecoderB (C := C) (Reg := Reg) plan tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSessionL H E Lf plan) R → (Fin C.N → Bool)) (k : ℕ)
    (hk : 1 ≤ k) (hr : ∀ S R j, rateB (plan S R) j k < 1)
    (hlow : P.LoweringSoundB plan tab decode) :
    ExtractionAnalysis L Reg (batchedSessionL H E Lf plan) where
  Tr := Fin C.N → Bool
  wrong := P.wrong
  committed := Xc
  ks S R τ u := ENNReal.ofReal (ksBoundAccBL execArith H E Lf (plan S R) (tab S R u) k
    (batchStratL H E Lf plan τ) / (1 - rateB (plan S R) (tab S R u) k))
  link S R τ u Xt := ENNReal.ofReal (accProbL H E Lf plan τ *
    extractProbAccBL execArith H E Lf (plan S R) (tab S R u) k (batchStratL H E Lf plan τ)
      (P.LinkEvB plan tab decode S R u Xt) / (1 - rateB (plan S R) (tab S R u) k))
  cover S R τ u Xt hS hu := by
    have hG : ∀ T, Committed execArith (Partition.tabOf plan tab S R u).S (Partition.tabOf plan tab S R u).sch T →
        P.LinkEvB plan tab decode S R u Xt T := fun T hc =>
      ⟨hc, fun M _ hsat => by
        by_contra hno
        push Not at hno
        exact P.mem_wrong.1 hu (P.correct_congr (fun g hg => (hno g hg).symm) (hlow S R u hS M hsat))⟩
    set ε := accProbL H E Lf plan τ with hεdef
    set r := rateB (plan S R) (tab S R u) k
    set a := ksBoundAccBL execArith H E Lf (plan S R) (tab S R u) k (batchStratL H E Lf plan τ)
    set Lp := extractProbAccBL execArith H E Lf (plan S R) (tab S R u) k (batchStratL H E Lf plan τ)
      (P.LinkEvB plan tab decode S R u Xt)
    have hr1 : 0 < 1 - r := by linarith [hr S R (tab S R u)]
    have ha : 0 ≤ a := ksBoundAccB_nonnegL _ _ _ _ _ _ _ _
    have hL : 0 ≤ Lp := extractProbAccB_nonnegL _ _ _ _ _ _ _ _ _
    have hε0 : 0 ≤ ε := expect_nonneg (ind_nonneg _) _ _
    have key : ε ≤ a / (1 - r) + ε * Lp / (1 - r) := by
      rcases hε0.eq_or_lt with h0 | hpos
      · rw [← h0]
        positivity
      · have h1 := one_le_fail_add_accBL execArith H E Lf (plan S R) (tab S R u) k
          (batchStratL H E Lf plan τ) _ hG hpos
        have h2 := joint_le_accBL execArith execArith_correct H E Lf (plan S R) (tab S R u) k hk
          (batchStratL H E Lf plan τ)
        rw [← add_div, le_div_iff₀ hr1]
        nlinarith
    rw [prob_batchedSession_eqL, ← ENNReal.ofReal_add (div_nonneg ha hr1.le)
      (div_nonneg (mul_nonneg hε0 hL) hr1.le)]
    exact ENNReal.ofReal_le_ofReal key

/-- **The batched audit's knowledge term, with the expected-time extractor**, per prover: after
each draw, the largest over the plan's tables of `(ε_c⁻ + √(k·adv₀L))/(1 − N₀/(ek))`, averaged over
the draw. -/
noncomputable def ksAvgBEL (k : ℕ) (R : Reg) (τ : Cont L Reg (batchedSessionL H E Lf plan) R) : ℝ≥0∞ :=
  avg fun ω => ⨆ j : Fin (plan (L.draw ω) R).length,
    ENNReal.ofReal (ksBoundAccBL execArith H E Lf (plan (L.draw ω) R) j k (batchStratL H E Lf plan (τ ω)) /
      (1 - rateB (plan (L.draw ω) R) j k))

/-- **`ε_ks` with the expected-time extractor**: the analysis's knowledge term is its own bound,
averaged over the draw. -/
theorem flock_batched_knowledgeSoundEL
    (decode : DecoderB (C := C) (Reg := Reg) plan tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSessionL H E Lf plan) R → (Fin C.N → Bool)) {k : ℕ}
    (hk : 1 ≤ k) (hr : ∀ S R j, rateB (plan S R) j k < 1) (hlow : P.LoweringSoundB plan tab decode) :
    (analysisBEL P H E Lf plan tab decode Xc k hk hr hlow).KnowledgeSound (ksAvgBEL H E Lf plan k) := by
  intro R τ tgt
  refine avg_mono fun ω => atTgt_le (fun u _ => ?_) (tgt ω)
  exact le_iSup (fun j : Fin (plan (L.draw ω) R).length =>
    ENNReal.ofReal (ksBoundAccBL execArith H E Lf (plan (L.draw ω) R) j k (batchStratL H E Lf plan (τ ω)) /
      (1 - rateB (plan (L.draw ω) R) j k))) (tab (L.draw ω) R u)

/-- **The batched Flock audit's count curve, with the expected-time extractor**:
`δ(K) = miss K + ε_ks(σ) + δ_link(σ)`. -/
theorem flock_batched_countEL
    (decode : DecoderB (C := C) (Reg := Reg) plan tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSessionL H E Lf plan) R → (Fin C.N → Bool)) {k : ℕ}
    (hk : 1 ≤ k) (hr : ∀ S R j, rateB (plan S R) j k < 1) (hlow : P.LoweringSoundB plan tab decode)
    {δlink : (R : Reg) → Cont L Reg (batchedSessionL H E Lf plan) R → ℝ≥0∞}
    (hlink : (analysisBEL P H E Lf plan tab decode Xc k hk hr hlow).LinkSound δlink) (K : ℕ)
    (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan))) :
    prob (fun o => o.1 = true ∧ K ≤ (P.wrong (Xc (reg σ) (cont σ))).card)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      L.miss K + ksAvgBEL H E Lf plan k (reg σ) (cont σ) + δlink (reg σ) (cont σ) :=
  extraction_audit_count (analysisBEL P H E Lf plan tab decode Xc k hk hr hlow)
    (flock_batched_knowledgeSoundEL P H E Lf plan tab decode Xc hk hr hlow) hlink K σ

/-- **Drawn units of the batched Flock audit, with the expected-time extractor**, are correct up to
`ε_ks(σ) + δ_link(σ)`. -/
theorem flock_batched_drawnEL
    (decode : DecoderB (C := C) (Reg := Reg) plan tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSessionL H E Lf plan) R → (Fin C.N → Bool)) {k : ℕ}
    (hk : 1 ≤ k) (hr : ∀ S R j, rateB (plan S R) j k < 1) (hlow : P.LoweringSoundB plan tab decode)
    {δlink : (R : Reg) → Cont L Reg (batchedSessionL H E Lf plan) R → ℝ≥0∞}
    (hlink : (analysisBEL P H E Lf plan tab decode Xc k hk hr hlow).LinkSound δlink)
    (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan))) :
    prob (fun o => o.1 = true ∧ ¬ Disjoint (P.wrong (Xc (reg σ) (cont σ))) o.2)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      ksAvgBEL H E Lf plan k (reg σ) (cont σ) + δlink (reg σ) (cont σ) :=
  extraction_audit_drawn (analysisBEL P H E Lf plan tab decode Xc k hk hr hlow)
    (flock_batched_knowledgeSoundEL P H E Lf plan tab decode Xc hk hr hlow) hlink σ

end FlockSoundness.Discharge.Exec
