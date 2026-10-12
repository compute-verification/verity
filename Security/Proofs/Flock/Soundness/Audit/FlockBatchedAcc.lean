import Proofs.Flock.Soundness.Audit.FlockBatched
import Proofs.Flock.Soundness.SessionBatchAcc

/-!
# The batched Flock audit with the expected-time extractor (DESIGN.md §3)

`FlockBatched.lean` with the extractor of expected-time collision resistance: each drawn unit's
table is extracted from `k` reruns of everything after `Commit` in which the session accepts, so an
extraction costs `1 + k` runs in expectation whatever the acceptance probability `ε`. Its knowledge
bound is relative to `ε` (`joint_le_accB`): `ε·Pr[fails] ≤ a + r·ε`, with `a = ε_c⁻ + √(k·adv₀)`
and `r = N₀/(ek) < 1`. So at a wrong drawn unit
$$\varepsilon\le\varepsilon\Pr[\text{fails}]+\varepsilon\Pr[\text{link}]\le a+r\varepsilon+
\varepsilon\Pr[\text{link}],\qquad\varepsilon\le\frac{a+\varepsilon\Pr[\text{link}]}{1-r}.$$

**The analysis** (`analysisBE`) carries that inequality as its `cover`:

* `ks`: `a/(1 − r)` for the unit's table, a bound rather than a probability;
* `link`: `ε·Pr[the extracted table has the link event]/(1 − r)`, with `LinkEvB` of
  `FlockBatched.lean` on the table extracted from accepted reruns.

At `k = cN₀/e`, `1/(1 − r) = c/(c−1)`, which is `2` at the record's `c = 2`. The knowledge term is
its own bound averaged over the draw (`flock_batched_knowledgeSoundE`), with no named assumption.
The link term stays named (`LinkSound`), for the link theorem under expected-time collision
resistance (`Assumptions.SHA512CRExpected`).
-/

namespace FlockSoundness.Audit

open Game Model Accounting
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

variable {L : Law n}

/-- The probability that the session accepts, after `Commit`. -/
noncomputable abbrev accProb {S : Finset (Fin n)} {R : Reg}
    (τ : Strategy (batchedSession H E plan S R)) : ℝ :=
  expect (ind (acceptedB (plan S R)))
    (sessAfterB (gamesB execArith H E (plan S R) (batchStrat H E plan τ).1)) (batchStrat H E plan τ).2

theorem prob_batchedSession_eq (S : Finset (Fin n)) (R : Reg)
    (τ : Strategy (batchedSession H E plan S R)) :
    prob (· = true) (batchedSession H E plan S R) τ = ENNReal.ofReal (accProb H E plan τ) := by
  rw [prob_batchedSession, ← ENNReal.ofReal_toReal
    (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)), prob_toReal]
  rfl

/-- **The batched audit's analysis, with the expected-time extractor.** `X` is the committed
transcript (the link theorem defines it), `decode` reads each drawn unit from its table's witness,
and `k` is the number of accepted reruns, with `N₀/(ek) < 1` for every table of every plan. -/
noncomputable def analysisBE
    (decode : DecoderB (C := C) (Reg := Reg) plan tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSession H E plan) R → (Fin C.N → Bool)) (k : ℕ)
    (hk : 1 ≤ k) (hr : ∀ S R j, rateB (plan S R) j k < 1)
    (hlow : P.LoweringSoundB plan tab decode) :
    ExtractionAnalysis L Reg (batchedSession H E plan) where
  Tr := Fin C.N → Bool
  wrong := P.wrong
  committed := Xc
  ks S R τ u := ENNReal.ofReal (ksBoundAccB execArith H E (plan S R) (tab S R u) k
    (batchStrat H E plan τ) / (1 - rateB (plan S R) (tab S R u) k))
  link S R τ u Xt := ENNReal.ofReal (accProb H E plan τ *
    extractProbAccB execArith H E (plan S R) (tab S R u) k (batchStrat H E plan τ)
      (P.LinkEvB plan tab decode S R u Xt) / (1 - rateB (plan S R) (tab S R u) k))
  cover S R τ u Xt hS hu := by
    have hG : ∀ T, Committed execArith (tabOf plan tab S R u).S (tabOf plan tab S R u).sch T →
        P.LinkEvB plan tab decode S R u Xt T := fun T hc =>
      ⟨hc, fun M _ hsat => by
        by_contra hno
        push Not at hno
        exact P.mem_wrong.1 hu (P.correct_congr (fun g hg => (hno g hg).symm) (hlow S R u hS M hsat))⟩
    set ε := accProb H E plan τ with hεdef
    set r := rateB (plan S R) (tab S R u) k
    set a := ksBoundAccB execArith H E (plan S R) (tab S R u) k (batchStrat H E plan τ)
    set Lp := extractProbAccB execArith H E (plan S R) (tab S R u) k (batchStrat H E plan τ)
      (P.LinkEvB plan tab decode S R u Xt)
    have hr1 : 0 < 1 - r := by linarith [hr S R (tab S R u)]
    have ha : 0 ≤ a := ksBoundAccB_nonneg _ _ _ _ _ _ _
    have hL : 0 ≤ Lp := extractProbAccB_nonneg _ _ _ _ _ _ _ _
    have hε0 : 0 ≤ ε := expect_nonneg (ind_nonneg _) _ _
    have key : ε ≤ a / (1 - r) + ε * Lp / (1 - r) := by
      rcases hε0.eq_or_lt with h0 | hpos
      · rw [← h0]
        positivity
      · have h1 := one_le_fail_add_accB execArith H E (plan S R) (tab S R u) k
          (batchStrat H E plan τ) _ hG hpos
        have h2 := joint_le_accB execArith execArith_correct H E (plan S R) (tab S R u) k hk
          (batchStrat H E plan τ)
        rw [← add_div, le_div_iff₀ hr1]
        nlinarith
    rw [prob_batchedSession_eq, ← ENNReal.ofReal_add (div_nonneg ha hr1.le)
      (div_nonneg (mul_nonneg hε0 hL) hr1.le)]
    exact ENNReal.ofReal_le_ofReal key

/-- **The batched audit's knowledge term, with the expected-time extractor**, per prover: after
each draw, the largest over the plan's tables of `(ε_c⁻ + √(k·adv₀))/(1 − N₀/(ek))`, averaged over
the draw. -/
noncomputable def ksAvgBE (k : ℕ) (R : Reg) (τ : Cont L Reg (batchedSession H E plan) R) : ℝ≥0∞ :=
  avg fun ω => ⨆ j : Fin (plan (L.draw ω) R).length,
    ENNReal.ofReal (ksBoundAccB execArith H E (plan (L.draw ω) R) j k (batchStrat H E plan (τ ω)) /
      (1 - rateB (plan (L.draw ω) R) j k))

/-- **`ε_ks` with the expected-time extractor**: the analysis's knowledge term is its own bound,
averaged over the draw. -/
theorem flock_batched_knowledgeSoundE
    (decode : DecoderB (C := C) (Reg := Reg) plan tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSession H E plan) R → (Fin C.N → Bool)) {k : ℕ}
    (hk : 1 ≤ k) (hr : ∀ S R j, rateB (plan S R) j k < 1) (hlow : P.LoweringSoundB plan tab decode) :
    (P.analysisBE H E plan tab decode Xc k hk hr hlow).KnowledgeSound (ksAvgBE H E plan k) := by
  intro R τ tgt
  refine avg_mono fun ω => atTgt_le (fun u _ => ?_) (tgt ω)
  exact le_iSup (fun j : Fin (plan (L.draw ω) R).length =>
    ENNReal.ofReal (ksBoundAccB execArith H E (plan (L.draw ω) R) j k (batchStrat H E plan (τ ω)) /
      (1 - rateB (plan (L.draw ω) R) j k))) (tab (L.draw ω) R u)

/-- **The batched Flock audit's count curve, with the expected-time extractor**:
`δ(K) = miss K + ε_ks(σ) + δ_link(σ)`. -/
theorem flock_batched_countE
    (decode : DecoderB (C := C) (Reg := Reg) plan tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSession H E plan) R → (Fin C.N → Bool)) {k : ℕ}
    (hk : 1 ≤ k) (hr : ∀ S R j, rateB (plan S R) j k < 1) (hlow : P.LoweringSoundB plan tab decode)
    {δlink : (R : Reg) → Cont L Reg (batchedSession H E plan) R → ℝ≥0∞}
    (hlink : (P.analysisBE H E plan tab decode Xc k hk hr hlow).LinkSound δlink) (K : ℕ)
    (σ : Strategy (audit L Reg (batchedSession H E plan))) :
    prob (fun o => o.1 = true ∧ K ≤ (P.wrong (Xc (reg σ) (cont σ))).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) + δlink (reg σ) (cont σ) :=
  extraction_audit_count (P.analysisBE H E plan tab decode Xc k hk hr hlow)
    (P.flock_batched_knowledgeSoundE H E plan tab decode Xc hk hr hlow) hlink K σ

/-- **Drawn units of the batched Flock audit, with the expected-time extractor**, are correct up to
`ε_ks(σ) + δ_link(σ)`. -/
theorem flock_batched_drawnE
    (decode : DecoderB (C := C) (Reg := Reg) plan tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSession H E plan) R → (Fin C.N → Bool)) {k : ℕ}
    (hk : 1 ≤ k) (hr : ∀ S R j, rateB (plan S R) j k < 1) (hlow : P.LoweringSoundB plan tab decode)
    {δlink : (R : Reg) → Cont L Reg (batchedSession H E plan) R → ℝ≥0∞}
    (hlink : (P.analysisBE H E plan tab decode Xc k hk hr hlow).LinkSound δlink)
    (σ : Strategy (audit L Reg (batchedSession H E plan))) :
    prob (fun o => o.1 = true ∧ ¬ Disjoint (P.wrong (Xc (reg σ) (cont σ))) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + δlink (reg σ) (cont σ) :=
  extraction_audit_drawn (P.analysisBE H E plan tab decode Xc k hk hr hlow)
    (P.flock_batched_knowledgeSoundE H E plan tab decode Xc hk hr hlow) hlink σ

end Partition

end FlockSoundness.Audit
