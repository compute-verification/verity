module

public import Proofs.Flock.SampledGeneric.Statements

/-!
# The sampled audit as a compiler, generic in the proof system (proved)

The proof is the one-stage audit with a randomized extractor (`Audit.extraction_audit_le`), whose sampling term is the
sampling design's alone, at the analysis the extractor defines (`Extractor.analysis`):

* `Extractor.analysis`: the committed transcript is the registration and the committed trace, "wrong" is the
  incorrect set, and the binding error is the extractor's at the committed registration (`⊤` at any other, which no
  committed transcript reaches);
* `Extractor.knowledgeSound_analysis`: the knowledge term at the proof block a rule picks is at most the largest over
  the sample (`Extractor.ksAvg`);
* `Trials.errOf_plur_le`: at the plurality, a trial's binding error is at most its finder's collision term,
  `2·E[cost]/2^256 + 1/(eM)` (`LinkFinder.link_mass_le`, with the finder's output from two openings,
  `LinkFinder.outOf_spec`);
* `Trials.link_le_bindErr`: so the binding error at the committed trace is at most `ε_bind` (`Trials.Transfers`);
* `GenericSampledKS`, `GenericSampledKSPa`, `GenericSampledKSEscape`, `GenericSampledKSCount`.
-/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open Game Audit LinkFinder
open scoped ENNReal

variable {n : ℕ} {Reg : Type} {Dh : Type} {Hc : List UInt8 → Dh} {B : Binding Reg Hc} {L : Law n}
  {PS : ProofSystem n Reg}

theorem mem_incorrect {Correct : Reg → B.Trace → Fin n → Prop} {R : Reg} {X : B.Trace} {u : Fin n} :
    u ∈ incorrect B Correct R X ↔ ¬ Correct R X u := by
  classical
  simp [incorrect]

namespace Extractor

variable (Ex : Extractor PS B) (Correct : Reg → B.Trace → Fin n → Prop)
  (Xc : (R : Reg) → Cont L Reg PS.session R → B.Trace)

open Classical in
/-- **The extraction analysis of the compiled audit**: the committed transcript is the registration and the committed
trace, "wrong" is the incorrect set at the registration, and the binding error is the extractor's at the committed
registration and `⊤` at any other. -/
noncomputable def analysis (h : Ex.Covers Correct) : ExtractionAnalysis L Reg PS.session where
  Tr := Reg × B.Trace
  wrong Xt := incorrect B Correct Xt.1 Xt.2
  committed R τ := (R, Xc R τ)
  ks := Ex.ks
  link S R τ u Xt := if Xt.1 = R then Ex.link S R τ u Xt.2 else ⊤
  cover S R τ u Xt hS hu := by
    obtain ⟨R', X⟩ := Xt
    dsimp only at hu ⊢
    by_cases hR : R' = R
    · subst hR
      rw [ite_eq_left rfl]
      exact h S R' τ u X hS (mem_incorrect.1 hu)
    · rw [ite_eq_right hR, add_top]
      exact le_top

theorem knowledgeSound_analysis (h : Ex.Covers Correct) :
    (Ex.analysis Correct Xc h).KnowledgeSound (Ex.ksAvg L) := by
  intro R τ tgt
  show avg _ ≤ avg _
  exact avg_mono fun ω => atTgt_le (fun u hu =>
    le_iSup₂ (f := fun u (_ : u ∈ L.draw ω) => Ex.ks (L.draw ω) R (τ ω) u) u hu) (tgt ω)

theorem linkSound_analysis (h : Ex.Covers Correct) (δ : (R : Reg) → Cont L Reg PS.session R → ℝ≥0∞)
    (hδ : ∀ R τ (tgt : L.Ω → Option (Fin n)),
      avg (fun ω => atTgt (L.draw ω) (tgt ω) fun u => Ex.link (L.draw ω) R (τ ω) u (Xc R τ)) ≤ δ R τ) :
    (Ex.analysis Correct Xc h).BindSound δ := by
  intro R τ tgt
  refine le_of_eq_of_le (congrArg avg (funext fun ω => ?_)) (hδ R τ tgt)
  simp only [analysis, ↓reduceIte]

end Extractor

namespace Trials

variable (Tr : Trials L PS B)

/-- **The binding error at the plurality**: at every position, the trial's binding error against the committed trace is
at most its finder's collision term, under ecr/sha-512 at the finder. -/
theorem errOf_plur_le {Xc : (R : Reg) → Cont L Reg PS.session R → B.Trace} (hX : Tr.Plurality Xc)
    (hwf : Tr.WellFormed) (hop : Tr.Opens) {M : ℕ} (hM : 1 ≤ M) (R : Reg) (τ : Cont L Reg PS.session R)
    (hCR : Tr.CR R τ M) (c : B.Pos) :
    Tr.errOf R τ c (Xc R τ c) ≤ 2 * Tr.ecost R τ c / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) := by
  unfold errOf ecost
  split_ifs with h
  · exact link_mass_le (Tr.T R τ c h) (Tr.sT R τ c h) (Tr.rd R τ c) Hc (Tr.finderOut R τ c) (Tr.cost R τ)
      (hwf.1 R τ) M hM
      (fun x hS => outOf_spec (Tr.rd R τ c) (Tr.op R τ c) B.collide (fun o => B.com o = B.cs R c) B.val
        (fun e v hv => hop R τ c e v hv) (fun o o' h h' hv => B.collide_spec o o' (h.trans h'.symm) hv) x hS)
      (hCR c h) (hX R τ c h)
  · simp only [mul_zero, zero_div, zero_add]
    positivity

/-- **The binding error at the committed trace is at most `ε_bind`**, at a prover whose link finders satisfy ecr/sha-512. -/
theorem link_le_bindErr {Ex : Extractor PS B} (hT : Tr.Transfers Ex)
    {Xc : (R : Reg) → Cont L Reg PS.session R → B.Trace} (hX : Tr.Plurality Xc) (hwf : Tr.WellFormed)
    (hop : Tr.Opens) {M : ℕ} (hM : 1 ≤ M) (R : Reg) (τ : Cont L Reg PS.session R) (hCR : Tr.CR R τ M)
    (tgt : L.Ω → Option (Fin n)) :
    avg (fun ω => atTgt (L.draw ω) (tgt ω) fun u => Ex.link (L.draw ω) R (τ ω) u (Xc R τ)) ≤
      Tr.bindErr R τ M := by
  refine (hT R τ tgt (Xc R τ)).trans (ENNReal.ofReal_le_ofReal ?_)
  have hρ : 0 < 1 - Tr.ρ := by linarith [hwf.2.2.1]
  refine div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => ?_) hρ.le
  refine mul_le_mul_of_nonneg_left ?_ (hwf.2.1 c)
  linarith [Tr.errOf_plur_le hX hwf hop hM R τ hCR c]

end Trials

end FlockSoundness.SampledGeneric

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.SampledGeneric in
/-- **The sampled audit's knowledge soundness, generic in the proof system** (`Flock.Guarantees.GenericSampledKS`):
`Audit.extraction_audit_le` at `Extractor.analysis`, with the knowledge term `Extractor.ksAvg` and the binding error
`ε_bind` at every prover whose link finders satisfy ecr/sha-512 (`⊤` at any other). -/
theorem Flock.SecurityProofs.GenericSampledKS : Flock.Guarantees.GenericSampledKS := by
  intro n Reg Dh Hc L PS B Correct Ex Tr Xc M hcov hop hwf hT hX hM σ E hCR
  classical
  have hlink := Ex.linkSound_analysis Correct Xc hcov
    (fun R τ => if Tr.CR R τ M then Tr.bindErr R τ M else ⊤) fun R τ tgt => by
      split_ifs with h
      · exact Tr.link_le_bindErr hT hX hwf hop hM R τ h tgt
      · exact le_top
  have h := extraction_audit_le (Ex.analysis Correct Xc hcov) (Ex.knowledgeSound_analysis Correct Xc hcov) hlink
    (fun Xt S => E Xt.2 S) σ
  simp only [ite_eq_left hCR] at h
  exact h

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.SampledGeneric in
/-- **`Pr[accept] ≤ P_a(F) + ε_ks + ε_bind`** (`Flock.Guarantees.GenericSampledKSPa`): `GenericSampledKS` at the sure
event. -/
theorem Flock.SecurityProofs.GenericSampledKSPa : Flock.Guarantees.GenericSampledKSPa := by
  intro n Reg Dh Hc L PS B Correct Ex Tr Xc M hcov hop hwf hT hX hM σ hCR
  refine (prob_mono (p := fun o : Bool × Finset (Fin n) => o.1 = true) (q := fun o => o.1 = true ∧ True)
    (fun _ h => ⟨h, trivial⟩) _ _).trans
    ((Flock.SecurityProofs.GenericSampledKS n Reg Dh Hc L PS B Correct Ex Tr Xc M hcov hop hwf hT hX hM σ
      (fun _ _ => True) hCR).trans ?_)
  gcongr
  exact prCoin_mono fun ω h => h.2

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.SampledGeneric in
/-- **The escape bound at a set of proof blocks** (`Flock.Guarantees.GenericSampledKSEscape`): `GenericSampledKS` at
`W ⊆ F`, whose sampling term is at most `P_a(W)`: a sample that misses `F` misses `W`. -/
theorem Flock.SecurityProofs.GenericSampledKSEscape : Flock.Guarantees.GenericSampledKSEscape := by
  intro n Reg Dh Hc L PS B Correct Ex Tr Xc M hcov hop hwf hT hX hM W σ hCR
  refine (Flock.SecurityProofs.GenericSampledKS n Reg Dh Hc L PS B Correct Ex Tr Xc M hcov hop hwf hT hX hM σ
    (fun X _ => W ⊆ incorrect B Correct (reg σ) X) hCR).trans ?_
  gcongr
  exact prCoin_mono fun ω h => h.2.mono_right h.1

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.SampledGeneric in
/-- **The count curve** (`Flock.Guarantees.GenericSampledKSCount`): `GenericSampledKS` at `|F| ≥ K`, whose sampling
term is at most `miss(K)` (`Audit.Law.escape_le_miss`). -/
theorem Flock.SecurityProofs.GenericSampledKSCount : Flock.Guarantees.GenericSampledKSCount := by
  intro n Reg Dh Hc L PS B Correct Ex Tr Xc M hcov hop hwf hT hX hM K σ hCR
  refine (Flock.SecurityProofs.GenericSampledKS n Reg Dh Hc L PS B Correct Ex Tr Xc M hcov hop hwf hT hX hM σ
    (fun X _ => K ≤ (incorrect B Correct (reg σ) X).card) hCR).trans ?_
  gcongr
  by_cases hB : K ≤ (incorrect B Correct (reg σ) (Xc (reg σ) (cont σ))).card
  · exact (prCoin_mono fun ω h => h.2).trans (L.escape_le_miss hB)
  · exact (prCoin_mono (q := fun _ => False) fun ω h => absurd h.1 hB).trans (prCoin_false.le.trans zero_le)
