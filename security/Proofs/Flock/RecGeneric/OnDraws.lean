module

public import Proofs.Flock.RecGeneric.SampledProof
public import Proofs.Flock.RecGeneric.StatementProof

/-!
# B1's chain read on the draws

`GenericSampledKS`, `GenericSampledKSJoint`, B1 and `RecursiveSampledKS` read their extractor's facts at every sample of
`n` proof blocks. The audit reads them only at the samples its design draws, so each has a form here whose hypotheses
are read there alone:

* `Extractor.CoversOn`, `JointExtractor.SoundOn`, `JointExtractor.KnowledgeSoundOn`: `Covers`, `Sound` and
  `KnowledgeSound` at `L.draw ω`;
* `Extractor.analysisOn`: `Extractor.analysis` with the knowledge term `⊤` at a sample `L` never draws, so its `cover`
  holds there with no fact about the extractor; `KnowledgeSound` and `BindSound` read only drawn samples, where it is
  `Extractor.analysis`'s;
* `genericSampledKSPa_on`, `genericSampledKSJoint_on`, `genericSampledKSClaims_on`, `recursiveSampledKS_on`: the four
  bounds unchanged, from the draws-only hypotheses (L1's at `stmt (L.draw ω) R`).
-/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open Game Spot.Sampling LinkFinder
open scoped ENNReal

variable {n : ℕ} {Reg : Type} {Dh : Type} {Hc : List UInt8 → Dh} {B : Binding Reg Hc} {PS : ProofSystem n Reg}

namespace Extractor

variable (Ex : Extractor PS B)

/-- `Covers` at the samples `L` draws. -/
def CoversOn (L : SamplingDesign n) (Correct : Reg → B.Trace → Fin n → Prop) : Prop :=
  ∀ ω R (τ : Strategy (PS.session (L.draw ω) R)) u (tr : B.Trace), u ∈ L.draw ω → ¬ Correct R tr u →
    prob (· = true) (PS.session (L.draw ω) R) τ ≤ Ex.ks (L.draw ω) R τ u + Ex.link (L.draw ω) R τ u tr

open Classical in
/-- `analysis` with the knowledge term `⊤` at a sample `L` never draws. -/
noncomputable def analysisOn (L : SamplingDesign n) (Correct : Reg → B.Trace → Fin n → Prop)
    (Xc : (R : Reg) → Cont L Reg PS.session R → B.Trace) (h : Ex.CoversOn L Correct) :
    ExtractionAnalysis L Reg PS.session where
  Tr := Reg × B.Trace
  incorrect Xt := incorrect B Correct Xt.1 Xt.2
  committed R τ := (R, Xc R τ)
  ks S R τ u := if ∃ ω, L.draw ω = S then Ex.ks S R τ u else ⊤
  link S R τ u Xt := if Xt.1 = R then Ex.link S R τ u Xt.2 else ⊤
  cover S R τ u Xt hS hu := by
    obtain ⟨R', tr⟩ := Xt
    dsimp only at hu ⊢
    by_cases hd : ∃ ω, L.draw ω = S
    · obtain ⟨ω, rfl⟩ := hd
      rw [ite_eq_left ⟨ω, rfl⟩]
      by_cases hR : R' = R
      · subst hR
        rw [ite_eq_left rfl]
        exact h ω R' τ u tr hS (mem_incorrect.1 hu)
      · rw [ite_eq_right hR, add_top]
        exact le_top
    · rw [ite_eq_right hd, top_add]
      exact le_top

variable {L : SamplingDesign n} {Correct : Reg → B.Trace → Fin n → Prop}
  {Xc : (R : Reg) → Cont L Reg PS.session R → B.Trace}

theorem knowledgeSound_analysisOn (h : Ex.CoversOn L Correct) :
    (Ex.analysisOn L Correct Xc h).KnowledgeSound (Ex.ksAvg L) := by
  intro R τ tgt
  show avg _ ≤ avg _
  refine avg_mono fun ω => atTgt_le (fun u hu => ?_) (tgt ω)
  show (if ∃ ω', L.draw ω' = L.draw ω then Ex.ks (L.draw ω) R (τ ω) u else ⊤) ≤ _
  rw [ite_eq_left ⟨ω, rfl⟩]
  exact le_iSup₂ (f := fun u (_ : u ∈ L.draw ω) => Ex.ks (L.draw ω) R (τ ω) u) u hu

theorem linkSound_analysisOn (h : Ex.CoversOn L Correct) (δ : (R : Reg) → Cont L Reg PS.session R → ℝ≥0∞)
    (hδ : ∀ R τ (tgt : L.Ω → Option (Fin n)),
      avg (fun ω => atTgt (L.draw ω) (tgt ω) fun u => Ex.link (L.draw ω) R (τ ω) u (Xc R τ)) ≤ δ R τ) :
    (Ex.analysisOn L Correct Xc h).BindSound δ := by
  intro R τ tgt
  refine le_of_eq_of_le (congrArg avg (funext fun ω => ?_)) (hδ R τ tgt)
  simp only [analysisOn, ↓reduceIte]

end Extractor

/-- **`GenericSampledKS`'s event form from `Covers` on the draws.** -/
theorem genericSampledKS_on {L : SamplingDesign n} (Correct : Reg → B.Trace → Fin n → Prop) (Ex : Extractor PS B)
    (Tr : Trials L PS B) (Xc : (R : Reg) → Cont L Reg PS.session R → B.Trace) {M : ℕ}
    (hcov : Ex.CoversOn L Correct) (hop : Tr.Opens) (hwf : Tr.WellFormed) (hT : Tr.Transfers Ex)
    (hX : Tr.Plurality Xc) (hM : 1 ≤ M) (σ : Strategy (audit L Reg PS.session))
    (E : B.Trace → Finset (Fin n) → Prop) (hCR : Tr.CR (reg σ) (cont σ) M) :
    prob (fun o => o.1 = true ∧ E (Xc (reg σ) (cont σ)) o.2) (audit L Reg PS.session) σ ≤
      prCoin (fun ω => E (Xc (reg σ) (cont σ)) (L.draw ω) ∧
        Disjoint (L.draw ω) (incorrect B Correct (reg σ) (Xc (reg σ) (cont σ)))) +
      Ex.ksAvg L (reg σ) (cont σ) + Tr.bindErr (reg σ) (cont σ) M := by
  classical
  have hlink := Ex.linkSound_analysisOn (Xc := Xc) hcov
    (fun R τ => if Tr.CR R τ M then Tr.bindErr R τ M else ⊤) fun R τ tgt => by
      split_ifs with h
      · exact Tr.link_le_bindErr hT hX hwf hop hM R τ h tgt
      · exact le_top
  have h := extraction_audit_le (Ex.analysisOn L Correct Xc hcov) (Ex.knowledgeSound_analysisOn hcov) hlink
    (fun Xt S => E Xt.2 S) σ
  simp only [ite_eq_left hCR] at h
  exact h

/-- **`GenericSampledKSPa` from `Covers` on the draws.** -/
theorem genericSampledKSPa_on {L : SamplingDesign n} (Correct : Reg → B.Trace → Fin n → Prop) (Ex : Extractor PS B)
    (Tr : Trials L PS B) (Xc : (R : Reg) → Cont L Reg PS.session R → B.Trace) {M : ℕ}
    (hcov : Ex.CoversOn L Correct) (hop : Tr.Opens) (hwf : Tr.WellFormed) (hT : Tr.Transfers Ex)
    (hX : Tr.Plurality Xc) (hM : 1 ≤ M) (σ : Strategy (audit L Reg PS.session)) (hCR : Tr.CR (reg σ) (cont σ) M) :
    prob (fun o => o.1 = true) (audit L Reg PS.session) σ ≤
      L.missProb (incorrect B Correct (reg σ) (Xc (reg σ) (cont σ))) + Ex.ksAvg L (reg σ) (cont σ) +
        Tr.bindErr (reg σ) (cont σ) M := by
  refine (prob_mono (p := fun o : Bool × Finset (Fin n) => o.1 = true) (q := fun o => o.1 = true ∧ True)
    (fun _ h => ⟨h, trivial⟩) _ _).trans
    ((genericSampledKS_on Correct Ex Tr Xc hcov hop hwf hT hX hM σ (fun _ _ => True) hCR).trans ?_)
  gcongr
  exact prCoin_mono fun ω h => h.2

namespace JointExtractor

variable (J : JointExtractor PS B) {Correct : Reg → B.Trace → Fin n → Prop}

/-- `Sound` at the samples `L` draws. -/
def SoundOn (L : SamplingDesign n) (Correct : Reg → B.Trace → Fin n → Prop) : Prop :=
  ∀ ω R u w, u ∈ L.draw ω → J.Rel (L.draw ω) R u w →
    (∀ c ∈ J.reads u, B.com (J.opens (L.draw ω) R u w c) = B.cs R c) ∧
      ∀ tr : B.Trace, (∀ c ∈ J.reads u, B.val (J.opens (L.draw ω) R u w c) = tr c) → Correct R tr u

/-- `KnowledgeSound` at the samples `L` draws. -/
def KnowledgeSoundOn (L : SamplingDesign n)
    (κ : (S : Finset (Fin n)) → (R : Reg) → Strategy (PS.session S R) → Fin n → ℝ≥0∞) : Prop :=
  ∀ ω R (τ : Strategy (PS.session (L.draw ω) R)) u, u ∈ L.draw ω →
    prob (fun o => o.1 = true ∧ J.Fails (L.draw ω) R u o) (J.G (L.draw ω) R) (J.strat (L.draw ω) R τ) ≤
      κ (L.draw ω) R τ u

theorem toExtractor_coversOn (hJ : J.Judged) {L : SamplingDesign n} (hS : J.SoundOn L Correct) :
    J.toExtractor.CoversOn L Correct := by
  intro ω R τ u tr hu hX
  rw [← hJ (L.draw ω) R τ]
  refine (prob_mono (q := fun o => (o.1 = true ∧ J.Fails (L.draw ω) R u o) ∨ (o.1 = true ∧ ∃ w, o.2 u = some w ∧
      J.Rel (L.draw ω) R u w ∧ ∃ c ∈ J.reads u, B.val (J.opens (L.draw ω) R u w c) ≠ tr c)) (fun o h => ?_) _ _).trans
    (prob_or_le _ _ _ _)
  by_cases hf : J.Fails (L.draw ω) R u o
  · exact Or.inl ⟨h, hf⟩
  · obtain ⟨w, hw, hR⟩ := not_not.1 hf
    refine Or.inr ⟨h, w, hw, hR, ?_⟩
    by_contra hc
    exact hX ((hS ω R u w hu hR).2 tr fun c hc' => by_contra fun hne => hc ⟨c, hc', hne⟩)

theorem toTrials_opensOn {L : SamplingDesign n} (hS : J.SoundOn L Correct) : (J.toTrials L).Opens := by
  intro R τ c e v hv
  obtain ⟨-, u, hu, hc, w, hw, hR, hval⟩ := (J.mem_readsOut (L := L)).1 hv
  obtain ⟨u', hu', hc', w', -, hR', hop, hval'⟩ := Classical.epsilon_spec
    (p := fun o => ∃ u ∈ L.draw e.1, c ∈ J.reads u ∧ ∃ w, e.2.2 u = some w ∧ J.Rel (L.draw e.1) R u w ∧
      o = J.opens (L.draw e.1) R u w c ∧ B.val o = v)
    ⟨J.opens (L.draw e.1) R u w c, u, hu, hc, w, hw, hR, rfl, hval⟩
  have hop' : (J.toTrials L).op R τ c e v = J.opens (L.draw e.1) R u' w' c := hop
  refine ⟨?_, hval'⟩
  rw [hop']
  exact (hS e.1 R u' w' hu' hR').1 c hc'

end JointExtractor

/-- **`GenericSampledKSJoint` from `Sound` and `KnowledgeSound` on the draws.** -/
theorem genericSampledKSJoint_on {L : SamplingDesign n} (Correct : Reg → B.Trace → Fin n → Prop)
    (J : JointExtractor PS B) (κ : (S : Finset (Fin n)) → (R : Reg) → Strategy (PS.session S R) → Fin n → ℝ≥0∞)
    {M : ℕ} (hJ : J.Judged) (hS : J.SoundOn L Correct) (hc : ∀ S R o, 0 ≤ J.cost S R o)
    (hks : J.KnowledgeSoundOn L κ) (hM : 1 ≤ M) (σ : Strategy (audit L Reg PS.session))
    (hCR : (J.toTrials L).CR (reg σ) (cont σ) M) :
    prob (fun o => o.1 = true) (audit L Reg PS.session) σ ≤
      L.missProb (incorrect B Correct (reg σ) ((J.toTrials L).plur (reg σ) (cont σ))) +
        avg (fun ω => ⨆ u ∈ L.draw ω, κ (L.draw ω) (reg σ) (cont σ ω) u) +
        (J.toTrials L).bindErr (reg σ) (cont σ) M := by
  refine (genericSampledKSPa_on Correct J.toExtractor (J.toTrials L) (J.toTrials L).plur
    (J.toExtractor_coversOn hJ hS) (J.toTrials_opensOn hS) (J.toTrials_wellFormed hc L)
    (J.toTrials_transfers L) (J.toTrials L).plurality_plur hM σ hCR).trans ?_
  gcongr
  exact avg_mono fun ω => iSup₂_mono fun u hu => hks ω _ _ u hu

end FlockSoundness.SampledGeneric

namespace Claims.RecGeneric

open Claims FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.SampledGeneric
open scoped ENNReal

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
variable {n : ℕ} {Reg Dh : Type} {Hc : List UInt8 → Dh}

/-- **`jointOf` is knowledge sound at `err` on the draws**, from `PrBad ≤ err` at the drawn samples' statements. -/
theorem jointOf_knowledgeSoundOn (H : Msg → Msg) (V : Verifier Coin) (Rl : Rel W) (E : Claims.Extractor Coin W V)
    (stmt : Finset (Fin n) → Reg → Msg) (B : FlockSoundness.SampledGeneric.Binding Reg Hc)
    (reads : Fin n → Finset B.Pos) (opens : Finset (Fin n) → Reg → Fin n → W → B.Pos → B.Opening)
    (L : SamplingDesign n) (err : Msg → Prover Unit → ℝ)
    (hks : ∀ ω R (P : Prover Unit), PrBad H V Rl E (stmt (L.draw ω) R) P ≤ err (stmt (L.draw ω) R) P) :
    (jointOf H V Rl E stmt B reads opens).KnowledgeSoundOn L
      (fun S R τ _ => ENNReal.ofReal (err (stmt S R) (proverOf τ))) := by
  intro ω R τ u hu
  rw [← ENNReal.ofReal_toReal (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))]
  refine ENNReal.ofReal_le_ofReal (le_trans (le_of_eq ?_) (hks ω R (proverOf τ)))
  rw [prob_toReal]
  unfold PrBad
  rw [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Ex_prod, Ex_unit, Claims.Bridge.Ex_prod]
  simp only [jointOf, proofSystemOf, expect, Ex]
  congr 1
  refine Finset.sum_congr rfl fun r₀ _ => ?_
  congr 1
  refine Finset.sum_congr rfl fun c _ => ind_congr_iff ?_
  simp only [FlockSoundness.SampledGeneric.JointExtractor.Fails, hu, ↓reduceIte, rel_ext_iff, proverOf,
    Option.some.injEq, exists_eq_left']
  exact and_congr ⟨of_decide_eq_true, decide_eq_true⟩ Iff.rfl

/-- **B1 on the draws**: `GenericSampledKSClaims`'s bound from `PrBad ≤ err` and `Sound` at the drawn samples only. -/
theorem genericSampledKSClaims_on (V : Verifier Coin) (Rl : Rel W) (E : Claims.Extractor Coin W V)
    (H : Hash.Msg → Hash.Msg) (err : Hash.Msg → Prover Unit → ℝ) (L : SamplingDesign n) (B : Binding Reg Hc)
    (Correct : Reg → B.Trace → Fin n → Prop) (stmt : Finset (Fin n) → Reg → Hash.Msg)
    (reads : Fin n → Finset B.Pos) (opens : Finset (Fin n) → Reg → Fin n → W → B.Pos → B.Opening) {M : ℕ}
    (hks : ∀ ω R (P : Prover Unit), PrBad H V Rl E (stmt (L.draw ω) R) P ≤ err (stmt (L.draw ω) R) P)
    (hS : (jointOf H V Rl E stmt B reads opens).SoundOn L Correct) (hM : 1 ≤ M)
    (σ : Strategy (audit L Reg (proofSystemOf H V stmt).session))
    (hCR : ((jointOf H V Rl E stmt B reads opens).toTrials L).CR (reg σ) (cont σ) M) :
    prob (fun o => o.1 = true) (audit L Reg (proofSystemOf H V stmt).session) σ ≤
      L.missProb (incorrect B Correct (reg σ)
          (((jointOf H V Rl E stmt B reads opens).toTrials L).plur (reg σ) (cont σ))) +
        avg (fun ω => ⨆ u ∈ L.draw ω,
          ENNReal.ofReal (err (stmt (L.draw ω) (reg σ)) (proverOf (x := stmt (L.draw ω) (reg σ)) (cont σ ω)))) +
        ((jointOf H V Rl E stmt B reads opens).toTrials L).bindErr (reg σ) (cont σ) M :=
  genericSampledKSJoint_on Correct (jointOf H V Rl E stmt B reads opens)
    (fun S R τ _ => ENNReal.ofReal (err (stmt S R) (proverOf τ)))
    (jointOf_judged H V Rl E stmt B reads opens) hS (fun _ _ _ => Nat.cast_nonneg _)
    (jointOf_knowledgeSoundOn H V Rl E stmt B reads opens L err hks) hM σ hCR

/-- **The recursive audit on the draws**: `RecursiveSampledKS`'s bound from L1's hypotheses at the drawn samples'
statements and `Sound` at the drawn samples only. -/
theorem recursiveSampledKS_on (H : Hash.Msg → Hash.Msg) (I : Proto Coin W) (Sv : Server) (K : Comp Coin)
    (mI : Fin I.nT → ℕ) (Mo : ℕ) (L : SamplingDesign n) (B : Binding Reg Hc)
    (Correct : Reg → B.Trace → Fin n → Prop) (stmt : Finset (Fin n) → Reg → Hash.Msg)
    (reads : Fin n → Finset B.Pos) (opens : Finset (Fin n) → Reg → Fin n → W → B.Pos → B.Opening) {M : ℕ}
    (hWF : I.WF) (hKS : I.KSAbortLin H) (hLC : I.LastCoinFree H) (hBd : Sv.Binds)
    (hAt : ∀ ω R, K.OuterKSAt H (stmt (L.draw ω) R) ∧ 0 ≤ K.κO (stmt (L.draw ω) R) ∧
      0 ≤ K.γO (stmt (L.draw ω) R) ∧
      K.Contract H (Sv.IVStar I.V I.ans) (fun v => I.opens (Sv.openView v)) (stmt (L.draw ω) R))
    (hS : (jointOf H K.V (K.R I) (K.E I Sv) stmt B reads opens).SoundOn L Correct) (hM : 1 ≤ M)
    (σ : Strategy (audit L Reg (proofSystemOf H K.V stmt).session))
    (hCR : ((jointOf H K.V (K.R I) (K.E I Sv) stmt B reads opens).toTrials L).CR (reg σ) (cont σ) M) :
    prob (fun o => o.1 = true) (audit L Reg (proofSystemOf H K.V stmt).session) σ ≤
      L.missProb (incorrect B Correct (reg σ)
          (((jointOf H K.V (K.R I) (K.E I Sv) stmt B reads opens).toTrials L).plur (reg σ) (cont σ))) +
        avg (fun ω => ⨆ u ∈ L.draw ω,
          ENNReal.ofReal (K.err I Sv H (proverOf (x := stmt (L.draw ω) (reg σ)) (cont σ ω))
            (stmt (L.draw ω) (reg σ)) mI Mo)) +
        ((jointOf H K.V (K.R I) (K.E I Sv) stmt B reads opens).toTrials L).bindErr (reg σ) (cont σ) M :=
  genericSampledKSClaims_on K.V (K.R I) (K.E I Sv) H (fun x P => K.err I Sv H P x mI Mo) L B Correct stmt reads
    opens (fun ω R P => compositionKS Coin W H I Sv K hWF hKS hLC hBd (stmt (L.draw ω) R) (hAt ω R).1
      (hAt ω R).2.1 (hAt ω R).2.2.1 (hAt ω R).2.2.2 Unit P mI Mo) hS hM σ hCR

end Claims.RecGeneric
