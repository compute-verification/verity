module

public import Proofs.Flock.Soundness.Audit.OneStage
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real
import Proofs.Flock.Soundness.Audit.SamplingDesign
import Proofs.Flock.Soundness.Game

/-! The one-stage audit with a randomized extractor, whose success is a probability over its reruns and whose bounds depend on the prover. -/

@[expose] public section

namespace FlockSoundness.Audit

open Game
open scoped ENNReal

variable {n : ℕ} {L : SamplingDesign n} {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

noncomputable def atTgt (S : Finset (Fin n)) (t : Option (Fin n)) (f : Fin n → ℝ≥0∞) : ℝ≥0∞ :=
  t.elim 0 fun u => if u ∈ S then f u else 0

theorem atTgt_le {S : Finset (Fin n)} {f : Fin n → ℝ≥0∞} {c : ℝ≥0∞} (h : ∀ u ∈ S, f u ≤ c)
    (t : Option (Fin n)) : atTgt S t f ≤ c := by
  cases t with
  | none => exact zero_le
  | some u =>
    simp only [atTgt, Option.elim_some]
    split_ifs with hu
    · exact h u hu
    · exact zero_le

structure ExtractionAnalysis (L : SamplingDesign n) (Reg : Type) (session : Finset (Fin n) → Reg → Game Bool)
    where
  Tr : Type
  incorrect : Tr → Finset (Fin n)
  committed : (R : Reg) → Cont L Reg session R → Tr
  /-- `Pr[the session accepts ∧ extraction of u fails]` -/
  ks : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → ℝ≥0∞
  /-- `Pr[it accepts ∧ extraction of u succeeds but differs from X on u's wires]` -/
  link : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → Tr → ℝ≥0∞
  cover : ∀ S R (τ : Strategy (session S R)) u X, u ∈ S → u ∈ incorrect X →
    prob (· = true) (session S R) τ ≤ ks S R τ u + link S R τ u X

namespace ExtractionAnalysis

variable (A : ExtractionAnalysis L Reg session)

def committedOf (σ : Strategy (audit L Reg session)) : A.Tr := A.committed (reg σ) (cont σ)

/-- Named hypothesis `ε_ks`, which may depend on the prover's state. -/
def KnowledgeSound (εks : (R : Reg) → Cont L Reg session R → ℝ≥0∞) : Prop :=
  ∀ (R : Reg) (τ : Cont L Reg session R) (tgt : L.Ω → Option (Fin n)),
    avg (fun ω => atTgt (L.draw ω) (tgt ω) (A.ks (L.draw ω) R (τ ω))) ≤ εks R τ

/-- Named hypothesis `ε_bind`, which may depend on the prover's state. -/
def BindSound (εbind : (R : Reg) → Cont L Reg session R → ℝ≥0∞) : Prop :=
  ∀ (R : Reg) (τ : Cont L Reg session R) (tgt : L.Ω → Option (Fin n)),
    avg (fun ω => atTgt (L.draw ω) (tgt ω) fun u => A.link (L.draw ω) R (τ ω) u (A.committed R τ)) ≤
      εbind R τ

end ExtractionAnalysis

namespace Analysis

variable (A : Analysis L Reg session)

open Classical in
noncomputable def toExtraction : ExtractionAnalysis L Reg session where
  Tr := A.Tr
  incorrect := A.incorrect
  committed := A.committed
  ks S R τ u := prob (fun ok => ok = true ∧ A.ksFail S R τ u) (session S R) τ
  link S R τ u X := prob (fun ok => ok = true ∧ A.linkFail S R τ u X) (session S R) τ
  cover S R τ u X _ hu :=
    (prob_mono (fun _ h => (A.cover S R τ u X hu).elim (fun hk => Or.inl ⟨h, hk⟩)
      fun hl => Or.inr ⟨h, hl⟩) _ _).trans (prob_or_le _ _ _ _)

theorem atTgt_toExtraction (S : Finset (Fin n)) (R : Reg) (τ : Strategy (session S R))
    (t : Option (Fin n)) (F : Fin n → Prop) :
    atTgt S t (fun u => prob (fun ok => ok = true ∧ F u) (session S R) τ) =
      prob (fun ok => ok = true ∧ ∃ u, t = some u ∧ u ∈ S ∧ F u) (session S R) τ := by
  classical
  rcases t with _ | u
  · simp [atTgt, prob_false]
  · by_cases hu : u ∈ S
    · simp [atTgt, hu]
    · simp [atTgt, hu, prob_false]

theorem KnowledgeSound.toExtraction {A : Analysis L Reg session} {εks : ℝ≥0∞}
    (h : A.KnowledgeSound εks) : A.toExtraction.KnowledgeSound fun _ _ => εks := fun R τ tgt =>
  le_of_eq_of_le (congrArg avg (funext fun ω =>
    atTgt_toExtraction (L.draw ω) R (τ ω) (tgt ω) (A.ksFail (L.draw ω) R (τ ω)))) (h R τ tgt)

theorem BindSound.toExtraction {A : Analysis L Reg session} {εbind : ℝ≥0∞}
    (h : A.BindSound εbind) : A.toExtraction.BindSound fun _ _ => εbind := fun R τ tgt =>
  le_of_eq_of_le (congrArg avg (funext fun ω =>
    atTgt_toExtraction (L.draw ω) R (τ ω) (tgt ω)
      fun u => A.linkFail (L.draw ω) R (τ ω) u (A.committed R τ))) (h R τ tgt)

end Analysis

theorem extraction_audit_le (A : ExtractionAnalysis L Reg session)
    {εks εbind : (R : Reg) → Cont L Reg session R → ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.BindSound εbind) (E : A.Tr → Finset (Fin n) → Prop)
    (σ : Strategy (audit L Reg session)) :
    prob (fun o => o.1 = true ∧ E (A.committedOf σ) o.2) (audit L Reg session) σ ≤
      prCoin (fun ω => E (A.committedOf σ) (L.draw ω) ∧
        Disjoint (L.draw ω) (A.incorrect (A.committedOf σ))) +
      εks (reg σ) (cont σ) + εbind (reg σ) (cont σ) := by
  classical
  obtain ⟨R, τ'⟩ := σ
  set τ : Cont L Reg session R := cont ⟨R, τ'⟩
  set X := A.committed R τ with hX
  show prob _ (audit L Reg session) ⟨R, τ'⟩ ≤
    prCoin (fun ω => E X (L.draw ω) ∧ Disjoint (L.draw ω) (A.incorrect X)) + εks R τ + εbind R τ
  let tgt : L.Ω → Option (Fin n) := fun ω =>
    if h : ∃ u, u ∈ L.draw ω ∧ u ∈ A.incorrect X then some (Classical.choose h) else none
  let ks : L.Ω → ℝ≥0∞ := fun ω => atTgt (L.draw ω) (tgt ω) (A.ks (L.draw ω) R (τ ω))
  let lk : L.Ω → ℝ≥0∞ := fun ω => atTgt (L.draw ω) (tgt ω) fun u => A.link (L.draw ω) R (τ ω) u X
  have step : ∀ ω, prob (fun ok => ok = true ∧ E X (L.draw ω)) (session (L.draw ω) R) (τ ω) ≤
      (if E X (L.draw ω) ∧ Disjoint (L.draw ω) (A.incorrect X) then 1 else 0) + ks ω + lk ω := by
    intro ω
    by_cases hd : Disjoint (L.draw ω) (A.incorrect X)
    · rw [prob_and_const]
      by_cases hE : E X (L.draw ω)
      · simp only [hE, hd, and_self, ite_true]
        exact (prob_le_one _ _ _).trans (le_add_right le_self_add)
      · simp only [hE, ite_false, false_and, zero_le]
    · have h : ∃ u, u ∈ L.draw ω ∧ u ∈ A.incorrect X := by
        obtain ⟨u, hu, hu'⟩ := Finset.not_disjoint_iff.1 hd
        exact ⟨u, hu, hu'⟩
      have ht : tgt ω = some (Classical.choose h) := by simp only [tgt, h, dite_true]
      obtain ⟨hmem, hw⟩ := Classical.choose_spec h
      have hc := A.cover (L.draw ω) R (τ ω) _ X hmem hw
      have hks : ks ω = A.ks (L.draw ω) R (τ ω) (Classical.choose h) := by
        simp [ks, atTgt, ht, hmem]
      have hlk : lk ω = A.link (L.draw ω) R (τ ω) (Classical.choose h) X := by
        simp [lk, atTgt, ht, hmem]
      calc prob (fun ok => ok = true ∧ E X (L.draw ω)) (session (L.draw ω) R) (τ ω)
          ≤ prob (· = true) (session (L.draw ω) R) (τ ω) := prob_mono (fun ok h => h.1) _ _
        _ ≤ ks ω + lk ω := by rw [hks, hlk]; exact hc
        _ ≤ _ := by rw [add_assoc]; exact le_add_left le_rfl
  calc prob (fun o => o.1 = true ∧ E X o.2) (audit L Reg session) ⟨R, τ'⟩
      = avg fun ω => prob (fun ok => ok = true ∧ E X (L.draw ω)) (session (L.draw ω) R) (τ ω) :=
        congrArg avg (funext fun ω => prob_map _ _ _ _)
    _ ≤ avg fun ω => (if E X (L.draw ω) ∧ Disjoint (L.draw ω) (A.incorrect X) then 1 else 0) +
        ks ω + lk ω := avg_mono step
    _ = _ := avg_add₃ _ _ _
    _ ≤ _ := add_le_add (add_le_add (avg_ite_eq_prCoin _).le (hks R τ tgt)) (hlink R τ tgt)

theorem extraction_audit_count (A : ExtractionAnalysis L Reg session)
    {εks εbind : (R : Reg) → Cont L Reg session R → ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.BindSound εbind) (K : ℕ) (σ : Strategy (audit L Reg session)) :
    prob (fun o => o.1 = true ∧ K ≤ (A.incorrect (A.committedOf σ)).card) (audit L Reg session) σ ≤
      L.miss K + εks (reg σ) (cont σ) + εbind (reg σ) (cont σ) := by
  refine (extraction_audit_le A hks hlink (fun X _ => K ≤ (A.incorrect X).card) σ).trans ?_
  gcongr
  by_cases hB : K ≤ (A.incorrect (A.committedOf σ)).card
  · exact (prCoin_mono fun ω h => h.2).trans (L.escape_le_miss hB)
  · exact (prCoin_mono (q := fun _ => False) fun ω h => absurd h.1 hB).trans
      (prCoin_false.le.trans (zero_le))

theorem extraction_audit_sampled (A : ExtractionAnalysis L Reg session)
    {εks εbind : (R : Reg) → Cont L Reg session R → ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.BindSound εbind) (σ : Strategy (audit L Reg session)) :
    prob (fun o => o.1 = true ∧ ¬ Disjoint (A.incorrect (A.committedOf σ)) o.2) (audit L Reg session) σ ≤
      εks (reg σ) (cont σ) + εbind (reg σ) (cont σ) := by
  refine (extraction_audit_le A hks hlink (fun X S => ¬ Disjoint (A.incorrect X) S) σ).trans ?_
  have : prCoin (fun ω => ¬ Disjoint (A.incorrect (A.committedOf σ)) (L.draw ω) ∧
      Disjoint (L.draw ω) (A.incorrect (A.committedOf σ))) = 0 :=
    le_antisymm ((prCoin_mono (q := fun _ => False) fun ω h => h.1 h.2.symm).trans
      prCoin_false.le) (zero_le)
  rw [this, zero_add]

end FlockSoundness.Audit
