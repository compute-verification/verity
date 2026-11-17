module

public import Proofs.Flock.Soundness.Audit.Work
import Mathlib.Algebra.Order.Algebra
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
import Proofs.Flock.Soundness.Game

/-! Closure samples: each sampled proof block proved together with its closure, the units its credit depends on. -/

@[expose] public section

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal

namespace SamplingDesign

variable {n m : ℕ}

/-- The units made unsound by the incorrect set `B`: those whose closure meets it. -/
def unsoundTiles (cl : Fin n → Finset (Fin n)) (B : Finset (Fin n)) : Finset (Fin n) :=
  univ.filter fun t => ¬ Disjoint (cl t) B

/-- **Closure samples**: `L`'s sample, each sampled proof block proved together with its closure. -/
def closure (L : SamplingDesign n) (cl : Fin n → Finset (Fin n)) : SamplingDesign n :=
  L.widen fun S => S ∪ S.biUnion cl

theorem mem_closure_sample (L : SamplingDesign n) (cl : Fin n → Finset (Fin n)) (ω : (L.closure cl).Ω) (u : Fin n) :
    u ∈ (L.closure cl).draw ω ↔ u ∈ L.draw ω ∨ ∃ t ∈ L.draw ω, u ∈ cl t := by
  show u ∈ L.draw ω ∪ (L.draw ω).biUnion cl ↔ _
  rw [mem_union, mem_biUnion]

theorem closure_escape (L : SamplingDesign n) (cl : Fin n → Finset (Fin n)) (B : Finset (Fin n)) :
    (L.closure cl).missProb B = L.missProb (B ∪ unsoundTiles cl B) := by
  show prCoin (fun ω : L.Ω => Disjoint (L.draw ω ∪ (L.draw ω).biUnion cl) B) =
    prCoin (fun ω : L.Ω => Disjoint (L.draw ω) (B ∪ unsoundTiles cl B))
  congr 1
  funext ω
  apply propext
  rw [disjoint_union_left, disjoint_biUnion_left, disjoint_union_right]
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨h1, disjoint_left.2 fun t ht htU => (mem_filter.1 htU).2 (h2 t ht)⟩
  · rintro ⟨h1, h2⟩
    refine ⟨h1, fun t ht => ?_⟩
    by_contra hc
    exact disjoint_left.1 h2 ht (mem_filter.2 ⟨mem_univ t, hc⟩)

/-- The work of the incorrect proof blocks `B` and of the units they make unsound. -/
abbrev unsoundWork (σ : Fin n → Fin m) (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) (B : Finset (Fin n)) : ℕ :=
  workOf σ w (B ∪ unsoundTiles cl B)

/-- A closure map need not list a unit in its own closure, so the incorrect proof blocks' own work is counted
explicitly. -/
theorem workOf_le_unsoundWork (σ : Fin n → Fin m) (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) (B : Finset (Fin n)) :
    workOf σ w B ≤ unsoundWork σ w cl B :=
  sum_le_sum_of_subset subset_union_left

def harm (σ : Fin n → Fin m) (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) (u : Fin n) : ℕ :=
  workOf σ w (insert u (univ.filter fun t => u ∈ cl t))

theorem harm_le_unsoundWork (σ : Fin n → Fin m) (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) {B : Finset (Fin n)}
    {u : Fin n} (hu : u ∈ B) : harm σ w cl u ≤ unsoundWork σ w cl B := by
  unfold harm unsoundWork unsoundTiles workOf
  refine sum_le_sum_of_subset fun t ht => ?_
  rcases mem_insert.1 ht with rfl | ht
  · exact mem_union_left _ hu
  · exact mem_union_right _ (mem_filter.2 ⟨mem_univ t, not_disjoint_iff.2 ⟨u, (mem_filter.1 ht).2, hu⟩⟩)

theorem unsoundWork_le_harm (σ : Fin n → Fin m) (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) (B : Finset (Fin n)) :
    unsoundWork σ w cl B ≤ ∑ u ∈ B, harm σ w cl u := by
  classical
  have e : ∀ S : Finset (Fin n), workOf σ w S = ∑ t, if t ∈ S then w (σ t) else 0 := fun S => by
    rw [sum_ite_mem, univ_inter, workOf]
  unfold unsoundWork harm
  simp only [e]
  rw [sum_comm]
  refine sum_le_sum fun t _ => ?_
  split_ifs with h
  · obtain ⟨u, hu, htu⟩ : ∃ u ∈ B, t ∈ insert u (univ.filter fun t => u ∈ cl t) := by
      rcases mem_union.1 h with htB | htU
      · exact ⟨t, htB, mem_insert_self _ _⟩
      · obtain ⟨u, hu, huB⟩ := not_disjoint_iff.1 (mem_filter.1 htU).2
        exact ⟨u, huB, mem_insert_of_mem (mem_filter.2 ⟨mem_univ t, hu⟩)⟩
    calc w (σ t) = (if t ∈ insert u (univ.filter fun t => u ∈ cl t) then w (σ t) else 0) := by simp only [htu, ite_true]
      _ ≤ ∑ u ∈ B, if t ∈ insert u (univ.filter fun t => u ∈ cl t) then w (σ t) else 0 :=
          single_le_sum (f := fun u => if t ∈ insert u (univ.filter fun t => u ∈ cl t) then w (σ t) else 0)
            (fun _ _ => Nat.zero_le _) hu
  · exact Nat.zero_le _

theorem unsoundWork_singleton (σ : Fin n → Fin m) (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) (u : Fin n) :
    unsoundWork σ w cl {u} = harm σ w cl u := by
  unfold unsoundWork unsoundTiles harm
  congr 1
  ext t
  simp [disjoint_singleton_right]

theorem closure_escape_le_work (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) (cl : Fin n → Finset (Fin n))
    (hf : ∀ s, 1 ≤ f s) (hW : 0 < totalWork σ w) {B : Finset (Fin n)} {T : ℕ} (hT : T ≤ unsoundWork σ w cl B) :
    ((work σ w f K).closure cl).missProb B ≤ (((totalWork σ w - T : ℕ) : ℝ≥0∞) / (totalWork σ w : ℝ≥0∞)) ^ K := by
  rw [closure_escape]
  exact work_escape_le_of_le σ w f K hf hW hT

end SamplingDesign

section Audits

variable {n m : ℕ} {σ : Fin n → Fin m} {w : Fin m → ℕ} {f : Fin m → ℕ} {K : ℕ} {cl : Fin n → Finset (Fin n)}
  {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

theorem audit_work_closure (A : Analysis ((SamplingDesign.work σ w f K).closure cl) Reg session) {εks εbind : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.BindSound εbind) (hf : ∀ s, 1 ≤ f s) (hW : 0 < SamplingDesign.totalWork σ w) (T : ℕ)
    (σ' : Strategy (audit ((SamplingDesign.work σ w f K).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ T ≤ SamplingDesign.unsoundWork σ w cl (A.incorrect (A.committedOf σ')))
        (audit ((SamplingDesign.work σ w f K).closure cl) Reg session) σ' ≤
      (((SamplingDesign.totalWork σ w - T : ℕ) : ℝ≥0∞) / (SamplingDesign.totalWork σ w : ℝ≥0∞)) ^ K + εks + εbind := by
  refine (audit_profile A hks hlink (fun B => T ≤ SamplingDesign.unsoundWork σ w cl B) σ').trans ?_
  gcongr
  exact iSup₂_le fun B hB => SamplingDesign.closure_escape_le_work σ w f K cl hf hW hB

theorem extraction_audit_work_closure (A : ExtractionAnalysis ((SamplingDesign.work σ w f K).closure cl) Reg session)
    {εks εbind : (R : Reg) → Cont ((SamplingDesign.work σ w f K).closure cl) Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.BindSound εbind) (hf : ∀ s, 1 ≤ f s) (hW : 0 < SamplingDesign.totalWork σ w) (T : ℕ)
    (σ' : Strategy (audit ((SamplingDesign.work σ w f K).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ T ≤ SamplingDesign.unsoundWork σ w cl (A.incorrect (A.committedOf σ')))
        (audit ((SamplingDesign.work σ w f K).closure cl) Reg session) σ' ≤
      (((SamplingDesign.totalWork σ w - T : ℕ) : ℝ≥0∞) / (SamplingDesign.totalWork σ w : ℝ≥0∞)) ^ K +
        εks (reg σ') (cont σ') + εbind (reg σ') (cont σ') := by
  refine (extraction_audit_le A hks hlink (fun X _ => T ≤ SamplingDesign.unsoundWork σ w cl (A.incorrect X)) σ').trans ?_
  gcongr
  by_cases hB : T ≤ SamplingDesign.unsoundWork σ w cl (A.incorrect (A.committedOf σ'))
  · exact (prCoin_mono fun ω h => h.2).trans (SamplingDesign.closure_escape_le_work σ w f K cl hf hW hB)
  · exact (prCoin_mono (q := fun _ => False) fun ω h => hB h.1).trans (prCoin_false.le.trans zero_le)

theorem audit_work_closure_harm (A : Analysis ((SamplingDesign.work σ w f K).closure cl) Reg session) {εks εbind : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.BindSound εbind) (hf : ∀ s, 1 ≤ f s) (hW : 0 < SamplingDesign.totalWork σ w) (T : ℕ)
    (σ' : Strategy (audit ((SamplingDesign.work σ w f K).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ ∃ u ∈ A.incorrect (A.committedOf σ'), T ≤ SamplingDesign.harm σ w cl u)
        (audit ((SamplingDesign.work σ w f K).closure cl) Reg session) σ' ≤
      (((SamplingDesign.totalWork σ w - T : ℕ) : ℝ≥0∞) / (SamplingDesign.totalWork σ w : ℝ≥0∞)) ^ K + εks + εbind := by
  refine le_trans (prob_mono (fun o ho => ⟨ho.1, ?_⟩) _ _) (audit_work_closure A hks hlink hf hW T σ')
  obtain ⟨u, hu, hT⟩ := ho.2
  exact hT.trans (SamplingDesign.harm_le_unsoundWork σ w cl hu)

theorem audit_work_closure_of_record (A : Analysis ((SamplingDesign.work σ w f 27713).closure cl) Reg session) {εks εbind : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.BindSound εbind) (hf : ∀ s, 1 ≤ f s) (hW : 0 < SamplingDesign.totalWork σ w)
    (σ' : Strategy (audit ((SamplingDesign.work σ w f 27713).closure cl) Reg session)) :
    prob (fun o => o.1 = true ∧ SamplingDesign.totalWork σ w ≤ 1000 * SamplingDesign.unsoundWork σ w cl (A.incorrect (A.committedOf σ')))
        (audit ((SamplingDesign.work σ w f 27713).closure cl) Reg session) σ' ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ + εks + εbind := by
  refine (audit_profile A hks hlink (fun B => SamplingDesign.totalWork σ w ≤ 1000 * SamplingDesign.unsoundWork σ w cl B) σ').trans ?_
  gcongr
  refine iSup₂_le fun B hB => ?_
  rw [SamplingDesign.closure_escape]
  exact SamplingDesign.work_escape_of_record σ w f hf hW hB

end Audits

end FlockSoundness.Audit
