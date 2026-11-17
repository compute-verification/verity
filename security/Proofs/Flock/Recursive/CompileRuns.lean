module

public import Proofs.Flock.Recursive.ClashFree
public import Proofs.Flock.Recursive.Target
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Game
import Proofs.Flock.Soundness.Refine.LiveCompiled

/-! The compile lemma on the runs of one prover (`compiled_runs_sound`). -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Game Model Accounting Exec StrictCR
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}

section Clash

open Classical in
theorem sum_iSup_le_iSup_pi {C : Type} [Fintype C] {ι : C → Type} (f : (c : C) → ι c → ℝ≥0∞)
    (hne : ∀ c, Nonempty (ι c)) : ∑ c, ⨆ i, f c i ≤ ⨆ s : (c : C) → ι c, ∑ c, f c (s c) := by
  let s₀ : (c : C) → ι c := fun c => Classical.choice (hne c)
  calc ∑ c, ⨆ i, f c i ≤ ∑ c, ⨆ s : (c : C) → ι c, f c (s c) :=
        Finset.sum_le_sum fun c _ => iSup_le fun i =>
          le_iSup_of_le (Function.update s₀ c i) (by rw [Function.update_self])
    _ = ⨆ s : (c : C) → ι c, ∑ c, f c (s c) := ENNReal.finsetSum_iSup (f := fun c s => f c (s c)) fun s s' =>
        ⟨fun c => if f c (s c) ≤ f c (s' c) then s' c else s c, fun c => by
          by_cases hle : f c (s c) ≤ f c (s' c)
          · simp only [if_pos hle]
            exact ⟨hle, le_rfl⟩
          · simp only [if_neg hle]
            exact ⟨le_rfl, le_of_lt (not_le.mp hle)⟩⟩

open Classical in
theorem value_le_iSup_prob {α : Type} (p : α → Prop) :
    ∀ {g : Game α}, AllInh g → value p g ≤ ⨆ s : Strategy g, prob p g s
  | .ret _, _ => le_iSup_of_le PUnit.unit le_rfl
  | .send M k, .send _ h => by
      rw [value_send]
      exact iSup_le fun m => (value_le_iSup_prob p (h m)).trans
        (iSup_le fun s' => le_iSup (fun s : Strategy (Game.send M k) => prob p _ s) ⟨m, s'⟩)
  | @Game.coin _ C iF iN k, .coin h => by
      rw [value_coin]
      calc ∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * value p (k c)
          ≤ ∑ c : C, ⨆ sc : Strategy (k c), (Fintype.card C : ℝ≥0∞)⁻¹ * prob p (k c) sc :=
            Finset.sum_le_sum fun c _ => by
              rw [← ENNReal.mul_iSup]
              exact mul_le_mul_right (value_le_iSup_prob p (h c)) _
        _ ≤ ⨆ s : (c : C) → Strategy (k c), ∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * prob p (k c) (s c) :=
            sum_iSup_le_iSup_pi (fun c sc => (Fintype.card C : ℝ≥0∞)⁻¹ * prob p (k c) sc) fun c => (h c).strategy
        _ = ⨆ s : Strategy (@Game.coin _ C iF iN k), prob p (@Game.coin _ C iF iN k) s := rfl

theorem value_le_of_forall_prob {α : Type} {p : α → Prop} {ε : ℝ≥0∞} {g : Game α} (hg : AllInh g)
    (h : ∀ s : Strategy g, prob p g s ≤ ε) : value p g ≤ ε :=
  (value_le_iSup_prob p hg).trans (iSup_le h)

theorem allInh_tableCL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) : AllInh (tableCL A H E Lf S sch hm ptLocal mPts) := by
  unfold tableCL
  repeat' (first
    | exact AllInh.interleave (allInh_repC A 0 S sch hm _) (allInh_repC A 1 S sch hm _)
    | exact AllInh.pure' _
    | exact AllInh.recv' _
    | exact AllInh.draw' _
    | refine AllInh.bind' ?_ fun _ => ?_)

theorem table_value_clashFree (A : Arith F K) (hA : A.Correct) [Inhabited D] [Inhabited Sl] (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule)
    (hsch : fast100 S.m = some sch) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (hlay : S.LinkLayout ptLocal mPts)
    (hno : ¬ ∃ z, S.Satisfies z) (W : OutCL F D Sl → Prop)
    (hW₀ : ∀ o, W o → ¬ SelfClashL H E Lf sch o)
    (hW₁ : ∀ o o', W o → W o' → ∀ p, ¬ Rewinding.ConflictAt (op₀L H E Lf sch o.cap₀) (o, o') p)
    (hW₂ : ∀ o o', W o → W o' → ∀ i < 2, ∀ ℓ, 1 ≤ ℓ → ∀ cap,
      (if i = 0 then o.pend.1 else o.pend.2).caps[ℓ - 1]? = some cap →
      ∀ p, ¬ Rewinding.ConflictAt (opRL H E Lf sch i ℓ cap) (o, o') p) :
    value (fun o : OutCL F D Sl => o.accepted = true ∧ W o) (tableCL A H E Lf S sch hm ptLocal mPts) ≤
      ENNReal.ofReal (tableError sch (S.shape mPts)) :=
  value_le_of_forall_prob (allInh_tableCL A H E Lf S sch hm ptLocal mPts) fun σ =>
    table_prob_clashFree A hA H E Lf S sch hsch hm ptLocal mPts hlay hno W hW₀ hW₁ hW₂ σ

end Clash

variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

open Classical in
theorem sched_prob_le_value {α : Type} (g : Game α) (p : α → Prop)
    (s : Strategy (rounds (List (SMsg T)) ((k : Fin r) → CT k) (r + 1) .ret)) :
    prob (fun t => ∃ a, PlaysT g (schedSteps T CT 0 t) a ∧ p a)
      (rounds (List (SMsg T)) ((k : Fin r) → CT k) (r + 1) .ret) s ≤ value p g := by
  refine le_trans (le_of_eq ?_) (leafAvg_plays_le T CT p (r + 1) 0 (by omega) .ret s g)
  exact prob_rounds _ (r + 1) _ s

def RunOut (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Schedule) (M : Hidden cls)
    (G : List (List (SMsg T) × ((k : Fin r) → CT k)) → Prop)
    (s : Strategy (flockInnerCL T CT A H E Lf cls regs sch).game) (o : OutCL F D Sl) : Prop :=
  ∃ t, Game.Outcome (flockInnerCL T CT A H E Lf cls regs sch).game s t ∧ G t ∧
    PlaysT (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) (schedSteps T CT 0 t) o

theorem compiled_runs_sound (A : Arith F K) (hA : A.Correct) [Inhabited D] [Inhabited Sl]
    (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass)
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Schedule) (hsch : fast100 cls.m = some sch)
    (hlay : ∀ M : Hidden cls, (M.toStatement regs).LinkLayout cls.ptLocal cls.mPts)
    (Cl : OutCL F D Sl → OutCL F D Sl → Prop)
    (hself : ∀ o o', SelfClashL H E Lf sch o → Cl o o')
    (h₀ : ∀ o o' p, Rewinding.ConflictAt (op₀L H E Lf sch o.cap₀) (o, o') p → Cl o o')
    (hR : ∀ o o', ∀ i < 2, ∀ ℓ, 1 ≤ ℓ → ∀ cap, (if i = 0 then o.pend.1 else o.pend.2).caps[ℓ - 1]? = some cap →
      ∀ p, Rewinding.ConflictAt (opRL H E Lf sch i ℓ cap) (o, o') p → Cl o o')
    (M : Hidden cls) (hno : ¬ (flockInnerCL T CT A H E Lf cls regs sch).holds M)
    (G : List (List (SMsg T) × ((k : Fin r) → CT k)) → Prop)
    (s : Strategy (flockInnerCL T CT A H E Lf cls regs sch).game) :
    (∃ t t' o o', Game.Outcome (flockInnerCL T CT A H E Lf cls regs sch).game s t ∧
      Game.Outcome (flockInnerCL T CT A H E Lf cls regs sch).game s t' ∧ G t ∧ G t' ∧
      PlaysT (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) (schedSteps T CT 0 t) o ∧
      PlaysT (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) (schedSteps T CT 0 t') o' ∧
      Cl o o') ∨
    prob (fun t => (flockInnerCL T CT A H E Lf cls regs sch).accepts M t ∧ G t)
        (flockInnerCL T CT A H E Lf cls regs sch).game s ≤
      ENNReal.ofReal (tableError sch ⟨cls.kLog, regs.length, cls.mPts⟩) := by
  by_cases hc : ∃ t t' o o', Game.Outcome (flockInnerCL T CT A H E Lf cls regs sch).game s t ∧
      Game.Outcome (flockInnerCL T CT A H E Lf cls regs sch).game s t' ∧ G t ∧ G t' ∧
      PlaysT (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) (schedSteps T CT 0 t) o ∧
      PlaysT (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) (schedSteps T CT 0 t') o' ∧
      Cl o o'
  · exact Or.inl hc
  right
  have hclean : ∀ o o', RunOut T CT A H E Lf cls regs sch M G s o → RunOut T CT A H E Lf cls regs sch M G s o' →
      ¬ Cl o o' := by
    rintro o o' ⟨t, ht, hG, ho⟩ ⟨t', ht', hG', ho'⟩ hcl
    exact hc ⟨t, t', o, o', ht, ht', hG, hG', ho, ho', hcl⟩
  have hval := table_value_clashFree A hA H E Lf (M.toStatement regs) sch hsch cls.hm cls.ptLocal cls.mPts (hlay M) hno
    (RunOut T CT A H E Lf cls regs sch M G s)
    (fun o ho h => hclean o o ho ho (hself o o h))
    (fun o o' ho ho' p h => hclean o o' ho ho' (h₀ o o' p h))
    (fun o o' ho ho' i hi ℓ hℓ cap hcap p h => hclean o o' ho ho' (hR o o' i hi ℓ hℓ cap hcap p h))
  calc prob (fun t => (flockInnerCL T CT A H E Lf cls regs sch).accepts M t ∧ G t)
        (flockInnerCL T CT A H E Lf cls regs sch).game s
      ≤ prob (fun t => ∃ a, PlaysT (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts)
          (schedSteps T CT 0 t) a ∧ (a.accepted = true ∧ RunOut T CT A H E Lf cls regs sch M G s a))
        (flockInnerCL T CT A H E Lf cls regs sch).game s := by
        refine prob_mono_outcome _ s fun t ht h => ?_
        obtain ⟨⟨a, ha, hacc⟩, hG⟩ := h
        exact ⟨a, ha, hacc, t, ht, hG, ha⟩
    _ ≤ value (fun a : OutCL F D Sl => a.accepted = true ∧ RunOut T CT A H E Lf cls regs sch M G s a)
          (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) :=
        sched_prob_le_value T CT (tableCL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts)
          (fun a : OutCL F D Sl => a.accepted = true ∧ RunOut T CT A H E Lf cls regs sch M G s a) s
    _ ≤ _ := hval

end FlockSoundness.Discharge.Recursion
