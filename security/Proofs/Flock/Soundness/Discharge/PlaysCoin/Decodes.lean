module

public import Proofs.Flock.Soundness.Discharge.PlaysCoin.Coins
public import Proofs.Flock.Soundness.Discharge.PlaysCoin.Statements
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Discharge.PlaysCoin.Tape

/-! `Decodes` from the reps' runs. -/

@[expose] public section

namespace FlockSoundness.Discharge.PlaysCoin

open Flock FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge.Exec Model Accounting Audit
  Audit.Partition

local notation "𝔽" => FlockLevel3.GF128

theorem length_steps : ∀ ts : List Step,
    ts.length = (ts.filterMap Sum.getLeft?).length + (ts.filterMap Sum.getRight?).length
  | [] => rfl
  | .inl _ :: ts => by
      simp only [List.filterMap_cons, Sum.getLeft?_inl, Sum.getRight?_inl, List.length_cons, length_steps ts]
      omega
  | .inr _ :: ts => by
      simp only [List.filterMap_cons, Sum.getLeft?_inr, Sum.getRight?_inr, List.length_cons, length_steps ts]
      omega

theorem playsT_of_repRun {P : Type} {g : Game P} {σ : List Exec.Kind} {dec : ℕ → ByteArray → List Msg} {rs : List Round}
    {tail : List Msg} {p : P} (hs : Spine g σ) (h : RepRun g σ dec rs tail p) :
    PlaysT g (tapeFrom dec (fun _ => true) 0 rs ++ tail.map .inl) p := by
  obtain ⟨hwf, ms, hp, hcut⟩ := h
  obtain ⟨ts, hpt, h1, h2⟩ := coinsOfRounds hwf hp
  have hk : ts.map kindOf = σ.take (ms.length + rs.length) := by
    have e := List.prefix_iff_eq_take.mp (Spine.prefix hpt hs)
    rw [List.length_map, length_steps ts, h1, h2, List.length_map] at e
    exact e
  have e : ts = tapeFrom dec (fun _ => true) 0 rs ++ tail.map .inl :=
    tapeOfGroups dec 0 rs tail ts h2 (by rw [hk, h1]; exact hcut)
  exact e ▸ hpt

theorem playsT_of_repRunAt {P : Type} {g : Game P} {σ : List Exec.Kind} {dec : ℕ → ByteArray → List Msg}
    {rs : List Round} {tail : List Msg} {p : P} (hs : SpineAt g σ p) (h : RepRun g σ dec rs tail p) :
    PlaysT g (tapeFrom dec (fun _ => true) 0 rs ++ tail.map .inl) p := by
  obtain ⟨hwf, ms, hp, hcut⟩ := h
  obtain ⟨ts, hpt, h1, h2⟩ := coinsOfRounds hwf hp
  have hk : ts.map kindOf = σ.take (ms.length + rs.length) := by
    have e := List.prefix_iff_eq_take.mp (hs ts hpt)
    rw [List.length_map, length_steps ts, h1, h2, List.length_map] at e
    exact e
  have e : ts = tapeFrom dec (fun _ => true) 0 rs ++ tail.map .inl :=
    tapeOfGroups dec 0 rs tail ts h2 (by rw [hk, h1]; exact hcut)
  exact e ▸ hpt

theorem decodesOfRunsAt : DecodesOfRunsAt := by
  intro P OutT Cp Op _ mPts out pL pg opn dec fin capOf G h o ho
  obtain ⟨pend, σ₀, σ₁, hs₀, h₀, hs₁, h₁, hv⟩ := h o ho
  exact ⟨pend, playsT_of_repRunAt hs₀ h₀, playsT_of_repRunAt hs₁ h₁, hv⟩

theorem decodesOfRuns : DecodesOfRuns := by
  intro P OutT Cp Op _ mPts out pL pg opn dec fin capOf G σ hσ h o ho
  obtain ⟨pend, h₀, h₁, hv⟩ := h o ho
  exact ⟨pend, playsT_of_repRun (hσ o.pts 0) h₀, playsT_of_repRun (hσ o.pts 1) h₁, hv⟩

theorem repsOf_spine (D : Type) {mPts : ℕ} (T : TabSpec mPts) (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) :
    Spine (repsOf (D := D) execArith T pts r) (repSpine D T pts) :=
  repC_spine execArith r.val T.S T.sch T.hm _

theorem decodesAllOfRuns : DecodesAllOfRuns := by
  intro D Sl _ _ _ mPts n Reg H E Lf T I wr opn dec fin capOf h S R
  exact decodesOfRuns _ _ _ (opn S R) (dec S R) (fin S R) (capOf S R) _ (fun pts _ => repSpine D (T S R) pts)
    (repsOf_spine D (T S R)) (h S R)

theorem decodesOnOfRuns : DecodesOnOfRuns := by
  intro D Sl _ _ _ mPts n Reg H E Lf T I wr opn dec fin capOf L h ω R
  exact decodesOfRuns _ _ _ (opn (L.draw ω) R) (dec (L.draw ω) R) (fin (L.draw ω) R) (capOf (L.draw ω) R) _
    (fun pts _ => repSpine D (T (L.draw ω) R) pts) (repsOf_spine D (T (L.draw ω) R)) (h ω R)

end FlockSoundness.Discharge.PlaysCoin
