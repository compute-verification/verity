import Proofs.Flock.Soundness.Discharge.PlaysCoin.Coins
import Proofs.Flock.Soundness.Discharge.PlaysCoin.Tape

/-!
# `Decodes` from the reps' runs

`playsT_of_repRun`: a rep's `RepRun` along a spine of its game is a run on the decoded tape. The run on two lists is a
run on one tape whose coin steps are the rounds' `coinMsg true` (`coinsOfRounds`). That tape's kinds are the spine's,
up to its length (`Spine.prefix`), so its messages cut at its coins are the `RepRun`'s groups, and the tape is the
decoded one (`tapeOfGroups`). So `decodesOfRuns`, `decodesAllOfRuns` and `decodesOnOfRuns`; the reps' spine is
`repShape` (`repC_spine`). `decodesOfRunsAt` reads the spine only along the runs that end where the `RepRun` ends
(`playsT_of_repRunAt`).
-/

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

/-- **A rep's run on its decoded tape.** A `RepRun` of `g` along a spine of `g` is a run of `g` on the tape the decoder
reads off the rounds, with `one := fun _ => true`, then the tail. -/
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

/-- **A rep's run on its decoded tape, at its end.** `playsT_of_repRun` with the spine needed only along the runs that
end in `p`. -/
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

/-- The reps of table `T` have the spine `repSpine`. -/
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
