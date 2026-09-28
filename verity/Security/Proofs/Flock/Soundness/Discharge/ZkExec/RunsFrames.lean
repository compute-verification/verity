import Proofs.Flock.Soundness.Discharge.ZkExec.RunsSeams
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsLig
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsReplay

/-!
# `repProg` emits a `--zk` rep's frames

`zk_rep_frames`: a run of `repProg` on retained rounds that consumes every round consumed one round per chunk of
`repFramesZ`, each round retaining its chunk's bytes. The domain, then `replay_emitted`, the non-ZK session's
`ringSwitch_emitted` on the two masked claims and the extras, `zkLigerito_frames` and `verifyInner_emitted` on the ten
constraints: the replay's three, the ring switch's two, the lane phase's two (`ligeritoConstraints_size`) and the
triple's three.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Transcript

theorem size_two_append {α : Type} (a b : α) (xs : Array α) : (#[a, b] ++ xs).size = 2 + xs.size := by
  simp [Nat.add_comm]

theorem size_inner_cons {α : Type} (xs : Array α) (a b : α) (ys : Array α) (hx : xs.size = 3)
    (hy : ys.size = 2) :
    (xs ++ #[a, b] ++ ys).size + 3 = 10 := by
  simp [hx, hy]

theorem zk_rep_frames : ZkRepFrames := by
  intro st sch tPad r extras zp rounds dom tr hsch hrun hk hret
  obtain ⟨h25, -, -, hk0, -, hdim, hlv, -, -⟩ := zk_schedule_seams _ _ _ hsch
  unfold repProg at hrun
  rw [StateT.run] at hrun
  try dsimp only at hrun
  obtain ⟨rp, t₁, h₁, hrun⟩ := bind_ok hrun
  try dsimp only at hrun
  obtain ⟨⟨T, sws⟩, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨rc, t₃, h₃, hrun⟩ := bind_ok hrun
  cases hv : Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rc zp.proof.ligerito zp.level0 with
  | ok v =>
    rw [hv] at hrun
    try dsimp only at hrun
    obtain ⟨lg, t₄, h₄, hrun'⟩ := bind_ok hrun
    clear hrun
    obtain ⟨rfl, rfl⟩ := pure_ok h₄
    refine ⟨T, ?_⟩
    have e₀ : Emitted ⟨rounds, 0, ByteArray.empty⟩ (Flock.Transcript.init rounds dom) [.dom dom] :=
      emitted_absorb rfl rfl rfl (by rw [init_enc]; rfl)
    have r₀ : AllRetained (Flock.Transcript.init rounds dom).rounds := hret
    obtain ⟨hcs, e₁⟩ := replay_emitted st _ zp.proof zp.inner.root (by omega) r₀ h₁
    have e₂ := ringSwitch_emitted _ _ _ (by rw [size_two_append]; omega) _ _ _ (e₁.retained r₀) h₂
    rw [size_two_append] at e₂
    have e₃ := zkLigerito_frames _ sch _ _ _ _ _ hdim hlv hk0 (e₂.retained (e₁.retained r₀)) h₃
    have e₄ := verifyInner_emitted _ _ _ _ (e₃.retained (e₂.retained (e₁.retained r₀))) hrun'
    rw [size_inner_cons _ _ _ _ hcs (ligeritoConstraints_size hv)] at e₄
    have e := e₀.trans (e₁.trans (e₂.trans (e₃.trans e₄)))
    have e' : Emitted ⟨rounds, 0, ByteArray.empty⟩ tr (repFramesZ st sch dom (2 + extras.size) zp T) := by
      simpa only [repFramesZ, restFramesZ, List.append_assoc, List.cons_append, List.nil_append,
        List.singleton_append] using e
    obtain ⟨-, hk', hR, -⟩ := e'
    refine ⟨by rw [← hk, hk']; simp, fun k hk2 => ?_⟩
    have hk3 : k < (chunks (repFramesZ st sch dom (2 + extras.size) zp T)).1.length := by
      rw [← hk, hk'] at hk2; simpa using hk2
    have := hR k hk3
    simpa using this
  | error e =>
    rw [hv] at hrun
    try dsimp only at hrun
    obtain ⟨_, _, h₄, _⟩ := bind_ok hrun
    cases h₄

end FlockSoundness.Discharge.ZkExec
