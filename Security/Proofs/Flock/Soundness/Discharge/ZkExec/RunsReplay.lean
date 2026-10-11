import Proofs.Flock.Soundness.Discharge.ZkExec.RunsStatements

/-!
# The masked replay and the inner proof, as frames

* `replay_emitted`: a `Zk.replay` run on retained rounds emits round 0's binding with the pads' label and root
  (`zcHeadZ`), the masked zerocheck's rest (`zcTailZ`) and the lincheck (`lcFrames`), and returns three constraints.
* `verifyInner_emitted`: a `Zk.verifyInner` run on `n` constraints emits `innerFr` on `n + 3` (the triple's three); its
  query loop emits nothing.
* `ligeritoConstraints_size`: level 0's lane phase gives two constraints.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Transcript

theorem flatMap_nil_const {α γ : Type} (l : List α) : l.flatMap (fun _ => ([] : List γ)) = [] := by
  simp

theorem replay_emitted (st : Setup) (lay : Zk.PadLayout) (p : Proof) (root : ByteArray) (hm : 14 ≤ st.m)
    {t t' : Flock.Transcript} {rp : Zk.Replayed} (hret : AllRetained t.rounds)
    (h : Zk.replay st lay p root t = .ok (rp, t')) :
    rp.cons.size = 3 ∧ Emitted t t' (zcHeadZ st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) root ++
      (zcTailZ st.m p.zerocheck ++ lcFrames st.kLog p.lincheck)) := by
  unfold Zk.replay at h
  obtain ⟨_, t₁, h₁, h⟩ := bind_ok h
  have e₁ := label_emitted h₁
  obtain ⟨_, t₂, h₂, h⟩ := bind_ok h
  have e₂ := absorbBytes_emitted h₂
  obtain ⟨_, t₃, h₃, h⟩ := bind_ok h
  have e₃ := absorbBytes_emitted h₃
  obtain ⟨_, t₄, h₄, h⟩ := bind_ok h
  have e₄ := label_emitted h₄
  obtain ⟨_, t₄', h₄', h⟩ := bind_ok h
  have e₄' := label_emitted h₄'
  obtain ⟨_, t₄'', h₄'', h⟩ := bind_ok h
  have e₄'' := absorbBytes_emitted h₄''
  have r₄ := e₄''.retained (e₄'.retained (e₄.retained (e₃.retained (e₂.retained (e₁.retained hret)))))
  obtain ⟨_, t₅, h₅, h⟩ := bind_ok h
  obtain ⟨e₅, -⟩ := sampleVec_emitted (by decide) h₅ r₄
  obtain ⟨_, t₆, h₆, h⟩ := bind_ok h
  obtain ⟨e₆, -⟩ := sampleVec_emitted (by omega) h₆ (e₅.retained r₄)
  obtain ⟨_, t₇, h₇, h⟩ := bind_ok h
  have e₇ := absorbSlice_emitted h₇
  obtain ⟨_, t₈, h₈, h⟩ := bind_ok h
  have e₈ := absorbSlice_emitted h₈
  obtain ⟨zz, t₉, h₉, h⟩ := bind_ok h
  have r₈ := e₈.retained (e₇.retained (e₆.retained (e₅.retained r₄)))
  obtain ⟨e₉, -⟩ := sample_emitted h₉ r₈
  obtain ⟨s, t₁₀, hloop, h⟩ := bind_ok h
  rw [forIn_range_eq] at hloop
  have eL := forIn_emitted _ (fun i => [.one (p.zerocheck.rounds[i]!).1, .one (p.zerocheck.rounds[i]!).2, .sq 1 1])
    (by
      intro i st tt b tt' hrt hb
      obtain ⟨_, u₁, k₁, hb⟩ := bind_ok hb
      have f₁ := absorbF_emitted k₁
      obtain ⟨_, u₂, k₂, hb⟩ := bind_ok hb
      have f₂ := absorbF_emitted k₂
      obtain ⟨_, u₃, k₃, hb⟩ := bind_ok hb
      obtain ⟨f₃, -⟩ := sample_emitted k₃ (f₂.retained (f₁.retained hrt))
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      exact ⟨_, rfl, f₁.trans (f₂.trans f₃)⟩) (st.m - 6) 0 _ _ t₁₀ s (e₉.retained r₈) hloop
  have r₁₀ := eL.retained (e₉.retained r₈)
  obtain ⟨_, t₁₁, h₁₁, h⟩ := bind_ok h
  have e₁₁ := absorbF_emitted h₁₁
  obtain ⟨_, t₁₂, h₁₂, h⟩ := bind_ok h
  have e₁₂ := absorbF_emitted h₁₂
  obtain ⟨_, t₁₃, h₁₃, h⟩ := bind_ok h
  have e₁₃ := label_emitted h₁₃
  have r₁₃ := e₁₃.retained (e₁₂.retained (e₁₁.retained r₁₀))
  obtain ⟨α₀, t₁₄, h₁₄, h⟩ := bind_ok h
  obtain ⟨e₁₄, -⟩ := sample_emitted h₁₄ r₁₃
  obtain ⟨β₀, t₁₅, h₁₅, h⟩ := bind_ok h
  obtain ⟨e₁₅, -⟩ := sample_emitted h₁₅ (e₁₄.retained r₁₃)
  obtain ⟨s₂, t₁₆, hloop₂, h⟩ := bind_ok h
  rw [forIn_range_eq] at hloop₂
  have eL₂ := forIn_emitted _ (fun j => [.one (p.lincheck.rounds[j]!).1, .one (p.lincheck.rounds[j]!).2, .sq 1 1])
    (by
      intro j st tt b tt' hrt hb
      obtain ⟨_, u₁, k₁, hb⟩ := bind_ok hb
      have f₁ := absorbF_emitted k₁
      obtain ⟨_, u₂, k₂, hb⟩ := bind_ok hb
      have f₂ := absorbF_emitted k₂
      obtain ⟨_, u₃, k₃, hb⟩ := bind_ok hb
      obtain ⟨f₃, -⟩ := sample_emitted k₃ (f₂.retained (f₁.retained hrt))
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      exact ⟨_, rfl, f₁.trans (f₂.trans f₃)⟩) (st.kLog - 6) 0 _ _ t₁₆ s₂
    (e₁₅.retained (e₁₄.retained r₁₃)) hloop₂
  have r₁₆ := eL₂.retained (e₁₅.retained (e₁₄.retained r₁₃))
  dsimp only at h
  cases hf : st.fold.folded α₀ β₀ (skipWeights zz) (s.2.extract 0 (st.kLog - 6)) s₂.2 with
  | ok v =>
    rw [hf] at h
    try dsimp only at h
    obtain ⟨combv, t₁₇, h₁₇, h'⟩ := bind_ok h
    clear h
    obtain ⟨rfl, rfl⟩ := pure_ok h₁₇
    obtain ⟨-, h⟩ := check_bind h'
    obtain ⟨_, t₁₈, h₁₈, h⟩ := bind_ok h
    have e₁₈ := absorbSlice_emitted h₁₈
    obtain ⟨_, t₁₉, h₁₉, h⟩ := bind_ok h
    obtain ⟨e₁₉, -⟩ := sample_emitted h₁₉ (e₁₈.retained r₁₆)
    obtain ⟨rfl, rfl⟩ := pure_ok h
    refine ⟨by simp, ?_⟩
    have := e₁.trans (e₂.trans (e₃.trans (e₄.trans (e₄'.trans (e₄''.trans (e₅.trans (e₆.trans (e₇.trans (e₈.trans
      (e₉.trans (eL.trans (e₁₁.trans (e₁₂.trans (e₁₃.trans (e₁₄.trans (e₁₅.trans (eL₂.trans
        (e₁₈.trans e₁₉))))))))))))))))))
    simpa [zcHeadZ, zcTailZ, lcFrames] using this
  | error e =>
    rw [hf] at h
    try dsimp only at h
    obtain ⟨_, _, h₁₇, _⟩ := bind_ok h
    cases h₁₇

theorem verifyInner_emitted (lay : Zk.PadLayout) (pr : Zk.InnerProof) (finalC : F128)
    (cons : Array (String × Zk.Lin)) {t t' : Flock.Transcript} (hret : AllRetained t.rounds)
    (h : Zk.verifyInner lay pr finalC cons t = .ok ((), t')) :
    Emitted t t' (innerFr pr finalC (cons.size + 3)) := by
  unfold Zk.verifyInner at h
  dsimp only at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨_, t₁, h₁, h⟩ := bind_ok h
  have e₁ := label_emitted h₁
  obtain ⟨_, t₂, h₂, h⟩ := bind_ok h
  have e₂ := absorbF_emitted h₂
  have r₂ := e₂.retained (e₁.retained hret)
  obtain ⟨eps, t₃, h₃, h⟩ := bind_ok h
  obtain ⟨e₃, -⟩ := sample_emitted h₃ r₂
  obtain ⟨_, t₄, h₄, h⟩ := bind_ok h
  have e₄ := absorbSlice_emitted h₄
  have r₄ := e₄.retained (e₃.retained r₂)
  obtain ⟨gs, t₅, h₅, h⟩ := bind_ok h
  obtain ⟨e₅, -⟩ := sampleVec_emitted (by simp [Zk.triple]) h₅ r₄
  obtain ⟨_, t₆, h₆, h⟩ := bind_ok h
  have e₆ := absorbBytes_emitted h₆
  have r₆ := e₆.retained (e₅.retained r₄)
  obtain ⟨beta, t₇, h₇, h⟩ := bind_ok h
  obtain ⟨e₇, -⟩ := sample_emitted h₇ r₆
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨_, t₈, h₈, h⟩ := bind_ok h
  have e₈ := absorbSlice_emitted h₈
  have r₈ := e₈.retained (e₇.retained r₆)
  obtain ⟨ws, t₉, h₉, h⟩ := bind_ok h
  obtain ⟨e₉, -⟩ := sampleVec_emitted (by decide) h₉ r₈
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨-, h⟩ := check_bind h
  rw [forIn_range_eq] at h
  obtain ⟨_, tL, hloop, hE⟩ := bind_ok h
  have eL := forIn_emitted _ (fun _ => []) (by
    intro i s tt b tt' _ hb
    obtain ⟨-, hb⟩ := check_bind hb
    obtain ⟨-, hb⟩ := check_bind hb
    obtain ⟨-, hb⟩ := check_bind hb
    obtain ⟨rfl, rfl⟩ := pure_ok hb
    exact ⟨_, rfl, Emitted.refl _⟩) Zk.Q 0 _ _ _ _ (e₉.retained r₈) hloop
  obtain ⟨-, rfl⟩ := pure_ok hE
  have e := e₁.trans (e₂.trans (e₃.trans (e₄.trans (e₅.trans (e₆.trans (e₇.trans (e₈.trans (e₉.trans eL))))))))
  rw [flatMap_nil_const] at e
  simpa [innerFr, Zk.triple] using e

theorem ligeritoConstraints_size {lay : Zk.PadLayout} {rec : Zk.L0Rec} {lg : LigeritoProof} {msgs : Zk.Level0Msgs}
    {v : Array (String × Zk.Lin)} (h : Zk.ligeritoConstraints lay rec lg msgs = .ok v) : v.size = 2 := by
  unfold Zk.ligeritoConstraints at h
  dsimp only at h
  split at h
  · cases h
  · obtain ⟨_, -, h⟩ := ebind_ok h
    obtain ⟨_, -, h⟩ := ebind_ok h
    cases h
    rfl

end FlockSoundness.Discharge.ZkExec
