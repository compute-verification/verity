import Proofs.Flock.Soundness.Discharge.Exec.FramesLigerito
import Proofs.Flock.Soundness.Refine.Table

/-!
# The PIOP's rounds, as the frames they emit (R11b)

The phases before Ligerito, in the form `Emitted` composes (`Emit.lean`): each succeeds on retained rounds only
by emitting its frames, in order.
* `zerocheck_emitted`: the statement binding and the zerocheck (PROTOCOL.md §9.1–§9.2), `zcFrames`; its final values
  are left pending for the lincheck's first round.
* `lincheck_emitted`: the lincheck (§10), `lcFrames`.
* `ringSwitch_emitted`: the opening's ring switch (§12.2–§12.3), `rsFrames`.

`rep_rounds` composes them with `ligerito_frames`: when `verifyRep` accepts, every round of the rep's stream retained
the bytes of its chunk of `repFrames`, one chunk per round.
-/

namespace FlockSoundness.Discharge.Exec

open Flock Refine Transcript

/-- The statement binding's and the zerocheck's frames (PROTOCOL.md §9.1–§9.2). -/
def zcFrames (m : ℕ) (digest cap : ByteArray) (p : ZerocheckProof) : List Frame :=
  [.label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap, .label "flock-zerocheck-v0".toUTF8, .sq 2 6, .sq 2 (m - 13),
    .vec p.round1ab.toList, .vec p.round1c.toList, .sq 1 1] ++
  (List.range' 0 (m - 6)).flatMap (fun i => [.one (p.rounds[i]!).1, .one (p.rounds[i]!).2, .sq 1 1]) ++
  [.one p.finalA, .one p.finalB]

theorem zerocheck_emitted (m : ℕ) (hm : 14 ≤ m) (digest cap : ByteArray) (p : ZerocheckProof) (t t' : Transcript)
    (zc : Flock.ZcOut) (hret : AllRetained t.rounds) (h : bindAndZerocheck m digest cap p t = .ok (zc, t')) :
    Emitted t t' (zcFrames m digest cap p) := by
  unfold bindAndZerocheck at h
  obtain ⟨_, t₁, h₁, h⟩ := bind_ok h
  have e₁ := label_emitted h₁
  obtain ⟨_, t₂, h₂, h⟩ := bind_ok h
  have e₂ := absorbBytes_emitted h₂
  obtain ⟨_, t₃, h₃, h⟩ := bind_ok h
  have e₃ := absorbBytes_emitted h₃
  obtain ⟨_, t₄, h₄, h⟩ := bind_ok h
  have e₄ := label_emitted h₄
  have r₄ := e₄.retained (e₃.retained (e₂.retained (e₁.retained hret)))
  obtain ⟨_, t₅, h₅, h⟩ := bind_ok h
  obtain ⟨e₅, -⟩ := sampleVec_emitted (by decide) h₅ r₄
  obtain ⟨_, t₆, h₆, h⟩ := bind_ok h
  obtain ⟨e₆, -⟩ := sampleVec_emitted (by omega) h₆ (e₅.retained r₄)
  obtain ⟨_, t₇, h₇, h⟩ := bind_ok h
  have e₇ := absorbSlice_emitted h₇
  obtain ⟨_, t₈, h₈, h⟩ := bind_ok h
  have e₈ := absorbSlice_emitted h₈
  obtain ⟨_, t₉, h₉, h⟩ := bind_ok h
  have r₈ := e₈.retained (e₇.retained (e₆.retained (e₅.retained r₄)))
  obtain ⟨e₉, -⟩ := sample_emitted h₉ r₈
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨s, t₁₀, hloop, h⟩ := bind_ok h
  rw [forIn_range_eq] at hloop
  have eL := forIn_emitted _ (fun i => [.one (p.rounds[i]!).1, .one (p.rounds[i]!).2, .sq 1 1]) (by
    intro i st tt b tt' hrt hb
    rcases hg : p.rounds[i]! with ⟨g1, ginf⟩
    simp only [hg] at hb
    obtain ⟨_, u₁, k₁, hb⟩ := bind_ok hb
    have f₁ := absorbF_emitted k₁
    obtain ⟨_, u₂, k₂, hb⟩ := bind_ok hb
    have f₂ := absorbF_emitted k₂
    obtain ⟨_, u₃, k₃, hb⟩ := bind_ok hb
    obtain ⟨f₃, -⟩ := sample_emitted k₃ (f₂.retained (f₁.retained hrt))
    obtain ⟨rfl, rfl⟩ := pure_ok hb
    exact ⟨_, rfl, by simpa using f₁.trans (f₂.trans f₃)⟩) (m - 6) 0 _ _ t₁₀ s (e₉.retained r₈) hloop
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨_, t₁₁, h₁₁, h⟩ := bind_ok h
  have e₁₁ := absorbF_emitted h₁₁
  obtain ⟨_, t₁₂, h₁₂, h⟩ := bind_ok h
  have e₁₂ := absorbF_emitted h₁₂
  obtain ⟨rfl, rfl⟩ := pure_ok h
  have := e₁.trans (e₂.trans (e₃.trans (e₄.trans (e₅.trans (e₆.trans (e₇.trans (e₈.trans (e₉.trans (eL.trans
    (e₁₁.trans e₁₂))))))))))
  simpa [zcFrames] using this

/-- The lincheck's frames (PROTOCOL.md §10). -/
def lcFrames (kLog : ℕ) (p : LincheckProof) : List Frame :=
  [.label "flock-lincheck-v0".toUTF8, .sq 1 1, .sq 1 1] ++
  (List.range' 0 (kLog - 6)).flatMap (fun j => [.one (p.rounds[j]!).1, .one (p.rounds[j]!).2, .sq 1 1]) ++
  [.vec p.zPartial.toList, .sq 1 1]

theorem lincheck_emitted (c : CircuitFold) (zc : Flock.ZcOut) (p : LincheckProof) (t t' : Transcript)
    (lc : Flock.LcOut) (hret : AllRetained t.rounds) (h : lincheck c zc p t = .ok (lc, t')) :
    Emitted t t' (lcFrames c.kLog p) := by
  unfold lincheck at h
  obtain ⟨_, t₁, h₁, h⟩ := bind_ok h
  have e₁ := label_emitted h₁
  obtain ⟨α₀, t₂, h₂, h⟩ := bind_ok h
  obtain ⟨e₂, -⟩ := sample_emitted h₂ (e₁.retained hret)
  obtain ⟨β₀, t₃, h₃, h⟩ := bind_ok h
  have r₂ := e₂.retained (e₁.retained hret)
  obtain ⟨e₃, -⟩ := sample_emitted h₃ r₂
  obtain ⟨s, t₁₀, hloop, h⟩ := bind_ok h
  rw [forIn_range_eq] at hloop
  have eL := forIn_emitted _ (fun j => [.one (p.rounds[j]!).1, .one (p.rounds[j]!).2, .sq 1 1]) (by
    intro j st tt b tt' hrt hb
    rcases hg : p.rounds[j]! with ⟨g1, ginf⟩
    simp only [hg] at hb
    obtain ⟨_, u₁, k₁, hb⟩ := bind_ok hb
    have f₁ := absorbF_emitted k₁
    obtain ⟨_, u₂, k₂, hb⟩ := bind_ok hb
    have f₂ := absorbF_emitted k₂
    obtain ⟨_, u₃, k₃, hb⟩ := bind_ok hb
    obtain ⟨f₃, -⟩ := sample_emitted k₃ (f₂.retained (f₁.retained hrt))
    obtain ⟨rfl, rfl⟩ := pure_ok hb
    exact ⟨_, rfl, by simpa using f₁.trans (f₂.trans f₃)⟩) (c.kLog - 6) 0 _ _ t₁₀ s (e₃.retained r₂) hloop
  dsimp only at h
  rcases hf : c.folded α₀ β₀ (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) s.2 with err | v <;>
    rw [hf] at h <;> dsimp only at h
  · obtain ⟨_, _, h0, _⟩ := bind_ok h
    cases h0
  · obtain ⟨_, _, h0, hc⟩ := bind_ok h
    obtain ⟨rfl, rfl⟩ := pure_ok h0
    obtain ⟨-, hc⟩ := check_bind hc
    obtain ⟨_, t₁₁, h₁₁, h⟩ := bind_ok hc
    have e₁₁ := absorbSlice_emitted h₁₁
    obtain ⟨-, h⟩ := check_bind h
    obtain ⟨_, t₁₂, h₁₂, h⟩ := bind_ok h
    obtain ⟨e₁₂, -⟩ := sample_emitted h₁₂ (e₁₁.retained (eL.retained (e₃.retained r₂)))
    obtain ⟨rfl, rfl⟩ := pure_ok h
    have := e₁.trans (e₂.trans (e₃.trans (eL.trans (e₁₁.trans e₁₂))))
    simpa [lcFrames] using this

/-- The ring switch's frames (PROTOCOL.md §12.2–§12.3), for `n` claims: the batch label; per claim its label, switch
vector and squeeze; then `γ`'s squeeze. -/
def rsFrames (n : ℕ) (rs : Array (Array F128)) : List Frame :=
  .label "flock-pcs-open-batch-v0".toUTF8 ::
    ((List.range' 0 n).flatMap (fun k => [.label "flock-ring-switch-v0".toUTF8, .vec (rs[k]!).toList, .sq 2 7]) ++
      [.sq 2 n])

theorem ringSwitch_emitted (m : ℕ) (claims : Array Flock.Claim) (rs : Array (Array F128)) (hn : claims.size ≠ 0)
    (t t' : Transcript) (out : F128 × Array Flock.Switched) (hret : AllRetained t.rounds)
    (h : ringSwitch m claims rs t = .ok (out, t')) : Emitted t t' (rsFrames claims.size rs) := by
  unfold ringSwitch at h
  obtain ⟨_, t₁, h₁, h⟩ := bind_ok h
  have e₁ := label_emitted h₁
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨s, t₂, hloop, h⟩ := bind_ok h
  rw [forIn_range_eq] at hloop
  have eL := forIn_emitted _ (fun k => [.label "flock-ring-switch-v0".toUTF8, .vec (rs[k]!).toList, .sq 2 7]) (by
    intro k st tt b tt' hrt hb
    obtain ⟨-, hb⟩ := check_bind hb
    obtain ⟨_, u₁, k₁, hb⟩ := bind_ok hb
    have f₁ := label_emitted k₁
    obtain ⟨_, u₂, k₂, hb⟩ := bind_ok hb
    have f₂ := absorbSlice_emitted k₂
    obtain ⟨-, hb⟩ := check_bind hb
    obtain ⟨_, u₃, k₃, hb⟩ := bind_ok hb
    obtain ⟨f₃, -⟩ := sampleVec_emitted (by decide) k₃ (f₂.retained (f₁.retained hrt))
    obtain ⟨rfl, rfl⟩ := pure_ok hb
    exact ⟨_, rfl, by simpa using f₁.trans (f₂.trans f₃)⟩) claims.size 0 _ _ t₂ s (e₁.retained hret) hloop
  obtain ⟨_, t₃, h₃, h⟩ := bind_ok h
  obtain ⟨e₃, -⟩ := sampleVec_emitted hn h₃ (eL.retained (e₁.retained hret))
  obtain ⟨rfl, rfl⟩ := pure_ok h
  have := e₁.trans (eL.trans e₃)
  simpa [rsFrames] using this

/-- **A rep's frames** (PROTOCOL.md §6–§13): its domain; the statement binding and the zerocheck; the lincheck; the ring
switch of `n` claims; and Ligerito, whose first frames absorb the ring switch's batched value `T`. -/
def repFrames (st : Setup) (sch : Flock.Schedule) (domain : ByteArray) (n : ℕ) (p : Proof) (T : F128) : List Frame :=
  .dom domain :: (zcFrames st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck ++
    lcFrames st.fold.kLog p.lincheck ++ rsFrames n p.ringSwitches ++ ligFrames sch T p.cap p.ligerito)

/-- **Every round of a verified rep's stream retained its frames** (PROTOCOL.md §6, §9–§13). When `verifyRep` accepts
rep `r` of table `i`, its stream and proof decode, and on a stream whose rounds all retained their bytes, round `j`
retained the bytes of the `j`-th chunk of the rep's frames, and there is one chunk per round. -/
theorem rep_rounds (spec : SessionSpec) (st : Setup) (sch : Flock.Schedule) (sess : Session) (i r : ℕ)
    (bytes cap : ByteArray) (hm : 14 ≤ st.m) (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (h : verifyRep spec st sch sess i r bytes = .ok cap) :
    ∃ (stream : StreamRec) (p : Proof) (extras : Array Flock.Claim) (T : F128),
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r) = some stream ∧
      st.extra sess.points = .ok extras ∧ decodeProof (shapeOf st sch extras.size) bytes = .ok p ∧
      (AllRetained stream.rounds →
        (chunks (repFrames st sch (spec.domain spec.tables[i]! r).toUTF8 (2 + extras.size) p T)).1.length =
          stream.rounds.size ∧
        ∀ j < stream.rounds.size, Retains (stream.rounds[j]!)
          (encs ((chunks (repFrames st sch (spec.domain spec.tables[i]! r).toUTF8 (2 + extras.size) p T)).1.getD j []))) := by
  unfold verifyRep at h
  dsimp only at h
  cases hs : Array.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r)
      sess.streams with
  | none => rw [hs] at h; cases h
  | some stream =>
      rw [hs] at h
      obtain ⟨extras, hex, h⟩ := ebind_ok h
      obtain ⟨p, hp, h⟩ := ebind_ok h
      by_cases c1 : (p.params != pcsParams st.merkle sch) = true
      · rw [ite_eq_left c1] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
      rw [ite_eq_right c1] at h
      obtain ⟨⟨u, tr⟩, hrun, h⟩ := ebind_ok h
      change (if (tr.k != stream.rounds.size) = true then _ else _) = _ at h
      by_cases c2 : (tr.k != stream.rounds.size) = true
      · rw [ite_eq_left c2] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
      simp only [bne_iff_ne, ne_eq, not_not] at c2
      refine ⟨stream, p, extras, ?_⟩
      rw [StateT.run] at hrun
      obtain ⟨zc, t₁, h₁, hrun⟩ := bind_ok hrun
      obtain ⟨lc, t₂, h₂, hrun⟩ := bind_ok hrun
      obtain ⟨⟨T, sws⟩, t₃, h₃, h₄⟩ := bind_ok hrun
      refine ⟨T, rfl, hex, hp, fun hret => ?_⟩
      have e₀ : Emitted ⟨stream.rounds, 0, ByteArray.empty⟩
          (Transcript.init stream.rounds (spec.domain spec.tables[i]! r).toUTF8)
          [.dom (spec.domain spec.tables[i]! r).toUTF8] :=
        emitted_absorb rfl rfl rfl (by rw [init_enc]; rfl)
      have r₀ : AllRetained (Transcript.init stream.rounds (spec.domain spec.tables[i]! r).toUTF8).rounds := hret
      have e₁ := zerocheck_emitted st.m hm _ _ _ _ _ zc r₀ h₁
      have e₂ := lincheck_emitted _ _ _ _ _ lc (e₁.retained r₀) h₂
      have hsz : (abcClaims st.m st.kLog zc lc ++ extras).size = 2 + extras.size := by simp [abcClaims]
      have e₃ := ringSwitch_emitted _ _ _ (by rw [hsz]; omega) _ _ _ (e₂.retained (e₁.retained r₀)) h₃
      rw [hsz] at e₃
      have e₄ := ligerito_frames _ sch _ _ _ _ hdim hlv (e₃.retained (e₂.retained (e₁.retained r₀))) h₄
      have e := e₀.trans (e₁.trans (e₂.trans (e₃.trans e₄)))
      have hfr : [Frame.dom (spec.domain spec.tables[i]! r).toUTF8] ++
          (zcFrames st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck ++
            (lcFrames st.fold.kLog p.lincheck ++ (rsFrames (2 + extras.size) p.ringSwitches ++
              ligFrames sch T p.cap p.ligerito))) =
          repFrames st sch (spec.domain spec.tables[i]! r).toUTF8 (2 + extras.size) p T := by
        simp [repFrames, List.append_assoc]
      rw [hfr] at e
      obtain ⟨-, hk, hR, -⟩ := e
      refine ⟨by rw [← c2, hk]; simp, fun j hj => ?_⟩
      have hj' : j < (chunks (repFrames st sch (spec.domain spec.tables[i]! r).toUTF8 (2 + extras.size) p T)).1.length := by
        rw [← c2, hk] at hj; simpa using hj
      have := hR j hj'
      simpa using this

end FlockSoundness.Discharge.Exec
