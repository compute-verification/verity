import FlockSoundness.Refine.FramesLigerito

/-!
# The PIOP's rounds, as the frames they emit (R11b)

The phases before Ligerito, in the form `Emitted` composes (`Refine/Emit.lean`): each succeeds on retained rounds only
by emitting its frames, in order.
* `zerocheck_emitted`: the statement binding and the zerocheck (PROTOCOL.md §9.1–§9.2), `zcFrames`; its final values
  are left pending for the lincheck's first round.
* `lincheck_emitted`: the lincheck (§10), `lcFrames`.
* `ringSwitch_emitted`: the opening's ring switch (§12.2–§12.3), `rsFrames`.
-/

namespace FlockSoundness.Refine

open Flock Transcript

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
  obtain ⟨-, h⟩ := check_bind h
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
  obtain ⟨_, t₂, h₂, h⟩ := bind_ok h
  obtain ⟨e₂, -⟩ := sample_emitted h₂ (e₁.retained hret)
  obtain ⟨_, t₃, h₃, h⟩ := bind_ok h
  have r₂ := e₂.retained (e₁.retained hret)
  obtain ⟨e₃, -⟩ := sample_emitted h₃ r₂
  split at h
  · rename_i v hv
    obtain ⟨_, _, h0, hc⟩ := bind_ok h
    obtain ⟨rfl, rfl⟩ := pure_ok h0
    obtain ⟨-, hc⟩ := check_bind hc
    obtain ⟨s, t₁₀, hloop, h⟩ := bind_ok hc
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
    obtain ⟨_, t₁₁, h₁₁, h⟩ := bind_ok h
    have e₁₁ := absorbSlice_emitted h₁₁
    obtain ⟨-, h⟩ := check_bind h
    obtain ⟨_, t₁₂, h₁₂, h⟩ := bind_ok h
    obtain ⟨e₁₂, -⟩ := sample_emitted h₁₂ (e₁₁.retained (eL.retained (e₃.retained r₂)))
    obtain ⟨rfl, rfl⟩ := pure_ok h
    have := e₁.trans (e₂.trans (e₃.trans (eL.trans (e₁₁.trans e₁₂))))
    simpa [lcFrames] using this
  · obtain ⟨_, _, h0, _⟩ := bind_ok h
    cases h0

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

end FlockSoundness.Refine
