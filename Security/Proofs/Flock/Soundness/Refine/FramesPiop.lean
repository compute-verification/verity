import Proofs.Flock.Soundness.Refine.Frames

/-!
# The PIOP's rounds, as frames (R11b)

Which frames each round of the PIOP's phases retained, on the unsalted path: `zerocheck_frames` for the statement
binding and the zerocheck (PROTOCOL.md §9.1–§9.2).
-/

namespace FlockSoundness.Refine

open Flock Transcript

/-- **The zerocheck's rounds' bytes** (PROTOCOL.md §9.1–§9.2). When the statement binding and the zerocheck succeed from
`t` to `t'` on retained rounds, the rounds they consume retained, in order:
* the pending frames, the binding's labels, digest and cap, and the first squeeze;
* the second squeeze's header;
* the round-1 message and `z`'s squeeze;
* each residual round's two values and its squeeze.
The zerocheck's final values are left pending. -/
theorem zerocheck_frames (m : ℕ) (hm : 14 ≤ m) (digest cap : ByteArray) (p : ZerocheckProof) (t t' : Transcript)
    (zc : Flock.ZcOut) (hret : AllRetained t.rounds) (h : bindAndZerocheck m digest cap p t = .ok (zc, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + 3 + (m - 6) ∧
    Retains (t.rounds[t.k]!) (t.pending.data.toList ++ encs [.label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap,
      .label "flock-zerocheck-v0".toUTF8, .sq 2 6]) ∧
    Retains (t.rounds[t.k + 1]!) (encs [.sq 2 (m - 13)]) ∧
    Retains (t.rounds[t.k + 2]!) (encs [.vec p.round1ab.toList, .vec p.round1c.toList, .sq 1 1]) ∧
    (∀ i < m - 6, Retains (t.rounds[t.k + 3 + i]!) (encs [.one (p.rounds[i]!).1, .one (p.rounds[i]!).2, .sq 1 1])) ∧
    t'.pending.data.toList = encs [.one p.finalA, .one p.finalB] := by
  unfold bindAndZerocheck at h
  have h := modify_bind h
  have h := modify_bind h
  have h := modify_bind h
  have h := modify_bind h
  obtain ⟨-, -, R₀, h⟩ := sampleVec_bindA (n := 6) (by decide) h hret
  obtain ⟨-, -, R₁, h⟩ := sampleVec_bindA (n := m - 13) (by omega) h hret
  have h := modify_bind h
  have h := modify_bind h
  obtain ⟨-, -, R₂, h⟩ := sample_bindA h hret
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨s, t₁₀, hloop, h⟩ := bind_ok h
  obtain ⟨-, h⟩ := check_bind h
  have h := modify_bind h
  have h := modify_bind h
  obtain ⟨rfl, rfl⟩ := pure_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₁₀, hk₁₀, hp₁₀, hR⟩ := forIn_answers _
    (fun i => encs [.one (p.rounds[i]!).1, .one (p.rounds[i]!).2, .sq 1 1]) (by
      intro i st tt b tt' hrt hp hb
      rcases hg : p.rounds[i]! with ⟨g1, ginf⟩
      simp only [hg] at hb
      have hb := modify_bind hb
      have hb := modify_bind hb
      obtain ⟨-, -, hRr, hb⟩ := sample_bindA hb hrt
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      refine ⟨_, rfl, rfl, rfl, rfl, ?_⟩
      dsimp only at hRr
      simp only [absorbF_enc, hp] at hRr
      simpa [encs] using hRr) (m - 6) 0 _ _ t₁₀ s (by exact hret) rfl hloop
  refine ⟨hr₁₀, by rw [hk₁₀], ?_, ?_, ?_, fun i hi => ?_, ?_⟩
  · dsimp only at R₀
    simp only [label_enc, absorbBytes_enc] at R₀
    simpa [encs] using R₀
  · simpa [encs] using R₁
  · dsimp only at R₂
    simp only [absorbSlice_enc] at R₂
    simpa [encs] using R₂
  · have := hR i hi
    rw [show t.k + 1 + 1 + 1 + i = t.k + 3 + i by omega, Nat.zero_add] at this
    exact this
  · simp only [absorbF_enc, hp₁₀]
    simp [encs]

end FlockSoundness.Refine
