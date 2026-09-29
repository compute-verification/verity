import FlockSoundness.Refine.Frames

/-!
# The PIOP's rounds, as frames (R11b)

Which frames each round of the PIOP's phases retained, on the unsalted path: `zerocheck_frames` for the statement
binding and the zerocheck (PROTOCOL.md §9.1–§9.2), `lincheck_frames` for the lincheck (§10), and `ringSwitch_frames` for
the opening's ring switch (§12.2–§12.3).
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
  obtain ⟨-, h⟩ := check_bind h
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

/-- **The lincheck's rounds' bytes** (PROTOCOL.md §10). When the lincheck succeeds from `t` to `t'` on retained rounds,
the rounds it consumes retained, in order:
* the pending frames (the zerocheck's final values), the lincheck's label, and `α`'s squeeze;
* `β`'s squeeze;
* each sumcheck round's two values and its squeeze;
* `z_partial` and `r'`'s squeeze.
Nothing is left pending. -/
theorem lincheck_frames (c : CircuitFold) (zc : Flock.ZcOut) (p : LincheckProof) (t t' : Transcript) (lc : Flock.LcOut)
    (hret : AllRetained t.rounds) (h : lincheck c zc p t = .ok (lc, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + 3 + (c.kLog - 6) ∧
    Retains (t.rounds[t.k]!) (t.pending.data.toList ++ encs [.label "flock-lincheck-v0".toUTF8, .sq 1 1]) ∧
    Retains (t.rounds[t.k + 1]!) (encs [.sq 1 1]) ∧
    (∀ j < c.kLog - 6, Retains (t.rounds[t.k + 2 + j]!) (encs [.one (p.rounds[j]!).1, .one (p.rounds[j]!).2, .sq 1 1])) ∧
    Retains (t.rounds[t.k + 2 + (c.kLog - 6)]!) (encs [.vec p.zPartial.toList, .sq 1 1]) ∧
    t'.pending = ByteArray.empty := by
  unfold lincheck at h
  have h := modify_bind h
  obtain ⟨-, -, R₀, h⟩ := sample_bindA h hret
  obtain ⟨-, -, R₁, h⟩ := sample_bindA h hret
  split at h
  · rename_i v hv
    obtain ⟨_, _, h0, h⟩ := bind_ok h
    obtain ⟨rfl, rfl⟩ := pure_ok h0
    obtain ⟨-, h⟩ := check_bind h
    obtain ⟨s, t₁₀, hloop, h⟩ := bind_ok h
    rw [forIn_range_eq] at hloop
    obtain ⟨hr₁₀, hk₁₀, hp₁₀, hR⟩ := forIn_answers _
      (fun j => encs [.one (p.rounds[j]!).1, .one (p.rounds[j]!).2, .sq 1 1]) (by
        intro j st tt b tt' hrt hp hb
        rcases hg : p.rounds[j]! with ⟨g1, ginf⟩
        simp only [hg] at hb
        have hb := modify_bind hb
        have hb := modify_bind hb
        obtain ⟨-, -, hRr, hb⟩ := sample_bindA hb hrt
        obtain ⟨rfl, rfl⟩ := pure_ok hb
        refine ⟨_, rfl, rfl, rfl, rfl, ?_⟩
        dsimp only at hRr
        simp only [absorbF_enc, hp] at hRr
        simpa [encs] using hRr) (c.kLog - 6) 0 _ _ t₁₀ s (by exact hret) rfl hloop
    have h := modify_bind h
    obtain ⟨-, h⟩ := check_bind h
    obtain ⟨-, -, R₃, h⟩ := sample_bindA h (by show AllRetained t₁₀.rounds; rw [hr₁₀]; exact hret)
    obtain ⟨rfl, rfl⟩ := pure_ok h
    refine ⟨hr₁₀, by simp only [hk₁₀]; omega, ?_, ?_, fun j hj => ?_, ?_, rfl⟩
    · dsimp only at R₀
      simp only [label_enc] at R₀
      simpa [encs] using R₀
    · simpa [encs] using R₁
    · have := hR j hj
      rw [show t.k + 1 + 1 + j = t.k + 2 + j by omega, Nat.zero_add] at this
      exact this
    · dsimp only at R₃
      simp only [absorbSlice_enc, hp₁₀, hr₁₀, hk₁₀] at R₃
      rw [show t.k + 1 + 1 + (c.kLog - 6) = t.k + 2 + (c.kLog - 6) by omega] at R₃
      simpa [encs] using R₃
  · obtain ⟨_, _, h0, _⟩ := bind_ok h
    cases h0

/-- **The ring switch's rounds' bytes** (PROTOCOL.md §12.2–§12.3). When the ring switch of `n ≠ 0` claims succeeds from `t`
to `t'` on retained rounds, the rounds it consumes retained, in order:
* the pending frames, the batch label, and claim 0's label, switch vector and squeeze;
* each later claim's label, switch vector and squeeze;
* `γ`'s squeeze, one coin per claim.
Nothing is left pending. -/
theorem ringSwitch_frames (m : ℕ) (claims : Array Flock.Claim) (rs : Array (Array F128)) (hn : claims.size ≠ 0)
    (t t' : Transcript) (out : F128 × Array Flock.Switched) (hret : AllRetained t.rounds)
    (h : ringSwitch m claims rs t = .ok (out, t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + claims.size + 1 ∧
    Retains (t.rounds[t.k]!) (t.pending.data.toList ++ encs [.label "flock-pcs-open-batch-v0".toUTF8,
      .label "flock-ring-switch-v0".toUTF8, .vec (rs[0]!).toList, .sq 2 7]) ∧
    (∀ k < claims.size - 1, Retains (t.rounds[t.k + 1 + k]!)
      (encs [.label "flock-ring-switch-v0".toUTF8, .vec (rs[k + 1]!).toList, .sq 2 7])) ∧
    Retains (t.rounds[t.k + claims.size]!) (encs [.sq 2 claims.size]) ∧
    t'.pending = ByteArray.empty := by
  unfold ringSwitch at h
  have h := modify_bind h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨s, t₁, hloop, h⟩ := bind_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨n', hn'⟩ : ∃ n', claims.size = n' + 1 := ⟨claims.size - 1, by omega⟩
  rw [hn', forIn_range'_cons] at hloop
  obtain ⟨r₀, t₀, h₀, hloop⟩ := bind_ok hloop
  obtain ⟨-, h₀⟩ := check_bind h₀
  have h₀ := modify_bind h₀
  have h₀ := modify_bind h₀
  obtain ⟨-, h₀⟩ := check_bind h₀
  obtain ⟨-, -, R₀, h₀⟩ := sampleVec_bindA (n := 7) (by decide) h₀ hret
  obtain ⟨rfl, rfl⟩ := pure_ok h₀
  simp only [Nat.zero_add] at hloop
  obtain ⟨hr₁, hk₁, hp₁, hR⟩ := forIn_answers _
    (fun k => encs [.label "flock-ring-switch-v0".toUTF8, .vec (rs[k]!).toList, .sq 2 7]) (by
      intro k st tt b tt' hrt hp hb
      obtain ⟨-, hb⟩ := check_bind hb
      have hb := modify_bind hb
      have hb := modify_bind hb
      obtain ⟨-, hb⟩ := check_bind hb
      obtain ⟨-, -, hRr, hb⟩ := sampleVec_bindA (n := 7) (by decide) hb hrt
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      refine ⟨_, rfl, rfl, rfl, rfl, ?_⟩
      dsimp only at hRr
      simp only [label_enc, absorbSlice_enc, hp] at hRr
      simpa [encs] using hRr) n' 1 _ _ t₁ s (by exact hret) rfl hloop
  obtain ⟨-, -, R₂, h⟩ := sampleVec_bindA (n := claims.size) hn h
    (by show AllRetained t₁.rounds; rw [hr₁]; exact hret)
  obtain ⟨rfl, rfl⟩ := pure_ok h
  refine ⟨hr₁, by simp only [hk₁, hn']; omega, ?_, fun k hk => ?_, ?_, rfl⟩
  · dsimp only at R₀
    simp only [label_enc, absorbSlice_enc] at R₀
    simpa [encs] using R₀
  · have := hR k (by omega)
    rw [show 1 + k = k + 1 by omega] at this
    exact this
  · simp only [hp₁, hr₁, hk₁] at R₂
    rw [show t.k + 1 + n' = t.k + claims.size by omega] at R₂
    simpa [encs] using R₂

end FlockSoundness.Refine
