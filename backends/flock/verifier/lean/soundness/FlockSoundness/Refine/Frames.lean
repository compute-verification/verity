import FlockSoundness.Refine.Paths
import FlockSoundness.Refine.Exec
import FlockProofs.Transcript
import FlockSoundness.Protocol.Refine.Frames

/-!
# The transcript's frames (R11b, framing)

The replay transcript (PROTOCOL.md §6) appends one frame per domain, label, absorb or zero-coin squeeze to its pending
bytes, and a squeeze answers the pending bytes and its own header; on the unsalted path the server retained each round's
bytes, and the replay compares them exactly. This file models those bytes.

* `Frame`, `Frame.enc`: a frame and its bytes, as the executable writes them (a 16-byte header naming its kind and
  length, then its payload).
* `encs_inj`: a list of frames is recoverable from its bytes, so a round's retained bytes determine what was absorbed.
* `label_enc`, `absorbF_enc`, `absorbSlice_enc`, `absorbBytes_enc`, `framed_enc`, `init_enc`: each step of the
  transcript appends its frame's bytes.
* `squeeze_bindA`, `sample_bindA`, `sampleVec_bindA`, `sampleK_bindA`, `forIn_answers`: a squeeze, or a loop of them,
  consumes rounds that retained the pending bytes and the squeeze's header.
-/

namespace FlockSoundness.Refine

open Flock

namespace Frame

end Frame

/-! ## What the transcript's steps append -/

open Transcript

theorem data_append (a b : ByteArray) : (a ++ b).data.toList = a.data.toList ++ b.data.toList := by simp

variable {β : Type}

theorem sampleVec_bindA {n : ℕ} (hn : n ≠ 0) {f : Array F128 → TM β} {t t' : Transcript} {b : β}
    (h : (sampleVec n >>= f) t = .ok (b, t')) (hret : AllRetained t.rounds) :
    t.k < t.rounds.size ∧ (t.rounds[t.k]!).n = n ∧ Retains (t.rounds[t.k]!) (t.pending.data.toList ++ (Frame.sq 2 n).enc) ∧
      f (t.rounds[t.k]!).coins { t with pending := ByteArray.empty, k := t.k + 1 } = .ok (b, t') :=
  squeeze_bindA hn h hret

/-- **A loop answering one round per step.** If every step of the body, with nothing pending, consumes the next round,
which retained `F i`, and leaves nothing pending, then the loop over `a, …, a + n − 1` consumes `n` rounds, which
retained `F a, …, F (a + n − 1)`. -/
theorem forIn_answers {σ : Type} (body : ℕ → σ → TM (ForInStep σ)) (F : ℕ → List UInt8)
    (hbody : ∀ i s t b t', AllRetained t.rounds → t.pending = ByteArray.empty → body i s t = .ok (b, t') →
      ∃ s', b = ForInStep.yield s' ∧ t'.rounds = t.rounds ∧ t'.k = t.k + 1 ∧ t'.pending = ByteArray.empty ∧
        Retains (t.rounds[t.k]!) (F i)) :
    ∀ (n a : ℕ) (s : σ) (t t' : Transcript) (s' : σ), AllRetained t.rounds → t.pending = ByteArray.empty →
      (forIn (List.range' a n) s body : TM σ) t = .ok (s', t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + n ∧ t'.pending = ByteArray.empty ∧
        ∀ i < n, Retains (t.rounds[t.k + i]!) (F (a + i))
  | 0, a, s, t, t', s', _, hp, h => by
      simp only [List.range'_zero, List.forIn_nil] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h
      exact ⟨rfl, rfl, hp, fun i hi => absurd hi (by omega)⟩
  | n + 1, a, s, t, t', s', hret, hp, h => by
      rw [forIn_range'_cons] at h
      obtain ⟨r, t₁, h₁, h₂⟩ := bind_ok h
      obtain ⟨s₁, rfl, hr₁, hk₁, hp₁, hR₁⟩ := hbody a s t r t₁ hret hp h₁
      obtain ⟨hr, hk, hpe, hR⟩ := forIn_answers body F hbody n (a + 1) s₁ t₁ t' s' (hr₁ ▸ hret) hp₁ h₂
      refine ⟨hr.trans hr₁, by rw [hk, hk₁]; omega, hpe, fun i hi => ?_⟩
      rcases i with _ | i
      · simpa using hR₁
      · have := hR i (by omega)
        rw [hr₁, hk₁, show t.k + 1 + i = t.k + (i + 1) by omega, show a + 1 + i = a + (i + 1) by omega] at this
        exact this

end FlockSoundness.Refine
