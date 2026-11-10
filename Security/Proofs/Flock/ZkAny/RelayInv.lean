module

public import Proofs.Flock.ZkAny.RelayFrames

/-!
# `EnvPre` from an invariant of the relay's requests

`EnvPre` asks, at each model slot of the relay's session (`schedSteps`), that what the relay read there be the model's
message (`EnvOk`). Each slot's case reads a different part of the loop and the model: the `Hello`'s nonce, the
`Commit`'s roots, a frame round's release, the `Finish`'s proofs. Here the induction over the session's requests is
stated once, over any invariant, so that each slot's case is a step of its own:

* **`envPre_inv`**: where an invariant `P` of the streams whose rounds came, the session, its placement, the requests to
  come and the model's messages to come gives, at each request at a slot, `EnvOk` at a model slot, and itself at the
  next request where the session goes on, after the slot's stream and a model slot's message, `EnvPre` holds wherever
  `P` does.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session
open _root_.Flock.Firewall hiding State
open _root_.Flock (SessionSpec)

/-- **`EnvPre` by an invariant of the relay's requests**: where, at each request `e` at a slot `k` of the session `st`
and placement `pl` at which `P` holds, what the relay read there is the model's next message at a model slot, and `P`
holds at the next request where the session goes on, with the slot's stream seen and a model slot's message taken,
`EnvPre` holds from wherever `P` holds. -/
theorem envPre_inv {spec : SessionSpec} (P : List String → State → Placed → List Env → List Claims.Msg → Prop)
    (hstep : ∀ seen st pl e rest ms k, slot st pl = some k → P seen st pl (e :: rest) ms →
      (isModel k = true → EnvOk spec seen k e (ms.headD [])) ∧
      (((stepK true st pl (some (tagOf k)) (.ok k) e).go &&
          (stepK true st pl (some (tagOf k)) (.ok k) e).st.stop.isNone) = true →
        P (seenAfter seen k) (stepK true st pl (some (tagOf k)) (.ok k) e).st
          (stepK true st pl (some (tagOf k)) (.ok k) e).pl rest (if isModel k then ms.tail else ms))) :
    ∀ (es : List Env) (seen : List String) (st : State) (pl : Placed) (ms : List Claims.Msg),
      P seen st pl es ms → EnvPre spec seen (schedSteps true st pl es) ms
  | [], _, _, _, _, _ => trivial
  | e :: rest, seen, st, pl, ms, hP => by
    rcases hsl : slot st pl with _ | k
    · simp only [schedSteps, hsl, EnvPre]
    obtain ⟨hok, hnext⟩ := hstep seen st pl e rest ms k hsl hP
    simp only [schedSteps, hsl]
    generalize stepK true st pl (some (tagOf k)) (.ok k) e = r at hnext ⊢
    by_cases hm : isModel k = true
    · simp only [EnvPre, hm, ↓reduceIte]
      refine ⟨hok hm, ?_⟩
      by_cases hc : (r.go && r.st.stop.isNone) = true
      · rw [ite_eq_left hc]
        have hP' := hnext hc
        rw [ite_eq_left hm] at hP'
        exact envPre_inv P hstep rest _ _ _ _ hP'
      · rw [ite_eq_right hc]
        trivial
    · have hm' : isModel k = false := by simpa using hm
      simp only [EnvPre, hm', Bool.false_eq_true, ite_false]
      by_cases hc : (r.go && r.st.stop.isNone) = true
      · rw [ite_eq_left hc]
        have hP' := hnext hc
        rw [ite_eq_right hm] at hP'
        exact envPre_inv P hstep rest _ _ _ _ hP'
      · rw [ite_eq_right hc]
        trivial

end Claims.ZKServer.Relay
