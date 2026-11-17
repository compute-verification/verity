module

public import Proofs.Flock.ZkAny.RelayFrames

/-! `EnvPre` from any invariant of the relay's requests that each slot's step keeps (`envPre_inv`). -/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.ZkServer.Session
open _root_.Flock.ZkServer hiding State
open _root_.Flock (SessionSpec)

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
