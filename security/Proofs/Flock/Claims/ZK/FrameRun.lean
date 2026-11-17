module

public import Proofs.Flock.Claims.ZK.FrameState

/-! The frame's fold over its requests. -/

@[expose] public section

namespace Flock.ZkServer.Frame

theorem run_append (c : Cfg) : ∀ (st : State) (qs rs : List Req),
    run c st (qs ++ rs) = if (run c st qs).2 then run c st qs else run c (run c st qs).1 rs
  | st, [], rs => by simp [run]
  | st, q :: qs, rs => by
    rcases e : step c st q with ⟨a, st'⟩
    simp only [List.cons_append, run, e]
    cases a.ends
    · simp only [Bool.false_eq_true, ↓reduceIte]
      exact run_append c st' qs rs
    · simp

theorem run_snoc (c : Cfg) (st : State) (qs : List Req) (q : Req) (h : (run c st qs).2 = false) :
    run c st (qs ++ [q]) = ((step c (run c st qs).1 q).2, (step c (run c st qs).1 q).1.ends) := by
  rw [run_append, h]
  simp only [Bool.false_eq_true, ↓reduceIte]
  rcases e : step c (run c st qs).1 q with ⟨a, st'⟩
  cases ha : a.ends <;> simp [run, e, ha]

theorem step_sched {c : Cfg} {st : State} {n : Nat} {v r out : ByteArray} {s' : Masker} {why : Option String}
    (ho : st.m.owed = none) (h : check c st.m v r n = .ok (out, s', why)) :
    step c st (.round n v (.sched r)) = (.round out (verdict (.sched r) why), { m := s', begun := true }) := by
  simp only [step, ho, h]

theorem step_coins {c : Cfg} {st : State} {cs : Array F128} {s' : Masker} (h : st.m.answer c cs = .ok s') :
    step c st (.coins (.ok cs)) = (.ok, { st with m := s' }) := by
  simp only [step, h]

theorem check_of_step {c : Cfg} {st st' : State} {n : Nat} {v r out : ByteArray} {vd : String}
    (h : step c st (.round n v (.sched r)) = (.round out vd, st')) :
    ∃ why, check c st.m v r n = .ok (out, st'.m, why) := by
  simp only [step] at h
  split at h
  · cases h
  · revert h
    cases hc : check c st.m v r n with
    | error e =>
      intro h
      cases hs : e.startsWith STOP <;> simp [hs] at h
    | ok q =>
      obtain ⟨o, s', w⟩ := q
      intro h
      cases h
      exact ⟨w, rfl⟩

theorem answer_of_step {c : Cfg} {st st' : State} {cs : Array F128} (h : step c st (.coins (.ok cs)) = (.ok, st')) :
    st.m.answer c cs = .ok st'.m := by
  simp only [step] at h
  revert h
  cases ha : st.m.answer c cs with
  | error e =>
    intro h
    cases hs : e.startsWith STOP <;> simp [hs] at h
  | ok s' =>
    intro h
    cases h
    rfl

end Flock.ZkServer.Frame
