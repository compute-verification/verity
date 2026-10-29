import Proofs.Flock.Claims.ZK.FrameState

/-!
# The frame's fold over its requests

The session loop drives each rep's frame on one schedule of requests (`Flock.Firewall.Drive.reqs`), each a pure step of
the rep's state (`Frame.step`), folded by `Frame.run` up to the first answer that ends the rep. These are the fold's and
the step's equations that the loop's release reads:

* `run_append`: the fold over `qs ++ rs` is the fold over `qs`, then, where that did not end the rep, the fold over `rs`
  from its state.
* `run_snoc`: so one request past a fold that did not end is one `step` past it.
* `step_sched`, `step_coins`: a round with its scheduled round in its place is the frame's `check` (`check_l0`'s), and
  coins are `Masker.answer`.
-/

namespace Flock.Firewall.Frame

/-- **The fold over two lists of requests**: over `qs`, then, where that did not end the rep, over `rs` from its state. -/
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

/-- **One request past a fold that did not end the rep** is one `step` past it. -/
theorem run_snoc (c : Cfg) (st : State) (qs : List Req) (q : Req) (h : (run c st qs).2 = false) :
    run c st (qs ++ [q]) = ((step c (run c st qs).1 q).2, (step c (run c st qs).1 q).1.ends) := by
  rw [run_append, h]
  simp only [Bool.false_eq_true, ↓reduceIte]
  rcases e : step c (run c st qs).1 q with ⟨a, st'⟩
  cases ha : a.ends <;> simp [run, e, ha]

/-- **A round with the statement's scheduled round in its place** is the frame's pass decision (`check`), once the last
round's coins came. -/
theorem step_sched {c : Cfg} {st : State} {n : Nat} {v r out : ByteArray} {s' : Masker} {why : Option String}
    (ho : st.m.owed = none) (h : check c st.m v r n = .ok (out, s', why)) :
    step c st (.round n v (.sched r)) = (.round out (verdict (.sched r) why), { m := s', begun := true }) := by
  simp only [step, ho, h]

/-- **Coins the frame takes** are `Masker.answer`'s state. -/
theorem step_coins {c : Cfg} {st : State} {cs : Array F128} {s' : Masker} (h : st.m.answer c cs = .ok s') :
    step c st (.coins (.ok cs)) = (.ok, { st with m := s' }) := by
  simp only [step, h]

end Flock.Firewall.Frame
