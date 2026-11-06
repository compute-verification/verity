module

public import Aesop.BuiltinRules
public import Batteries.Data.ByteArray
public import Flock.Firewall.Frame
public import Mathlib.Tactic.Attr.Core
public import Mathlib.Tactic.Basic
public import Mathlib.Tactic.ToAdditive
public import Mathlib.Tactic.ToDual
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import VCVio.Prelude.Core

/-!
# The frame's fold over its requests

The session loop drives each rep's frame on one schedule of requests (`Flock.Firewall.Drive.reqs`), each a pure step of
the rep's state (`Frame.step`), folded by `Frame.run` up to the first answer that ends the rep. These are the fold's and
the step's equations that the loop's release reads:

* `run_append`: the fold over `qs ++ rs` is the fold over `qs`, then, where that did not end the rep, the fold over `rs`
  from its state.
* `run_snoc`: so one request past a fold that did not end is one `step` past it.
* `step_sched`, `step_coins`: a round with its scheduled round in its place is the frame's `check` (`check_l0`'s), and
  coins are `Masker.answer`; `check_of_step`, `answer_of_step`: conversely, a round that left was the check and coins
  taken were the answer.
-/

@[expose] public section

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

/-- **A round that left with a scheduled round in its place** was the frame's pass decision (`check`), at the state the
round left. -/
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

/-- **Coins the frame took** left `Masker.answer`'s state. -/
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

end Flock.Firewall.Frame
