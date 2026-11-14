module

public import Proofs.Flock.SpotFlock.RunsAs.Steps
public import Spot.Run
import Proofs.Flock.Claims.Sampling.Select.Code

/-!
# An accepted run passed its population check

`Run.run`'s register step refuses a boundary registration whose program and query do not lay out the registration's
population as its units in order (`Slots.popOk`, code `population-order`), at the population `n` the boundary record
states (`population`). So a run that does not refuse passed that check at that `n` (`run_popOk`), and so did every
accepted two-stage run (`popOk_of_accepted`, `population_of_accepted`).
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

-- `run` unfolds to one `match` or `if` per step; the walk stops at the population check, the register step's last.
set_option maxHeartbeats 4000000 in
/-- **A run that does not refuse passed its population check**: `Run.run` either refuses at a step (a `{"step",
"codes"}` object) or passed `Slots.popOk` at the params' program and query and at the population its boundary record
states. -/
theorem run_popOk {i : Spot.Run.Input} {form : Bool} {j : Json} (h : Spot.Run.run i form = .ok j) :
    (∃ (step : String) (codes : Array String),
      j = Json.mkObj [("step", Json.str step), ("codes", Json.arr (codes.map Json.str))]) ∨
    ∃ n, Spot.nat? (Spot.getD i.record "population") = some n ∧
      Spot.Slots.popOk i.params.verifier.program i.params.verifier.query n = true := by
  unfold Spot.Run.run at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat'
    first
    | exact Or.inr ⟨_, by assumption, by simpa using ‹¬(!Spot.Slots.popOk _ _ _) = true›⟩
    | split at h
    | focus (cases h; done)
    | (obtain ⟨_, h⟩ | ⟨_, h⟩ := Spot.Run.ite_eq_ok h; exact Or.inl ⟨_, _, h.symm⟩)
    | exact Or.inl ⟨_, _, (Except.ok.inj h).symm⟩

/-- **An accepted two-stage run passed its population check** at the population its boundary record states. -/
theorem population_of_accepted {i : Spot.Run.Input} {it : Spot.Run.Interior} {j : Json} (ha : AcceptedTwo i it j) :
    ∃ n, Spot.nat? (Spot.getD i.record "population") = some n ∧
      Spot.Slots.popOk i.params.verifier.program i.params.verifier.query n = true := by
  rcases run_popOk ha.run with ⟨step, codes, hj⟩ | hp
  · have hs := ha.accept
    rw [hj, symbol_refused] at hs
    cases hs
  · exact hp

/-- **An accepted two-stage run passed its population check at `n`**, wherever `n` is the population its boundary
record states. -/
theorem popOk_of_accepted {i : Spot.Run.Input} {it : Spot.Run.Interior} {j : Json} {n : ℕ} (ha : AcceptedTwo i it j)
    (hn : Spot.nat? (Spot.getD i.record "population") = some n) :
    Spot.Slots.popOk i.params.verifier.program i.params.verifier.query n = true := by
  obtain ⟨m, hm, hp⟩ := population_of_accepted ha
  rw [hn] at hm
  cases hm
  exact hp

end SpotFlock.RunsAs
