module

public import Proofs.Flock.SpotFlock.RunsAs.Steps
public import Proofs.Flock.SpotFlock.HeldOf
public import Spot.Run
import Proofs.Flock.Claims.Sampling.Select.Code

/-!
# An accepted two-stage run's registration checks have no code

`Run.run` refuses at its `register` step where the boundary record's registration check (`Registration.check`) has a
code, and the interior's steps refuse where the interior record's has one. Each check runs at the registration with
the run's own law (and the boundary's, its work table), so at the same ports and registered values (`namesOf`). An
accepted run was refused at neither (`checks_of_accepted`).
-/

@[expose] public section

namespace SpotFlock.RunChecks

open Lean (Json)
open SpotFlock.HeldOf (namesOf)

-- `run` unfolds to about 50 decided leaves under one `match` or `if` per step, hence the heartbeat budget.
set_option maxHeartbeats 8000000 in
/-- **A run's answer**: refused at a step, or past the `register` step, with a registration check of no code at the
boundary record, and past its second stage. -/
theorem run_checks {i : Spot.Run.Input} {form : Bool} {j : Json} (h : Spot.Run.run i form = .ok j) :
    (∃ (step : String) (codes : Array String),
      j = Json.mkObj [("step", Json.str step), ("codes", Json.arr (codes.map Json.str))]) ∨
    ((∃ v, ¬(!(Spot.Registration.check i.record v).isEmpty) = true ∧ namesOf v = namesOf i.params.verifier) ∧
      ∃ (draw : Json) (inner : Option (Spot.Run.Interior × Json × Json × Json × ℕ)),
        Spot.Run.secondStage i ((Spot.Draw.proved draw).map Int.toNat) = .ok (.ok inner)) := by
  unfold Spot.Run.run at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat'
    first
    | split at h
    | focus (cases h; done)
    | (obtain ⟨_, h⟩ | ⟨_, h⟩ := Spot.Run.ite_eq_ok h; exact Or.inl ⟨_, _, h.symm⟩)
    | exact Or.inl ⟨_, _, (Except.ok.inj h).symm⟩
    | (refine Or.inr ⟨⟨_, ?_, ?_⟩, _, _, ?_⟩ <;> first | assumption | rfl)

/-- **The second stage under a two-stage rule with an interior** is the interior's steps. -/
theorem secondStage_two {i : Spot.Run.Input} {sample : Array ℕ} {grain : Json} {replay : String}
    {k : Spot.Select.Counts} {it : Spot.Run.Interior} {inner : Option (Spot.Run.Interior × Json × Json × Json × ℕ)}
    (hr : i.rule = .twoStage grain replay k) (hi : i.interior = some it)
    (h : Spot.Run.secondStage i sample = .ok (.ok inner)) :
    ∃ x, Spot.Run.interiorSteps i it k sample = .ok (.ok x) := by
  unfold Spot.Run.secondStage at h
  rw [hr, hi] at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  cases hs : Spot.Run.interiorSteps i it k sample with
  | error e => simp [hs] at h
  | ok r =>
    cases r with
    | error e => simp [hs, Except.map] at h
    | ok x => exact ⟨x, rfl⟩

set_option maxHeartbeats 4000000 in
/-- **The interior's steps, accepted**, passed the interior record's registration check with no code. -/
theorem interiorSteps_check {i : Spot.Run.Input} {it : Spot.Run.Interior} {k : Spot.Select.Counts}
    {sample : Array ℕ} {x : Spot.Run.Interior × Json × Json × Json × ℕ}
    (h : Spot.Run.interiorSteps i it k sample = .ok (.ok x)) :
    ∃ v, ¬(!(Spot.Registration.check it.record v).isEmpty) = true ∧ namesOf v = namesOf it.params.verifier := by
  unfold Spot.Run.interiorSteps at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat'
    first
    | split at h
    | focus (cases h; done)
    | (refine ⟨_, ?_, ?_⟩ <;> first | assumption | rfl)

/-- **An accepted two-stage run's registration checks have no code**, the boundary record's and the interior
record's, each at a registration with its own ports and registered values. -/
theorem checks_of_accepted {i : Spot.Run.Input} {it : Spot.Run.Interior} {j : Json}
    (ha : SpotFlock.RunsAs.AcceptedTwo i it j) :
    (∃ v, Spot.Registration.check i.record v = #[] ∧ namesOf v = namesOf i.params.verifier) ∧
      ∃ v, Spot.Registration.check it.record v = #[] ∧ namesOf v = namesOf it.params.verifier := by
  obtain ⟨grain, replay, k, hr⟩ := ha.two
  rcases run_checks ha.run with ⟨step, codes, hj⟩ | ⟨⟨v, hv, hvn⟩, draw, inner, hsec⟩
  · have hs := ha.accept
    rw [hj, SpotFlock.RunsAs.symbol_refused] at hs
    cases hs
  · obtain ⟨x, hx⟩ := secondStage_two hr ha.interior hsec
    obtain ⟨w, hw, hwn⟩ := interiorSteps_check hx
    exact ⟨⟨v, Array.isEmpty_iff.1 (by simpa using hv), hvn⟩, w, Array.isEmpty_iff.1 (by simpa using hw), hwn⟩

end SpotFlock.RunChecks
