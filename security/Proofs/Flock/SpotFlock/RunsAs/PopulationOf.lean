module

public import Proofs.Flock.SpotFlock.RunsAs.Steps
public import Spot.Run
import Proofs.Flock.Claims.Sampling.Select.Code

/-!
# An accepted run passed its population check

`Run.run`'s register step refuses a boundary registration whose program and query do not lay out the registration's
population as its units in order (`Slots.popOk`, code `population-order`), at the population `n` the boundary record
states (`population`), or as a disjoint cover of them under a one-stage rule and a subset design (`Slots.popCovers`,
`OneStageZk.subsetDesign`, code `strata-design`). So a run that does not refuse passed that check at that `n`
(`run_popHeld`), and every accepted two-stage run passed it in order (`popOk_of_accepted`, `population_of_accepted`).
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

private theorem true_of_not {b : Bool} (h : ¬(!b) = true) : b = true := by simpa using h

-- `run` unfolds to about 50 decided leaves under one `match` or `if` per step, as `Spot.Run.run_decided`'s walk;
-- `split` takes `matches .twoStage ..` apart, whose `twoStage` branch leaves `¬true = true`.
set_option maxHeartbeats 4000000 in
/-- **A run that does not refuse passed its population check** (the run's bridge): `Run.run` either refuses at a step
(a `{"step", "codes"}` object) or, at the population its boundary record states, the params' program and query lay
it out as its units in order (`Slots.popOk`), or as a disjoint cover of them (`Slots.popCovers`) under a one-stage rule
and a design that is subset at both the registration and the verifier the run derives, at the sampling design object
of the rule over the population and the work table `Rule.tableOver` gives (`OneStageZk.subsetDesign`). -/
theorem run_popHeld {i : Spot.Run.Input} {j : Json} (h : Spot.Run.run i = .ok j) :
    (∃ (step : String) (codes : Array String),
      j = Json.mkObj [("step", Json.str step), ("codes", Json.arr (codes.map Json.str))]) ∨
    ∃ n, Spot.nat? (Spot.getD i.record "population") = some n ∧
      (Spot.Slots.popOk i.params.verifier.program i.params.verifier.query n = true ∨
        (Spot.Slots.popCovers i.params.verifier.program i.params.verifier.query n = true ∧
          (∀ g r k, i.rule ≠ .twoStage g r k) ∧
          ∃ rule sd wt, i.rule.over n
              (Spot.Select.strataSizes i.params.verifier.program i.params.verifier.query) = some rule ∧
            rule.samplingDesignOver n = .ok sd ∧
            i.rule.tableOver i.params.verifier.program i.params.verifier.query i.params.verifier.workTable = .ok wt ∧
            Spot.OneStageZk.subsetDesign { i.params.verifier with samplingDesign := sd, workTable := wt }
              i.record = true)) := by
  unfold Spot.Run.run at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat'
    first
    | exact absurd rfl ‹¬true = true›
    | split at h
    | focus (cases h; done)
    | (obtain ⟨_, h⟩ | ⟨_, h⟩ := Spot.Run.ite_eq_ok h; exact Or.inl ⟨_, _, h.symm⟩)
    | exact Or.inl ⟨_, _, (Except.ok.inj h).symm⟩
    | exact Or.inr ⟨_, by assumption, Or.inl (true_of_not ‹¬(!Spot.Slots.popOk _ _ _) = true›)⟩
    | exact Or.inr ⟨_, by assumption, Or.inr ⟨true_of_not ‹¬(!Spot.Slots.popCovers _ _ _) = true›,
        by assumption, _, _, _, by assumption, by assumption, by assumption,
        true_of_not ‹¬(!Spot.OneStageZk.subsetDesign _ _) = true›⟩⟩
    | (obtain ⟨_, h⟩ | ⟨_, h⟩ := Spot.Run.ite_eq_cases h)

/-- **An accepted two-stage run passed its population check** at the population its boundary record states: the run
takes a cover only under a one-stage rule (`run_popHeld`). -/
theorem population_of_accepted {i : Spot.Run.Input} {it : Spot.Run.Interior} {j : Json} (ha : AcceptedTwo i it j) :
    ∃ n, Spot.nat? (Spot.getD i.record "population") = some n ∧
      Spot.Slots.popOk i.params.verifier.program i.params.verifier.query n = true := by
  rcases run_popHeld ha.run with ⟨step, codes, hj⟩ | ⟨n, hn, hp | ⟨-, ht, -⟩⟩
  · have hs := ha.accept
    rw [hj, symbol_refused] at hs
    cases hs
  · exact ⟨n, hn, hp⟩
  · obtain ⟨g, r, k, hr⟩ := ha.two
    exact absurd hr (ht g r k)

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
