module

public import Proofs.Flock.SpotFlock.RunsAs.Statement
public import Proofs.Flock.SpotFlock.Jeq
public import Proofs.Properties.Verity.SampledTwo.Steps
public import VStarRecursive.SampledTwo

/-!
# The two-stage form's check, accepting, accepted a two-stage run

Where the two-stage form's check (`VStar.Recursive.SampledTwo.checkBlobsAtWith`) accepts, it ran `Spot.Run.run` with
the two-stage form allowed on the input `inputAt` built, and the run's symbol is `accept`
(`SampledTwo.accepted_steps`). That input carries the parameters' rule and an interior registration (`inputOf`), so where
the parameters' rule is two-stage, which the parameters' parser requires, the run is an accepted two-stage run
(`AcceptedTwo`): the premise `spotRunAcceptsPlay` reads (`acceptedTwo_of_check`).
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

/-- **The form's check, accepting, accepted a two-stage run**: at parameters whose rule is two-stage, an accept of
`checkBlobsAtWith form` is at the switch on, and the run of record `inputAt` built the input of is an accepted
two-stage run of `Run.run` with the form allowed (`AcceptedTwo`), whose first stage's bytes run no further than what it
read (`sourceRead`). -/
theorem acceptedTwo_of_check {form : Bool} {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment}
    {ξ : ByteArray} {e : VStar.Recursive.SampledTwo.Evidence} {bs pub prof}
    (hp : ∃ grain replay k, p.rule = .twoStage grain replay k)
    (h : VStar.Recursive.SampledTwo.checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    form = true ∧ ∃ c₂ ξ₂ r i it run, VStar.Recursive.SampledTwo.inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧
      AcceptedTwo i it run ∧ VStar.Recursive.SampledTwo.sourceRead ξ run = .ok () := by
  obtain ⟨hf, c₂, ξ₂, r, i, run, hin, hrun, hsr, -, hj, -⟩ := VStar.Recursive.SampledTwo.accepted_steps h
  obtain ⟨-, -, -, -, vs, received₂, j₁, j₂, -, rfl, -, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  exact ⟨hf, c₂, ξ₂, r, _, _, run, hin, ⟨hrun, str_of_jeq (jeq_symm hj), hp, rfl⟩, hsr⟩

end SpotFlock.RunsAs
