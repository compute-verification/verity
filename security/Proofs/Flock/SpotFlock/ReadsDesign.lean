module

public import Proofs.Flock.SpotFlock.Code

/-! Two clauses an accepted statement's check holds on its own: `second-inputs` and the design's derived links and
wires. -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable {h : Held}

theorem readsOk_inputs_design {t : ℕ} {x m₁ xS : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hr : readsOk h t x m₁ xS c₀ c₁ = true) :
    ∃ st rec draw, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some draw ∧
      Spot.SecondStage.inputsFailure (ownOf h st rec draw c₀) = none ∧
      Spot.SecondStage.designOk (partsAt h (ownOf h st rec draw c₀).sample.size) (ownOf h st rec draw c₀) =
        true := by
  unfold readsOk at hr
  cases hst : decode x with
  | none => rw [hst] at hr; cases hr
  | some st =>
    cases hrec : jsonMsg m₁ with
    | none => rw [hst, hrec] at hr; cases hr
    | some rec =>
      cases hdraw : secondDraw h c₀ c₁ with
      | none => rw [hst, hrec, hdraw] at hr; cases hr
      | some draw =>
        cases hw : FlockDeploy.decode xS with
        | none => rw [hst, hrec, hdraw, hw] at hr; cases hr
        | some w =>
          rw [hst, hrec, hdraw, hw] at hr
          simp only [Bool.and_eq_true] at hr
          obtain ⟨⟨⟨⟨-, hin⟩, hdes⟩, -⟩, -⟩ := hr
          exact ⟨st, rec, draw, rfl, rfl, rfl, Option.isNone_iff_eq_none.1 hin, hdes⟩

theorem inputs_of_readsOk {t : ℕ} {x m₁ xS : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hr : readsOk h t x m₁ xS c₀ c₁ = true) :
    ∃ st rec draw, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some draw ∧
      Spot.SecondStage.inputsFailure (ownOf h st rec draw c₀) = none := by
  obtain ⟨st, rec, draw, hst, hrec, hdraw, hin, -⟩ := readsOk_inputs_design hr
  exact ⟨st, rec, draw, hst, hrec, hdraw, hin⟩

theorem design_of_readsOk {t : ℕ} {x m₁ xS : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hr : readsOk h t x m₁ xS c₀ c₁ = true) :
    ∃ st rec draw, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some draw ∧
      Spot.SecondStage.designOk (partsAt h (ownOf h st rec draw c₀).sample.size) (ownOf h st rec draw c₀) =
        true := by
  obtain ⟨st, rec, draw, hst, hrec, hdraw, -, hdes⟩ := readsOk_inputs_design hr
  exact ⟨st, rec, draw, hst, hrec, hdraw, hdes⟩

end SpotFlock
