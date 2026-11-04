import Definitions.NetworkCertifier.Tied

/-!
# The deployment's parse of a phase's statements: a tie's two ends open at one position

A tie of a rows-form phase (`NetTiming.Ties`) is right by binding: the registrations commit its reader and its writer at
one position under one root (`NetTiming.Placed`). That makes them one value only if the deployment's parse of each
statement puts its registration's roots among the statement's `roots`, each gate at the position the registration
names, so that two passing openings of them are openings of one committed value. `SharedRootsParsed` states what that
gives, in binding's form (`CommitmentScheme.Binds`): two such openings that differ make the run's `H` calls collide.
It is a fact about the deployed parse, which the sampled leg's bridge proves at the deployed record; it is no
assumption about the period check, which checks the placements (`NetTiming.RowsStateHeld`, `NetTiming.RowsProgWired`).
-/

namespace NetTiming.Assumptions

open Claims Claims.Sampling

/-- **SharedRootsParsed**: at the deployment `D`, two gates of statements it parses that the registrations commit at one
position under one root (`place`) open to one value, or the two openings' `H` calls collide. -/
def SharedRootsParsed {Coin Col T : Type} (H : Msg → Msg) (D : Deploy Coin Col T) (place : Placement) : Prop :=
  ∀ (x x' : Msg) (s s' : AuditStatement) (g g' : ℕ) (p : Msg × ℕ), D.parse x = some s → D.parse x' = some s' →
    place s g = some p → place s' g' = some p →
      ∀ o o' : Opening Bool, o.pos = g → o'.pos = g' → o.val ≠ o'.val → D.commit.Opens H s.roots o = true →
        D.commit.Opens H s'.roots o' = true →
          LogColl (((D.commit.opens s.roots o).eval H).2 ++ ((D.commit.opens s'.roots o').eval H).2)

end NetTiming.Assumptions
