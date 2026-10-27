import Definitions.Accounting.TS.Stages

/-!
# Per-stage coins' named assumption (trusted)

STEER-PROOF's route A: the one premise per-stage coins add to the tensor–scalar
model's, a `Prop` over `Accounting.TS.StageWindow`, taken as a hypothesis, never an `axiom`.

* `StageLuck`: (L) at `p` per unit at its stage's coin, the per-stage form of (I7) `LuckAt`. Under one σ the same
  statement is false for a steered consumer (it is cheap and passes with probability 1), which is why one σ needs a
  closure and a steering bound (`Accounting.Guarantees.TS.SegmentSteer`, `SteerUnion`); with each stage's coin released
  after its operands are committed there is no such unit.
-/

namespace Accounting.Assumptions.TS

open Accounting.TS Pouw.RelativeSecurity

/-- `accounting-ts-stage-luck` ((L) per unit, per-stage coins). **(L) at `p` under per-stage coins**: in every round,
at every stage, whatever the past, a unit of the stage that the cheater skips (does not pay for) passes its proof with
probability at most `p` over the stage's coin: `Pr[skip_u ∧ ok_u] ≤ p·Pr[skip_u]`. The unit's operands are committed
before the coin, so its value is paid for or is a bet on the coin: none is made cheap and right by a choice of earlier
values, which under one σ is ε_adapt. -/
def StageLuck {C U D : Type} [Fintype C] {M s n : ℕ} (p : ℝ) (stg : U → ℕ) (W : StageWindow C U D M s n) : Prop :=
  W.LuckAt p stg

end Accounting.Assumptions.TS
