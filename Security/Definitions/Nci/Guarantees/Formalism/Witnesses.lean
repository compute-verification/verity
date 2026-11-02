import Definitions.Nci.Guarantees.Formalism
import Definitions.Nci.Formalism.Instances

/-!
# The training formalism v1: honest and nontrivial lemmas

Slowdown, its two cases and Security each have two statements here, at the instances of
`Security/Definitions/Nci/Formalism/Instances.lean`. `<Guarantee>_honest`: its premises other than the named assumptions
hold together at `Honest`, and so does A1, at `c = 1` and the update profile `u(x) = x` (and, for Security, A2).
`<Guarantee>_nontrivial`: at `Counter` every premise but A1 (`CutCost`) holds, A1 fails, and the bound fails, its last
conjunct the bound's negation: the block of two gates does all of the token's dependent work and reads no computed
value, so `ρ ≤ 1` at `c = 2`, below each bound (`2`, `4`, `2` and, for Security, `4 > N n c = 2`). Fit concludes an
equation, `ρ = 1`, not a bound, and has no such lemmas.

`Honest` is a small faithful instance of a training step and a token run on its result: a Boolean declaration with a
registered weight, a data bit, the new weight computed from both and a token computed from it and the data, each gate
a block. It is accepted as the honest instance of these structural bounds, which are about one circuit: it rests on no run
and no `art:`, and a training run's annotation, which no check reads yet, would supply the deployed premises.

The proofs are in `Security/Proofs/Nci/Formalism/Witnesses.lean`.
-/

namespace Nci.Formalism.Guarantees

open Nci.Formalism Nci.Formalism.Assumptions
open scoped NNReal ENNReal

/-- **Slowdown's premises hold** at `Honest`, A1 among them, at `(c, k) = (1, 2)`, `u(x) = x`, `a = 0`, `x₀ = 1`, `ε = 0`
and `M = 1`. -/
def Slowdown_honest : Prop :=
  Honest.A.Valid ∧ CutCost Honest.d Honest.A 1 (fun x => (x : ℝ)) 0 ∧ Honest.A.TokensLinked 0 ∧
    PeakBound (fun x => (x : ℝ)) 2 1 ∧ BottomShare (fun x => (x : ℝ)) 2 1 ∧ (0 : ℝ) ≤ 0

/-- **Slowdown is false without A1**: at `Counter`, with `(c, k) = (2, 0)`, `u(x) = x`, `a = 2`, `x₀ = ε = 0` and
`M = 1`, every other premise holds, A1 fails, and the bound `2` is above the slowdown, at most `1`. -/
def Slowdown_nontrivial : Prop :=
  Counter.A.Valid ∧ ¬ CutCost Counter.d Counter.A 2 (fun x => (x : ℝ)) 2 ∧ Counter.A.TokensLinked 0 ∧
    PeakBound (fun x => (x : ℝ)) 0 1 ∧ BottomShare (fun x => (x : ℝ)) 0 0 ∧ (0 : ℝ) ≤ 2 ∧
    ¬ ENNReal.ofReal (Counter.A.depShare * (1 - 0 - 0) * Counter.A.tokensPerBlock 2 * 2 / 1) ≤ slowdown Counter.d 2 0

/-- **The evenly spread case's premises hold** at `Honest`, A1 among them, at `(c, k) = (1, 1)`, `u(1) = 1`, `a = 0` and
`ε = 0`. -/
def SlowdownEven_honest : Prop :=
  Honest.A.Valid ∧ CutCost Honest.d Honest.A 1 (fun x => (x : ℝ) * 1) 0 ∧ Honest.A.TokensLinked 0 ∧ 0 < (1 : ℕ) ∧
    ((1 : ℕ) : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 0

/-- **The evenly spread case is false without A1**: at `Counter`, with `(c, k) = (2, 1)`, `u(1) = 2`, `a = 1` and
`ε = 0`, every other premise holds, A1 fails, and the bound `4` is above the slowdown, at most `1`. -/
def SlowdownEven_nontrivial : Prop :=
  Counter.A.Valid ∧ ¬ CutCost Counter.d Counter.A 2 (fun x => (x : ℝ) * 2) 1 ∧ Counter.A.TokensLinked 0 ∧
    0 < (1 : ℕ) ∧ ((1 : ℕ) : ℝ) ≤ 2 ∧ (0 : ℝ) ≤ 1 ∧
    ¬ ENNReal.ofReal (4 * Counter.A.depShare * Counter.A.tokensPerBlock 2 * 1 * (2 - ((1 : ℕ) : ℝ) - 0 * 2) /
      ((1 : ℕ) : ℝ) ^ 2) ≤ slowdown Counter.d 2 1

/-- **The frozen-tail case's premises hold** at `Honest`, A1 among them, at `(c, k) = (1, 2)`, `u(x) = x`, `a = 0`,
`t = 1` and `ε = 0`. -/
def SlowdownFrozenTail_honest : Prop :=
  Honest.A.Valid ∧ CutCost Honest.d Honest.A 1 (fun x => (x : ℝ)) 0 ∧ Honest.A.TokensLinked 0 ∧
    FrozenTail (fun x => (x : ℝ)) 2 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 0

/-- **The frozen-tail case is false without A1**: at `Counter`, with `(c, k) = (2, 1)`, `u = 2`, `a = 2`, `t = 1/2` and
`ε = 0`, every other premise holds, A1 fails, and the bound `2` is above the slowdown, at most `1`. -/
def SlowdownFrozenTail_nontrivial : Prop :=
  Counter.A.Valid ∧ ¬ CutCost Counter.d Counter.A 2 (fun _ => 2) 2 ∧ Counter.A.TokensLinked 0 ∧
    FrozenTail (fun _ => 2) 1 (1 / 2) ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) ≤ 2 ∧
    ¬ ENNReal.ofReal (Counter.A.depShare * (1 - 1 / 2 - 0) * Counter.A.tokensPerBlock 2 * 2 /
      (1 / 2 * ((1 : ℕ) : ℝ))) ≤ slowdown Counter.d 2 1

/-- **Security's premises hold** at `Honest`, A1 and A2 among them: its run the one method, with the decoder `silent`,
`D = ⊤`, the loss `0` and `ℓ* = 0`, `p = 1`, its two blocks in one period of two, `(c, k) = (1, 2)`, `u(x) = x`, `a = 0`,
`x₀ = 1`, `ε = 0` and `M = 1`. -/
def Security_honest : Prop :=
  EfficiencyFloor Honest.methods silent 1 (fun _ => 0) 0 ⊤ 1 ∧ Honest.r.Among Honest.methods ∧ Honest.r.A.Valid ∧
    Honest.r.Within silent 1 ⊤ ∧ Honest.r.Reaches (fun _ => 0) 0 ∧
    CutCost Honest.r.d Honest.r.A 1 (fun x => (x : ℝ)) 0 ∧ Honest.r.A.TokensLinked 0 ∧
    PeakBound (fun x => (x : ℝ)) 2 1 ∧ BottomShare (fun x => (x : ℝ)) 2 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    IsCKPartition Honest.r.d Honest.P 1 2 ∧ InPeriods 2 1 2

/-- **Security is false without A1**: at `Counter`, its run the one method, with the decoder `silent`, `D = ⊤`, the loss
`0` and `ℓ* = 0`, `p = 1`, its one block in one period of one, `(c, k) = (2, 0)`, `u(x) = x`, `a = 2`, `x₀ = ε = 0` and
`M = 1`, every other premise holds, A2 among them, A1 fails, and `ρ̲ p H(ℓ*) = 4` is above `N n c = 2`. -/
def Security_nontrivial : Prop :=
  EfficiencyFloor Counter.methods silent 2 (fun _ => 0) 0 ⊤ 1 ∧ Counter.r.Among Counter.methods ∧
    Counter.r.A.Valid ∧ Counter.r.Within silent 2 ⊤ ∧ Counter.r.Reaches (fun _ => 0) 0 ∧
    ¬ CutCost Counter.r.d Counter.r.A 2 (fun x => (x : ℝ)) 2 ∧ Counter.r.A.TokensLinked 0 ∧
    PeakBound (fun x => (x : ℝ)) 0 1 ∧ BottomShare (fun x => (x : ℝ)) 0 0 ∧ (0 : ℝ) ≤ 2 ∧ (0 : ℝ) ≤ 1 ∧
    IsCKPartition Counter.r.d Counter.P 2 0 ∧ InPeriods 1 1 1 ∧
    ¬ ENNReal.ofReal (Counter.r.A.depShare * (1 - 0 - 0) * Counter.r.A.tokensPerBlock 2 * 2 / 1 * 1) *
      honestCost Counter.methods (fun _ => 0) 0 ≤ ((1 * 1 * 2 : ℕ) : ℝ≥0∞)

end Nci.Formalism.Guarantees
