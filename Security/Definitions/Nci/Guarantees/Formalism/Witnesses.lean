import Definitions.Nci.Guarantees.Formalism
import Definitions.Nci.Formalism.Instances

/-!
# The training formalism v2: honest and nontrivial lemmas

The bounds on `ρ`, the claim and its two cases, Security and Fit's two bounds each have two statements here, at the
instances of `Security/Definitions/Nci/Formalism/Instances.lean`. `<Guarantee>_honest`: its premises hold together, the
named assumptions among them. `<Guarantee>_nontrivial`: every premise but one holds, that one fails, and the bound fails,
its last conjunct the bound's negation.

The bounds on `ρ`, the claim and Security are at `Chain`, at `(c, k) = (1, 1)`: a token of three gates in a chain, of
which an inference block reaches only the first, so `σ = 3/4`, `s = 1/3`, `s* = 1/3` and every reachable share and
dependency profile at `1/3` is `1/3`. Each premise holds at `blocks`, four blocks of one gate, priced at `1`.

* The four bounds that take no assumption (`Counting`, `Slowdown`, `TokenForm`, `TokenFormVarying`) drop the partition
  into inference blocks: each bound is `1/2`, above `whole`'s price `1/4`, and `whole`'s one block of four gates is no
  inference block at `c = 1`.
* The claim, its two cases and Security drop A1 (`CutCost`): at a cut cost `a` that the block of gate `3`, which cuts
  the token and reads one computed value, does not pay, the bound is above `blocks`' price (`2`, `2` and `3/2`), and
  for Security `ρ̲ p H(ℓ*) = 8` is above `N n c = 4`.
* Fit's bounds are at `Honest`: its two blocks fit at `k = 2`, and at `k = 1` they don't and no partition into inference
  blocks exists, since gate `3` reads two values, so `ρ = ⊤`, above `c / (c - d) = 1`.

`FitHonestServing` concludes an equation, not a bound, and `CapacityBound` and `ReachableShareBound` read no partition:
none of the three has such lemmas.

`Chain` and `Honest` are small faithful instances of a training step and of serving, accepted as the honest instances of
these structural bounds: they rest on no run and no `art:`, and a training run's annotation, which no check reads yet,
would supply the deployed premises.

The proofs are in `Security/Proofs/Nci/Formalism/Witnesses.lean`.
-/

namespace Nci.Formalism.Guarantees

open Nci.Formalism Nci.Formalism.Assumptions
open scoped NNReal ENNReal

/-- **The counting lemma's premises hold** at `Chain`: `S = {3, 4}` is a set of gates, and `blocks` is a partition into
inference blocks at `(c, k) = (1, 1)`. -/
def Counting_honest : Prop :=
  GateSet Chain.d.D {3, 4} ∧ Chain.blocks.IsCK Chain.d 1 1

/-- **The counting lemma is false without a partition into inference blocks**: at `Chain`, `whole` is none at
`(c, k) = (1, 1)`, and the bound `1/2` at `S = {3, 4}` is above its price `1/4`. -/
def Counting_nontrivial : Prop :=
  GateSet Chain.d.D {3, 4} ∧ ¬ Chain.whole.IsCK Chain.d 1 1 ∧
    ¬ (((({3, 4} : Finset (Fin Chain.d.D.N)).card : ℝ≥0∞) / (Chain.d.D.computed.card : ℝ≥0∞)) /
      ((maxInBlock Chain.d 1 1 {3, 4} : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ≤ partitionSlowdown Chain.d 1 1)

/-- **Slowdown's premise holds** at `Chain`: `blocks` is a partition into inference blocks at `(c, k) = (1, 1)`. -/
def Slowdown_honest : Prop :=
  Chain.blocks.IsCK Chain.d 1 1

/-- **Slowdown is false without a partition into inference blocks**: at `Chain`, `whole` is none at `(c, k) = (1, 1)`,
and the bound `1/2` at `A \ R` is above its price `1/4`. -/
def Slowdown_nontrivial : Prop :=
  ¬ Chain.whole.IsCK Chain.d 1 1 ∧
    ¬ ((((Chain.A.affected \ Chain.A.reachable 1 1).card : ℝ≥0∞) / (Chain.d.D.computed.card : ℝ≥0∞)) /
      ((maxInBlock Chain.d 1 1 (Chain.A.affected \ Chain.A.reachable 1 1) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ≤
        partitionSlowdown Chain.d 1 1)

/-- **The token form's premises hold** at `Chain`: its token blocks have `3` affected gates each, every reachable share
is at most `1/3`, and `blocks` is a partition into inference blocks at `(c, k) = (1, 1)`. -/
def TokenForm_honest : Prop :=
  Chain.A.Identical 3 ∧ Chain.A.ReachableAtMost 1 1 (1 / 3) ∧ Chain.blocks.IsCK Chain.d 1 1

/-- **The token form is false without a partition into inference blocks**: at `Chain`, every other premise holds,
`whole` is no partition into inference blocks at `(c, k) = (1, 1)`, and the bound `1/2` is above its price `1/4`. -/
def TokenForm_nontrivial : Prop :=
  Chain.A.Identical 3 ∧ Chain.A.ReachableAtMost 1 1 (1 / 3) ∧ ¬ Chain.whole.IsCK Chain.d 1 1 ∧
    ¬ (((Chain.A.tokenShare * (1 - 1 / 3) * Chain.A.tokensPerBlock 1 : ℝ≥0) : ℝ≥0∞) /
      (Chain.A.unreachableCapacity 1 1 : ℝ≥0∞) ≤ partitionSlowdown Chain.d 1 1)

/-- **The varying token form's premises hold** at `Chain`: every reachable share is at most `1/3`, and `blocks` is a
partition into inference blocks at `(c, k) = (1, 1)`. -/
def TokenFormVarying_honest : Prop :=
  Chain.A.ReachableAtMost 1 1 (1 / 3) ∧ Chain.blocks.IsCK Chain.d 1 1

/-- **The varying token form is false without a partition into inference blocks**: at `Chain`, every reachable share is
at most `1/3`, `whole` is no partition into inference blocks at `(c, k) = (1, 1)`, and the bound `1/2` is above its
price `1/4`. -/
def TokenFormVarying_nontrivial : Prop :=
  Chain.A.ReachableAtMost 1 1 (1 / 3) ∧ ¬ Chain.whole.IsCK Chain.d 1 1 ∧
    ¬ ((((1 - 1 / 3) * (Chain.A.affected ∩ Chain.A.tokenGates).card * ((1 : ℕ) : ℝ≥0) : ℝ≥0) : ℝ≥0∞) /
      ((Chain.d.D.computed.card * Chain.A.maxAffected * Chain.A.unreachableCapacity 1 1 : ℝ≥0) : ℝ≥0∞) ≤
        partitionSlowdown Chain.d 1 1)

/-- **The claim's premises hold** at `Chain`, A1 and A1′ among them, at `(c, k) = (1, 1)`: `u = 0` with `a = 1` and
`M = 1`, `u_in(x) = 3 x` with `ξ = 1/3`, and `x₀ = 1/3`. -/
def SlowdownClaim_honest : Prop :=
  Chain.A.Valid ∧ Chain.A.Identical 3 ∧ CutCost Chain.d Chain.A 1 1 (fun _ => 0) 1 ∧ PeakBound (fun _ => 0) 1 1 ∧
    InputCost Chain.d Chain.A 1 1 (fun x => x * 3) ∧ ReachLimit (fun x => x * 3) 1 (1 / 3) ∧
    Chain.A.ProfileAtMost (1 / 3) (1 / 3) ∧ Chain.blocks.IsCK Chain.d 1 1

/-- **The claim is false without A1**: at `Chain`, with `a = 12` and the rest as in `SlowdownClaim_honest`, every other
premise holds, A1 fails at the block of gate `3`, and the bound `2` is above `blocks`' price `1`. -/
def SlowdownClaim_nontrivial : Prop :=
  Chain.A.Valid ∧ Chain.A.Identical 3 ∧ ¬ CutCost Chain.d Chain.A 1 1 (fun _ => 0) 12 ∧ PeakBound (fun _ => 0) 1 1 ∧
    InputCost Chain.d Chain.A 1 1 (fun x => x * 3) ∧ ReachLimit (fun x => x * 3) 1 (1 / 3) ∧
    Chain.A.ProfileAtMost (1 / 3) (1 / 3) ∧ Chain.blocks.IsCK Chain.d 1 1 ∧
    ¬ (((Chain.A.tokenShare * (1 - 1 / 3) * Chain.A.tokensPerBlock 1 * 12 : ℝ≥0) : ℝ≥0∞) / ((1 : ℝ≥0) : ℝ≥0∞) ≤
      partitionSlowdown Chain.d 4 1)

/-- **The evenly spread case's premises hold** at `Chain`, A1 and A1′ among them, at `(c, k) = (1, 1)`, `u(1) = 3`,
`a = 0` and `x₀ = 1/3`. -/
def SlowdownEven_honest : Prop :=
  Chain.A.Valid ∧ Chain.A.Identical 3 ∧ CutCost Chain.d Chain.A 1 1 (fun x => x * 3) 0 ∧
    InputCost Chain.d Chain.A 1 1 (fun x => x * 3) ∧ (0 : ℝ≥0) < 3 ∧
    Chain.A.ProfileAtMost (((1 : ℕ) : ℝ≥0) / 3) (1 / 3) ∧ Chain.blocks.IsCK Chain.d 1 1

/-- **The evenly spread case is false without A1**: at `Chain`, with `a = 1` and the rest as in `SlowdownEven_honest`,
every other premise holds, A1 fails at the block of gate `3`, and the bound `2` is above `blocks`' price `1`. -/
def SlowdownEven_nontrivial : Prop :=
  Chain.A.Valid ∧ Chain.A.Identical 3 ∧ ¬ CutCost Chain.d Chain.A 1 1 (fun x => x * 3) 1 ∧
    InputCost Chain.d Chain.A 1 1 (fun x => x * 3) ∧ (0 : ℝ≥0) < 3 ∧
    Chain.A.ProfileAtMost (((1 : ℕ) : ℝ≥0) / 3) (1 / 3) ∧ Chain.blocks.IsCK Chain.d 1 1 ∧
    ¬ (((4 * Chain.A.tokenShare * (1 - 1 / 3) * Chain.A.tokensPerBlock 1 * 1 * 3 : ℝ≥0) : ℝ≥0∞) /
      ((((1 : ℕ) : ℝ≥0) ^ 2 : ℝ≥0) : ℝ≥0∞) ≤ partitionSlowdown Chain.d 4 1)

/-- **The frozen case's premises hold** at `Chain`, A1 and A1′ among them, at `(c, k) = (1, 1)`: `u(x) = u_in(x) = 3 x`
with `a = 0`, above `k` beyond the frozen share `t = 1/3`, `ξ = 1/3` and `x₀ = 1/3`. -/
def SlowdownFrozen_honest : Prop :=
  Chain.A.Valid ∧ Chain.A.Identical 3 ∧ CutCost Chain.d Chain.A 1 1 (fun x => x * 3) 0 ∧
    FrozenShare (fun x => x * 3) 1 (1 / 3) ∧ InputCost Chain.d Chain.A 1 1 (fun x => x * 3) ∧
    ReachLimit (fun x => x * 3) 1 (1 / 3) ∧ Chain.A.ProfileAtMost (1 / 3) (1 / 3) ∧ Chain.blocks.IsCK Chain.d 1 1

/-- **The frozen case is false without A1**: at `Chain`, with `a = 3` and the rest as in `SlowdownFrozen_honest`, every
other premise holds, A1 fails at the block of gate `3`, and the bound `3/2` is above `blocks`' price `1`. -/
def SlowdownFrozen_nontrivial : Prop :=
  Chain.A.Valid ∧ Chain.A.Identical 3 ∧ ¬ CutCost Chain.d Chain.A 1 1 (fun x => x * 3) 3 ∧
    FrozenShare (fun x => x * 3) 1 (1 / 3) ∧ InputCost Chain.d Chain.A 1 1 (fun x => x * 3) ∧
    ReachLimit (fun x => x * 3) 1 (1 / 3) ∧ Chain.A.ProfileAtMost (1 / 3) (1 / 3) ∧ Chain.blocks.IsCK Chain.d 1 1 ∧
    ¬ (((Chain.A.tokenShare * (1 - 1 / 3) * Chain.A.tokensPerBlock 1 * 3 : ℝ≥0) : ℝ≥0∞) /
      ((1 / 3 * ((1 : ℕ) : ℝ≥0) : ℝ≥0) : ℝ≥0∞) ≤ partitionSlowdown Chain.d 4 1)

/-- **Security's premises hold** at `Chain`, A1, A1′ and A2 among them: its run the one method, with the decoder
`silent`, `D = ⊤`, the loss `0` and `ℓ* = 0`, `p = 1`, its four blocks in one period of four, and the claim's premises as
in `SlowdownClaim_honest`. -/
def Security_honest : Prop :=
  EfficiencyFloor Chain.methods silent 1 (fun _ => 0) 0 ⊤ 1 ∧ Chain.r.Among Chain.methods ∧ Chain.r.A.Valid ∧
    Chain.r.Within silent 1 ⊤ ∧ Chain.r.Reaches (fun _ => 0) 0 ∧ Chain.r.A.Identical 3 ∧
    CutCost Chain.r.d Chain.r.A 1 1 (fun _ => 0) 1 ∧ PeakBound (fun _ => 0) 1 1 ∧
    InputCost Chain.r.d Chain.r.A 1 1 (fun x => x * 3) ∧ ReachLimit (fun x => x * 3) 1 (1 / 3) ∧
    Chain.r.A.ProfileAtMost (1 / 3) (1 / 3) ∧ Chain.blocks.IsCK Chain.r.d 1 1 ∧ InPeriods 4 1 4

/-- **Security is false without A1**: at `Chain`, with `a = 12` and the rest as in `Security_honest`, every other premise
holds, A2 among them, A1 fails at the block of gate `3`, and `ρ̲ p H(ℓ*) = 8` is above `N n c = 4`. -/
def Security_nontrivial : Prop :=
  EfficiencyFloor Chain.methods silent 1 (fun _ => 0) 0 ⊤ 1 ∧ Chain.r.Among Chain.methods ∧ Chain.r.A.Valid ∧
    Chain.r.Within silent 1 ⊤ ∧ Chain.r.Reaches (fun _ => 0) 0 ∧ Chain.r.A.Identical 3 ∧
    ¬ CutCost Chain.r.d Chain.r.A 1 1 (fun _ => 0) 12 ∧ PeakBound (fun _ => 0) 1 1 ∧
    InputCost Chain.r.d Chain.r.A 1 1 (fun x => x * 3) ∧ ReachLimit (fun x => x * 3) 1 (1 / 3) ∧
    Chain.r.A.ProfileAtMost (1 / 3) (1 / 3) ∧ Chain.blocks.IsCK Chain.r.d 1 1 ∧ InPeriods 4 1 4 ∧
    ¬ (((Chain.r.A.tokenShare * (1 - 1 / 3) * Chain.r.A.tokensPerBlock 1 * 12 : ℝ≥0) : ℝ≥0∞) / ((1 : ℝ≥0) : ℝ≥0∞) *
      ENNReal.ofReal 1 * honestCost Chain.methods (fun _ => 0) 0 ≤ ((1 * 4 * 1 : ℕ) : ℝ≥0∞))

/-- **Fit's serving premises hold** at `Honest`: it is Boolean, and its two blocks of one gate are session blocks that
read a description of one value and one token's one data bit, `Δ + s τ = 2 ≤ k = 2`, with no decoder. -/
def FitServing_honest : Prop :=
  Honest.d.D.Boolean ∧ SessionBlocks Honest.d Honest.P 1 0 1 1 1 ∧ ServingFits 1 1 1 2

/-- **Fit is false for session blocks that don't fit**: at `Honest`, at `k = 1`, `Δ + s τ = 2` does not fit, gate `3`
reads two values, so no partition into inference blocks exists and `ρ = ⊤`, above `c / (c - d) = 1`. -/
def FitServing_nontrivial : Prop :=
  Honest.d.D.Boolean ∧ SessionBlocks Honest.d Honest.P 1 0 1 1 1 ∧ ¬ ServingFits 1 1 1 1 ∧
    ¬ slowdown Honest.d 1 1 ≤ ((1 : ℕ) : ℝ≥0∞) / ((1 - 0 : ℕ) : ℝ≥0∞)

/-- **Fit's training premises hold** at `Honest`: it is Boolean, and its two blocks of one gate are compute blocks that
fit at `k = 2`, with `Δ = Δ_grad = s = τ = 1` and no decoder. -/
def FitTraining_honest : Prop :=
  Honest.d.D.Boolean ∧ TrainingBlocks Honest.d Honest.P 1 0 1 1 1 1 ∧ TrainingFits 1 1 1 1 2

/-- **Fit is false for training blocks that don't fit**: at `Honest`, at `k = 1` the blocks don't fit, no partition into
inference blocks exists, and `ρ = ⊤`, above `c / (c - d) = 1`. -/
def FitTraining_nontrivial : Prop :=
  Honest.d.D.Boolean ∧ TrainingBlocks Honest.d Honest.P 1 0 1 1 1 1 ∧ ¬ TrainingFits 1 1 1 1 1 ∧
    ¬ slowdown Honest.d 1 1 ≤ ((1 : ℕ) : ℝ≥0∞) / ((1 - 0 : ℕ) : ℝ≥0∞)

end Nci.Formalism.Guarantees
