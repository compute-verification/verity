import Definitions.Nci.Guarantees.Formalism
import Definitions.Nci.Formalism.Instances

/-!
# The training formalism v2: honest and nontrivial lemmas

The counting lemma, Slowdown, Security and Fit's two bounds each have two statements here, at the instances of
`Security/Definitions/Nci/Formalism/Instances.lean`. `<Guarantee>_honest`: its premises hold together, the named
assumptions among them. `<Guarantee>_nontrivial`: every premise but one holds, that one fails, and the bound fails, its
last conjunct the bound's negation.

The counting lemma, Slowdown and Security are at `Chain`, at `(c, k) = (1, 1)`: a token of three gates in a chain, of
which only the first is computed from the token's inputs. Each premise holds at `blocks`, four blocks of one gate,
priced at `1`. A1 holds at `Chain.entry`, `Chain.state` and `Chain.update`: gate `2`'s block reads the new weight, one
update bit, and gates `3` and `4`'s each read one value of the token's state. The token kind's work is `X = {3, 4}`, the
token's gates no block computing it from its inputs holds, with the cap `1`, and Slowdown's other kind is the new
weight, gate `1`, with the cap `1`, so the bound is `max(1/2, 1/4) = 1/2`.

* The counting lemma drops the partition into inference blocks: at the kind `{3, 4}` with the cap `1`, its bound `1/2`
  is above `whole`'s price `1/4`, and `whole`'s one block of four gates is no inference block at `c = 1`.
* Slowdown and Security drop A1 (`CutCost`): at twice the state, which the block of gate `3` does not pay, A1's budget
  admits no block that holds a gate of `X`, the cap `0` bounds the token work within it, and the bound is `⊤`, above
  `blocks`' price `1` and, for Security, above `N n c = 4`.
* Fit's bounds are at `Honest`: its two blocks fit at `k = 2`, and at `k = 1` they don't and no partition into inference
  blocks exists, since gate `3` reads two values, so `ρ = ⊤`, above `c / (c - d) = 1`.

`FitHonestServing` concludes an equation, not a bound, and `TokenCap` and `TokenCapClean` read no partition: none of the
three has such lemmas.

`Chain` and `Honest` are small faithful instances of a training step and of serving, accepted as the honest instances of
these structural bounds: they rest on no run and no `art:`, and a training run's annotation and checked witness blocks,
which no check reads yet, would supply the deployed premises.

The proofs are in `Security/Proofs/Nci/Formalism/Witnesses.lean`.
-/

namespace Nci.Formalism.Guarantees

open Nci.Formalism Nci.Formalism.Assumptions
open scoped NNReal ENNReal

/-- **The counting lemma's premises hold** at `Chain`: the kind `{3, 4}` has the cap `1`, and `blocks` is a partition
into inference blocks at `(c, k) = (1, 1)`. -/
def Counting_honest : Prop :=
  Caps Chain.d 1 1 (fun _ : Unit => {3, 4}) (fun _ => 1) ∧ Chain.blocks.IsCK Chain.d 1 1

/-- **The counting lemma is false without a partition into inference blocks**: at `Chain`, `whole` is none at
`(c, k) = (1, 1)`, and the bound `1/2` of the kind `{3, 4}` with the cap `1` is above its price `1/4`. -/
def Counting_nontrivial : Prop :=
  Caps Chain.d 1 1 (fun _ : Unit => {3, 4}) (fun _ => 1) ∧ ¬ Chain.whole.IsCK Chain.d 1 1 ∧
    ¬ (⨆ _K : Unit, kindBound Chain.d 1 {3, 4} 1) ≤ partitionSlowdown Chain.d 1 1

/-- **Slowdown's premises hold** at `Chain`, A1 among them, at `(c, k) = (1, 1)`: its annotation is valid, A1 holds at
`Chain.entry`, `Chain.state` and `Chain.update`, the cap `1` bounds the token work in `{3, 4}` within A1's budget, the
new weight has the cap `1`, and `blocks` is a partition into inference blocks. -/
def Slowdown_honest : Prop :=
  Chain.A.Valid ∧ CutCost Chain.d 1 1 Chain.entry Chain.state Chain.update ∧
    Chain.A.TokenCapAt 1 1 Chain.entry Chain.state Chain.update {3, 4} 1 ∧
    Caps Chain.d 1 1 (fun _ : Unit => {1}) (fun _ => 1) ∧ Chain.blocks.IsCK Chain.d 1 1

/-- **Slowdown is false without A1**: at `Chain`, at twice the state and the rest as in `Slowdown_honest`, every other
premise holds, the cap `0` bounding the token work in `{3, 4}` within A1's budget, A1 fails at the block of gate `3`,
and the bound `⊤` is above `blocks`' price `1`. -/
def Slowdown_nontrivial : Prop :=
  Chain.A.Valid ∧ ¬ CutCost Chain.d 1 1 Chain.entry (fun T x => 2 * Chain.state T x) Chain.update ∧
    Chain.A.TokenCapAt 1 1 Chain.entry (fun T x => 2 * Chain.state T x) Chain.update {3, 4} 0 ∧
    Caps Chain.d 1 1 (fun _ : Unit => {1}) (fun _ => 1) ∧ Chain.blocks.IsCK Chain.d 1 1 ∧
    ¬ max (kindBound Chain.d 1 {3, 4} 0) (⨆ _K : Unit, kindBound Chain.d 1 {1} 1) ≤ partitionSlowdown Chain.d 4 1

/-- **Security's premises hold** at `Chain`, A1 and A2 among them: its run the one method, with the decoder `silent`,
`D = ⊤`, the loss `0` and `ℓ* = 0`, `p = 1`, its four blocks in one period of four, and Slowdown's premises as in
`Slowdown_honest`. -/
def Security_honest : Prop :=
  EfficiencyFloor Chain.methods silent 1 (fun _ => 0) 0 ⊤ 1 ∧ Chain.r.Among Chain.methods ∧ Chain.r.A.Valid ∧
    Chain.r.Within silent 1 ⊤ ∧ Chain.r.Reaches (fun _ => 0) 0 ∧
    CutCost Chain.r.d 1 1 Chain.entry Chain.state Chain.update ∧
    Chain.r.A.TokenCapAt 1 1 Chain.entry Chain.state Chain.update {3, 4} 1 ∧
    Caps Chain.r.d 1 1 (fun _ : Unit => {1}) (fun _ => 1) ∧ Chain.blocks.IsCK Chain.r.d 1 1 ∧ InPeriods 4 1 4

/-- **Security is false without A1**: at `Chain`, at twice the state and the rest as in `Security_honest`, every other
premise holds, A2 among them, A1 fails at the block of gate `3`, and `ρ̲ p H(ℓ*) = ⊤` is above `N n c = 4`. -/
def Security_nontrivial : Prop :=
  EfficiencyFloor Chain.methods silent 1 (fun _ => 0) 0 ⊤ 1 ∧ Chain.r.Among Chain.methods ∧ Chain.r.A.Valid ∧
    Chain.r.Within silent 1 ⊤ ∧ Chain.r.Reaches (fun _ => 0) 0 ∧
    ¬ CutCost Chain.r.d 1 1 Chain.entry (fun T x => 2 * Chain.state T x) Chain.update ∧
    Chain.r.A.TokenCapAt 1 1 Chain.entry (fun T x => 2 * Chain.state T x) Chain.update {3, 4} 0 ∧
    Caps Chain.r.d 1 1 (fun _ : Unit => {1}) (fun _ => 1) ∧ Chain.blocks.IsCK Chain.r.d 1 1 ∧ InPeriods 4 1 4 ∧
    ¬ max (kindBound Chain.r.d 1 {3, 4} 0) (⨆ _K : Unit, kindBound Chain.r.d 1 {1} 1) * ENNReal.ofReal 1 *
      honestCost Chain.methods (fun _ => 0) 0 ≤ ((1 * 4 * 1 : ℕ) : ℝ≥0∞)

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
