module

public import Definitions.Slowdown.Guarantees.Formalism
public import Definitions.Slowdown.Formalism.Instances

/-! The training formalism v2's honest and nontrivial lemmas, at small instances of a training step and of serving; `FitHonestServing` and `TokenCapClean` have none. -/

@[expose] public section

namespace Slowdown.Formalism.Guarantees

open Slowdown.Formalism Slowdown.Formalism.Assumptions
open scoped NNReal ENNReal

/-- **The counting lemma's premises hold** at `Chain`: the kind `{3, 4}` has the cap `1`, `1` is a partition-level
bound, and `blocks` is a partition into inference blocks at `(c, k) = (1, 1)`. -/
def Counting_honest : Prop :=
  Caps Chain.d 1 1 (fun _ : Unit => {3, 4}) (fun _ => 1) ∧ PartitionBound Chain.d 1 1 1 ∧
    Chain.blocks.IsCK Chain.d 1 1

/-- **The counting lemma is false without a partition into inference blocks**: at `Chain`, `whole` is none at
`(c, k) = (1, 1)`, and the bound `max(1/2, 1) = 1` of the kind `{3, 4}` with the cap `1` and the partition-level bound
`1` is above its price `1/4`. -/
def Counting_nontrivial : Prop :=
  Caps Chain.d 1 1 (fun _ : Unit => {3, 4}) (fun _ => 1) ∧ PartitionBound Chain.d 1 1 1 ∧
    ¬ Chain.whole.IsCK Chain.d 1 1 ∧
    ¬ max (⨆ _K : Unit, kindBound Chain.d 1 {3, 4} 1) (⨆ _J : Unit, (1 : ℝ≥0∞)) ≤ partitionSlowdown Chain.d 1 1

/-- **The token cap's premises hold** at `Chain`, at `(c, k) = (1, 1)`: A1 holds at the witness block of the token's
three gates (`Chain.witnesses`), the token-work bound holds for `{3, 4}` at the cap `1`, and gate `3`'s block is an
inference block. -/
def TokenCap_honest : Prop :=
  CutCost Chain.A 1 1 Chain.entry Chain.witnesses ∧ TokenWork Chain.A 1 1 Chain.entry Chain.witnesses {3, 4} 1 ∧
    InferenceBlock Chain.d 1 1 {3}

/-- **The token cap is false without A1**: at `Chain`, at the witness block of gates `2` and `4` (`Chain.skipping`),
the token-work bound holds for `{3, 4}` at the cap `0`, A1 fails at the block of gate `3`, and that block, an
inference block, holds one gate of `{3, 4}`, above the cap `0`. -/
def TokenCap_nontrivial : Prop :=
  ¬ CutCost Chain.A 1 1 Chain.entry Chain.skipping ∧ TokenWork Chain.A 1 1 Chain.entry Chain.skipping {3, 4} 0 ∧
    InferenceBlock Chain.d 1 1 {3} ∧ ¬ ((({3} : Finset (Fin Chain.d.D.N)) ∩ {3, 4}).card : ℝ≥0) ≤ 0

/-- **Slowdown's premises hold** at `Chain`, A1 and the token-work bound among them, at `(c, k) = (1, 1)`: its
annotation is valid, A1 holds at the witness block of the token's three gates (`Chain.witnesses`), the token-work bound
holds for `{3, 4}` at the cap `1`, the new weight has the cap `1`, `1` is a partition-level bound, and `blocks` is a
partition into inference blocks. -/
def Slowdown_honest : Prop :=
  Chain.A.Valid ∧ CutCost Chain.A 1 1 Chain.entry Chain.witnesses ∧
    TokenWork Chain.A 1 1 Chain.entry Chain.witnesses {3, 4} 1 ∧
    Caps Chain.d 1 1 (fun _ : Unit => {1}) (fun _ => 1) ∧ PartitionBound Chain.d 1 1 1 ∧
    Chain.blocks.IsCK Chain.d 1 1

/-- **Slowdown is false without A1**: at `Chain`, at the witness block of gates `2` and `4` (`Chain.skipping`) and the
rest as in `Slowdown_honest`, every other premise holds, the token-work bound for `{3, 4}` at the cap `0` among them,
A1 fails at the block of gate `3`, and the bound `⊤` is above `blocks`' price `1`. -/
def Slowdown_nontrivial : Prop :=
  Chain.A.Valid ∧ ¬ CutCost Chain.A 1 1 Chain.entry Chain.skipping ∧
    TokenWork Chain.A 1 1 Chain.entry Chain.skipping {3, 4} 0 ∧
    Caps Chain.d 1 1 (fun _ : Unit => {1}) (fun _ => 1) ∧ PartitionBound Chain.d 1 1 1 ∧
    Chain.blocks.IsCK Chain.d 1 1 ∧
    ¬ max (kindBound Chain.d 1 {3, 4} 0) (max (⨆ _K : Unit, kindBound Chain.d 1 {1} 1) (⨆ _J : Unit, (1 : ℝ≥0∞))) ≤
      partitionSlowdown Chain.d 4 1

/-- **Security's premises hold** at `Chain`, A1, the token-work bound and A2 among them: its run the one method, with
the decoder `silent`, `D = ⊤`, the loss `0` and `ℓ* = 0`, `p = 1`, its four blocks in one period of four, and
Slowdown's premises as in `Slowdown_honest`. -/
def Security_honest : Prop :=
  EfficiencyFloor Chain.methods silent 1 (fun _ => 0) 0 ⊤ 1 ∧ Chain.r.Among Chain.methods ∧ Chain.r.A.Valid ∧
    Chain.r.Within silent 1 ⊤ ∧ Chain.r.Reaches (fun _ => 0) 0 ∧ CutCost Chain.r.A 1 1 Chain.entry Chain.witnesses ∧
    TokenWork Chain.r.A 1 1 Chain.entry Chain.witnesses {3, 4} 1 ∧
    Caps Chain.r.d 1 1 (fun _ : Unit => {1}) (fun _ => 1) ∧ PartitionBound Chain.r.d 1 1 1 ∧
    Chain.blocks.IsCK Chain.r.d 1 1 ∧ InPeriods 4 1 4

/-- **Security is false without A1**: at `Chain`, at the witness block of gates `2` and `4` and the rest as in
`Security_honest`, every other premise holds, the token-work bound and A2 among them, A1 fails at the block of gate
`3`, and `ρ̲ p H(ℓ*) = ⊤` is above `N n c = 4`. -/
def Security_nontrivial : Prop :=
  EfficiencyFloor Chain.methods silent 1 (fun _ => 0) 0 ⊤ 1 ∧ Chain.r.Among Chain.methods ∧ Chain.r.A.Valid ∧
    Chain.r.Within silent 1 ⊤ ∧ Chain.r.Reaches (fun _ => 0) 0 ∧ ¬ CutCost Chain.r.A 1 1 Chain.entry Chain.skipping ∧
    TokenWork Chain.r.A 1 1 Chain.entry Chain.skipping {3, 4} 0 ∧
    Caps Chain.r.d 1 1 (fun _ : Unit => {1}) (fun _ => 1) ∧ PartitionBound Chain.r.d 1 1 1 ∧
    Chain.blocks.IsCK Chain.r.d 1 1 ∧ InPeriods 4 1 4 ∧
    ¬ max (kindBound Chain.r.d 1 {3, 4} 0) (max (⨆ _K : Unit, kindBound Chain.r.d 1 {1} 1) (⨆ _J : Unit, (1 : ℝ≥0∞))) *
      ENNReal.ofReal 1 * honestCost Chain.methods (fun _ => 0) 0 ≤ ((1 * 4 * 1 : ℕ) : ℝ≥0∞)

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
fit at `k = 2`, with `Δ = Δ_t = Δ_grad = s = τ = 1` and no decoder. -/
def FitTraining_honest : Prop :=
  Honest.d.D.Boolean ∧ TrainingBlocks Honest.d Honest.P 1 0 1 1 1 1 1 ∧ TrainingFits 1 1 1 1 1 2

/-- **Fit is false for training blocks that don't fit**: at `Honest`, at `k = 1` the blocks don't fit, no partition into
inference blocks exists, and `ρ = ⊤`, above `c / (c - d) = 1`. -/
def FitTraining_nontrivial : Prop :=
  Honest.d.D.Boolean ∧ TrainingBlocks Honest.d Honest.P 1 0 1 1 1 1 1 ∧ ¬ TrainingFits 1 1 1 1 1 1 ∧
    ¬ slowdown Honest.d 1 1 ≤ ((1 : ℕ) : ℝ≥0∞) / ((1 - 0 : ℕ) : ℝ≥0∞)

end Slowdown.Formalism.Guarantees
