module

public import Definitions.Nci.Guarantees.Formalism
public import Definitions.Nci.Formalism.Instances

/-!
# The training formalism v2: honest and nontrivial lemmas

The counting lemma, the token cap, Slowdown, Security and Fit's two bounds each have two statements here, at the
instances of `Security/Definitions/Nci/Formalism/Instances.lean`. `<Guarantee>_honest`: its premises hold together, the named
assumptions among them. `<Guarantee>_nontrivial`: every premise but one holds, that one fails, and the bound fails, its
last conjunct the bound's negation.

The counting lemma, the token cap, Slowdown and Security are at `Chain`, at `(c, k) = (1, 1)`: a token of three gates in a chain, of
which only the first is computed from the token's inputs. Each premise holds at `blocks`, four blocks of one gate,
priced at `1`. A1 holds at `Chain.witnesses`, the token's three gates as one witness block, which reads one shared bit,
the new weight, and none of the token's state: every block that works on the token reads one bit. The token kind's
work is `X = {3, 4}`, the token's gates no block computing it from its inputs holds, with the cap `1`, Slowdown's other
kind is the new weight, gate `1`, with the cap `1`, and the partition-level kind is all of `C`, whose bound `1` every
partition into blocks of one gate meets with its `|C| = 4` blocks, so the bound is `max(1/2, 1/4, 1) = 1`.

* The counting lemma drops the partition into inference blocks: at the kind `{3, 4}` with the cap `1` and the
  partition-level bound `1`, its bound `1` is above `whole`'s price `1/4`, and `whole`'s one block of four gates is no
  inference block at `c = 1`.
* The token cap, Slowdown and Security drop A1 (`CutCost`): at `Chain.skipping`, the witness block of gates `2` and
  `4`, which reads one shared bit, the new weight, and one bit of the token's state, gate `3`, A1 prices every block
  that works on the token at two bits, which the block of gate `3` does not read. A1's budget then admits no block that
  holds a gate of `X`, and the token-work bound holds at the cap `0`. The block of gate `3` holds one gate of `X`, above
  that cap, and Slowdown's bound is `⊤`, above `blocks`' price `1` and, for Security, above `N n c = 4`.
* Fit's bounds are at `Honest`: its two blocks fit at `k = 2`, and at `k = 1` they don't and no partition into inference
  blocks exists, since gate `3` reads two values, so `ρ = ⊤`, above `c / (c - d) = 1`.

`FitHonestServing` concludes an equation, not a bound, and `TokenCapClean` concludes the token-work bound, a named
assumption: neither has such lemmas.

`Chain` and `Honest` are small faithful instances of a training step and of serving, accepted as the honest instances of
these structural bounds: they rest on no run and no `art:`, and a training run's annotation and checked witness blocks,
which no check reads yet, would supply the deployed premises.

The proofs are in `Security/Proofs/Nci/Formalism/Witnesses.lean`.
-/

@[expose] public section

namespace Nci.Formalism.Guarantees

open Nci.Formalism Nci.Formalism.Assumptions
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

end Nci.Formalism.Guarantees
