module

public import Definitions.Slowdown.Guarantees.Capability.Budgets
public import Definitions.Slowdown.Guarantees.Capability.Dependency
public import Definitions.Slowdown.Capability.Toy

/-! Each local bound's honest and nontrivial witnesses at the toy instance, listed under the lock's `witnesses`. -/

@[expose] public section

namespace Slowdown.Guarantees

open Verity Finset Slowdown.Capability Slowdown.Capability.Toy

/-- **`HeldBudget`'s premises hold together** at the toy, with `c = 2`, `k = 1`, `aBlk = 3`, `Λ = 1` and the value of
gates `2` and `3` priced two bits: every held selection has an evaluator of at most three bits, and description hardness
holds at deficit `1`. The deficit is spent: at `Λ = 0` description hardness fails, since `M` describes `11` in one
bit. A selection is written out, not as `Selects`: the lock's witnesses rule refuses an honest witness's hypothesis
headed by a definition outside an assumptions module. -/
def HeldBudgetHonest : Prop :=
  R1.d.𝓘.IsCK R1.d 2 1 ∧
    (∀ (U : Fin R1.d.n → Finset (Fin 1)) (σ : Fin 1 → Finset (Fin R1.d.D.N)),
      (∀ u, (∀ i ∈ U u, σ i ∈ (Y2 2).cells i) ∧ ∀ i ∈ U u, σ i ⊆ holds R1.d.D (R1.d.𝓘.gates u)) →
      ∃ e : List Bool, e.length ≤ 3 ∧ Evaluates M () 0 R1 e fun u T => selectionBits T (U u) σ) ∧
    Assumptions.DescriptionHard R1 (Y2 2) M () 0 1 ∧ ¬ Assumptions.DescriptionHard R1 (Y2 2) M () 0 0

/-- **`HeldBudget`'s conclusion fails without description hardness**: the partition and the three-bit evaluators hold
as in its honest witness; with the value of gates `2` and `3` priced six bits, description hardness fails at deficit
`1`, since `M` describes `11` in one bit; and the proof block holds more than `k + ⌈log₂ B⌉ + aBlk + Λ = 5`. -/
def HeldBudgetNontrivial : Prop :=
  R1.d.𝓘.IsCK R1.d 2 1 ∧
    (∀ (U : Fin R1.d.n → Finset (Fin 1)) (σ : Fin 1 → Finset (Fin R1.d.D.N)),
      (∀ u, (∀ i ∈ U u, σ i ∈ (Y2 6).cells i) ∧ ∀ i ∈ U u, σ i ⊆ holds R1.d.D (R1.d.𝓘.gates u)) →
      ∃ e : List Bool, e.length ≤ 3 ∧ Evaluates M () 0 R1 e fun u T => selectionBits T (U u) σ) ∧
    ¬ Assumptions.DescriptionHard R1 (Y2 6) M () 0 1 ∧
    ¬ HeldPriceAtMost R1 (Y2 6) (((1 + Nat.clog 2 R1.d.n + 3 : ℕ) : ℝ) + 1)

/-- **`StatHeldBudget`'s premises hold together** at the toy, with `c = 2`, `k = 1`, `Λ = 0`, the token sampled
uniformly and the product's value priced one bit. -/
def StatHeldBudgetHonest : Prop :=
  R1.d.𝓘.IsCK R1.d 2 1 ∧ (∀ a ∈ μ.support, R1.Honest a) ∧ Assumptions.MinEntropyHard R1 (Y 1) μ 0

/-- **`StatHeldBudget`'s conclusion fails without min-entropy hardness**: the partition and the honest support hold as
in its honest witness; with the product's value priced three bits, more than its one bit of entropy, min-entropy
hardness fails at deficit `0`; and the proof block holds more than `k + Λ = 1`. -/
def StatHeldBudgetNontrivial : Prop :=
  R1.d.𝓘.IsCK R1.d 2 1 ∧ (∀ a ∈ μ.support, R1.Honest a) ∧ ¬ Assumptions.MinEntropyHard R1 (Y 3) μ 0 ∧
    ¬ HeldPriceAtMost R1 (Y 3) (((1 : ℕ) : ℝ) + 0)

/-- **`RecoveredBudget`'s premises hold together** at the toy, with `c = 2`, `k = 1`, the evaluator `[1, 1, 1]`
(`aBlk = 3`), the value of gates `2` and `3` as the source (`ℓz = 2`, `δ = 1`) and one eligible output per block
recovering it with a one-bit complement (`βv = 2`, `amb = 1`). Both parameters are spent: at `δ = 0` source hardness
fails, since `M` describes `11` in one bit, and at `amb = 0` the recovery fails, since it needs its bit picking the
source. -/
def RecoveredBudgetHonest : Prop :=
  R1.d.𝓘.IsCK R1.d 2 1 ∧ ([true, true, true] : List Bool).length ≤ 3 ∧ Assumptions.SourceHard R1 M () 0 z 2 1 ∧
    Assumptions.Recovers R1 M () 0 [true, true, true] z 2 2 1 (fun _ => 1) ∧
    ¬ Assumptions.SourceHard R1 M () 0 z 2 0 ∧
    ¬ Assumptions.Recovers R1 M () 0 [true, true, true] z 2 2 0 fun _ => 1

/-- **`RecoveredBudget`'s conclusion fails without the recovery**: the partition, the three-bit evaluator and source
hardness hold as in its honest witness; at seven bits per eligible output, more than the source holds, the source is
not recovered; and `βv · J = 7` exceeds `k + ⌈log₂ B⌉ + aBlk + δ + amb = 6`. -/
def RecoveredBudgetNontrivial : Prop :=
  R1.d.𝓘.IsCK R1.d 2 1 ∧ ([true, true, true] : List Bool).length ≤ 3 ∧ Assumptions.SourceHard R1 M () 0 z 2 1 ∧
    ¬ Assumptions.Recovers R1 M () 0 [true, true, true] z 2 7 1 (fun _ => 1) ∧
    ¬ ((7 : ℝ) * ((1 : ℕ) : ℝ) ≤ ((1 + Nat.clog 2 R1.d.n + 3 : ℕ) : ℝ) + 1 + 1)

/-- **`ImportOrRecompute`'s premises hold together** at the toy: the token reaches the gate the product reads, and the
one consumer has weight `1`. -/
def ImportOrRecomputeHonest : Prop :=
  K.Upstream ∧ K.OverlapWeights fun _ => 1

/-- **`ImportOrRecompute`'s conclusion fails without overlap weights**: the token reaches the gate the product reads;
the weight `2` is over what one gate's regions allow; and the proof block finishes more weight than its one
unregistered input bit and the consumers it recomputes. -/
def ImportOrRecomputeNontrivial : Prop :=
  K.Upstream ∧ ¬ (K.OverlapWeights fun _ => 2) ∧
    ¬ (∑ _c ∈ K.servedIn (R1.d.𝓘.gates 0), (2 : ℝ) ≤
        ((R1.d.unregistered (R1.d.𝓘.gates 0)).card : ℝ) +
          ∑ _c ∈ (K.servedIn (R1.d.𝓘.gates 0)).filter (fun c => K.G c ⊆ R1.d.𝓘.gates 0), (2 : ℝ))

/-- **`RecomputeByGates`'s premises hold together** at the toy: the consumer has weight `1` and one required gate
(`g = 1`). -/
def RecomputeByGatesHonest : Prop :=
  (K.OverlapWeights fun _ => 1) ∧ ∀ c, (1 : ℝ) ≤ (K.G c).card

/-- **`RecomputeByGates`'s conclusion fails without its gate count**: the consumer has weight `1`; `g = 2` is above
its one required gate; and the gates `S = {1}` contain more recomputed weight than `|S| / g`. -/
def RecomputeByGatesNontrivial : Prop :=
  (K.OverlapWeights fun _ => 1) ∧ ¬ (∀ c, (2 : ℝ) ≤ (K.G c).card) ∧
    ¬ ((2 : ℝ) * ∑ _c ∈ (univ : Finset (Fin 1)).filter (fun c => K.G c ⊆ ({1} : Finset (Fin 4))), (1 : ℝ) ≤
        (({1} : Finset (Fin 4)).card : ℝ))

/-- **`RecomputeByState`'s premises hold together** at the toy with two proof blocks: each holds one computed gate's
value, priced one bit (`h = 1`), and the consumer requiring both gates holds two. -/
def RecomputeByStateHonest : Prop :=
  HeldPriceAtMost R2 W2 1 ∧ K2.RequiresState W2 1

/-- **`RecomputeByState`'s conclusion fails without required state**: the one proof block holding both gates holds
their values priced at most `h = 2`; recomputing the consumer needs no state above `h`; and the block contains the
consumer's required gates. -/
def RecomputeByStateNontrivial : Prop :=
  HeldPriceAtMost R1 W2 2 ∧ ¬ K2.RequiresState W2 2 ∧ ¬ ∀ (u : Fin R1.d.n) (c : Fin 1), ¬ K2.G c ⊆ R1.d.𝓘.gates u

end Slowdown.Guarantees
