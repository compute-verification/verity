import Definitions.Nci.Guarantees.Capability.Budgets
import Definitions.Nci.Guarantees.Capability.Incidence
import Definitions.Nci.Guarantees.Capability.Dependency
import Definitions.Nci.Capability.Toy

/-!
# Local bounds: their witnesses

Each local bound has two witnesses at the toy instance (`Nci.Capability.Toy`), listed under the lock's `witnesses`:

* `XHonest`: the bound's premises, every one, hold together at the toy, at the parameters its conclusion is read at;
* `XNontrivial`: its conclusion fails at the toy where a premise is dropped, so the conclusion is not a tautology.

The structural premises (`HeldPriceAtMost`, `SlotsRead`, the incidence, `Consumers.Upstream`, `OverlapWeights`,
`RequiresState`) are proved at the toy. So are the information premises, at the toy's decoder `M`, which reads only
programs of two bits: every description length it gives is `2`, so description and source hardness hold at deficit `0`
for prices summing to at most two bits; min-entropy hardness holds at deficit `0` for the product's value, a uniform
bit. The operand caps share one nontrivial witness, since their conclusions are the same.
-/

namespace Nci.Guarantees

open Verity Finset Nci.Capability Nci.Capability.Toy

/-- **`HeldBudget`'s premises hold together** at the toy, with `c = 2`, `k = 1`, `aBlk = 1`, `Λ = 0` and the product's
value priced one bit: every held selection has an evaluator of one bit (its tag), and description hardness holds. -/
def HeldBudgetHonest : Prop :=
  R1.P.IsCK R1.d 2 1 ∧
    (∀ (U : Fin R1.B → Finset (Fin 1)) (σ : Fin 1 → Finset (Fin R1.d.D.N)),
      (∀ u, Selects (Y 1) (U u) σ ∧ ∀ i ∈ U u, σ i ⊆ holds R1.d.D (R1.P.gates u)) →
      ∃ e : List Bool, e.length ≤ 1 ∧ Evaluates M () 0 R1 e fun u T => selectionBits T (U u) σ) ∧
    Assumptions.DescriptionHard R1 (Y 1) M () 0 0

/-- **`HeldBudget`'s conclusion can fail**: with the product's value priced three bits, above every description of
it, the proof block holds more than `k + ⌈log₂ B⌉ + aBlk + Λ = 2`. -/
def HeldBudgetNontrivial : Prop :=
  ¬ HeldPriceAtMost R1 (Y 3) (((1 + Nat.clog 2 R1.B + 1 : ℕ) : ℝ) + 0)

/-- **`StatHeldBudget`'s premises hold together** at the toy, with `c = 2`, `k = 1`, `Λ = 0`, the token drawn
uniformly and the product's value priced one bit. -/
def StatHeldBudgetHonest : Prop :=
  R1.P.IsCK R1.d 2 1 ∧ (∀ a ∈ μ.support, R1.Honest a) ∧ Assumptions.MinEntropyHard R1 (Y 1) μ 0

/-- **`StatHeldBudget`'s conclusion can fail**: with the product's value priced three bits, more than its one bit of
entropy, the proof block holds more than `k + Λ = 1`. -/
def StatHeldBudgetNontrivial : Prop :=
  ¬ HeldPriceAtMost R1 (Y 3) (((1 : ℕ) : ℝ) + 0)

/-- **`RecoveredBudget`'s premises hold together** at the toy, with `c = 2`, `k = 1`, the evaluator `[true]`
(`aBlk = 1`), the token's bit as the source (`ℓz = 1`, `δ = 0`) and one eligible output per block recovering it
(`βv = 1`, `amb = 0`). -/
def RecoveredBudgetHonest : Prop :=
  R1.P.IsCK R1.d 2 1 ∧ ([true] : List Bool).length ≤ 1 ∧ Assumptions.SourceHard R1 M () 0 z 1 0 ∧
    Assumptions.Recovers R1 M () 0 [true] z 1 1 0 fun _ => 1

/-- **`RecoveredBudget`'s conclusion can fail**: at three bits per eligible output, more than the source holds,
`βv · J = 3` exceeds `k + ⌈log₂ B⌉ + aBlk + δ + amb = 2`. -/
def RecoveredBudgetNontrivial : Prop :=
  ¬ ((3 : ℝ) * ((1 : ℕ) : ℝ) ≤ ((1 + Nat.clog 2 R1.B + 1 : ℕ) : ℝ) + 0 + 0)

/-- **`IncidenceCap`'s premises hold together** at the toy's product: it reads its slot and its operand, with at most
one product per slot-operand pair (`r = 1`) and per operand (`aout = 1`). -/
def IncidenceCapHonest : Prop :=
  SlotsRead B W ∧ (∀ q ∈ B.P, ∃ c ∈ O.cells (I.occ q), c ⊆ holds D.D (B.m q)) ∧
    (∀ i o, ((B.P.filter fun q => q.1 = i ∧ I.occ q = o).card : ℝ) ≤ 1) ∧
    ∀ o, ((B.P.filter fun q => I.occ q = o).card : ℝ) ≤ 1

/-- **`IncidenceCap`'s conclusion can fail**: at `r = aout = 0`, below the one product per pair, the gate computing
the product completes more than `min(r |U| |V|, aout |V|) = 0` products. -/
def IncidenceCapNontrivial : Prop :=
  ¬ (((completedIn B ({2} : Finset (Fin 4))).card : ℝ) ≤
      min (0 * ((completedIn B ({2} : Finset (Fin 4))).image Prod.fst).card *
          ((completedIn B ({2} : Finset (Fin 4))).image I.occ).card)
        (0 * ((completedIn B ({2} : Finset (Fin 4))).image I.occ).card))

/-- **`OperandCapSeparate`'s premises hold together** at the toy: one bit per slot and per operand, the block holding
slots and operands priced at most one bit each, and `Q = 1`. -/
def OperandCapSeparateHonest : Prop :=
  (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (∀ i, W.price i = 1) ∧ (∀ o, O.price o = 1) ∧
    (∀ i o, ((B.P.filter fun q => q.1 = i ∧ I.occ q = o).card : ℝ) ≤ 1) ∧
    (∀ o, ((B.P.filter fun q => I.occ q = o).card : ℝ) ≤ 1) ∧
    HeldPriceAtMost R1 W 1 ∧ HeldPriceAtMost R1 O 1 ∧
    ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → 1 * x ≤ 1 → 1 * y ≤ 1 → min (1 * x * y) (1 * y) ≤ 1

/-- **`OperandCapJoint`'s premises hold together** at the toy: one bit per slot and per operand, the block holding
slots and operands priced at most two bits together, and `Q = 2`. -/
def OperandCapJointHonest : Prop :=
  (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (∀ i, W.price i = 1) ∧ (∀ o, O.price o = 1) ∧
    (∀ i o, ((B.P.filter fun q => q.1 = i ∧ I.occ q = o).card : ℝ) ≤ 1) ∧
    (∀ o, ((B.P.filter fun q => I.occ q = o).card : ℝ) ≤ 1) ∧
    HeldPriceAtMost R1 (W.sum O) 2 ∧
    ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → 1 * x + 1 * y ≤ 2 → min (1 * x * y) (1 * y) ≤ 2

/-- **The operand caps' conclusion can fail**: the proof block completes more than `Q = 0` products, a `Q` no budget
on what it holds bounds. -/
def OperandCapNontrivial : Prop :=
  ¬ (((completedIn B (R1.P.gates 0)).card : ℝ) ≤ 0)

/-- **`ImportOrRecompute`'s premises hold together** at the toy: the token reaches the gate the product reads, and the
one consumer has weight `1`. -/
def ImportOrRecomputeHonest : Prop :=
  K.Upstream ∧ K.OverlapWeights fun _ => 1

/-- **`ImportOrRecompute`'s conclusion can fail**: at weight `2`, over what one gate's regions allow, the proof block
finishes more weight than its one unregistered input bit and the consumers it recomputes. -/
def ImportOrRecomputeNontrivial : Prop :=
  ¬ (∑ _c ∈ K.servedIn (R1.P.gates 0), (2 : ℝ) ≤
      ((R1.d.unregistered (R1.P.gates 0)).card : ℝ) +
        ∑ _c ∈ (K.servedIn (R1.P.gates 0)).filter (fun c => K.G c ⊆ R1.P.gates 0), (2 : ℝ))

/-- **`RecomputeByGates`'s premises hold together** at the toy: the consumer has weight `1` and one required gate
(`g = 1`). -/
def RecomputeByGatesHonest : Prop :=
  (K.OverlapWeights fun _ => 1) ∧ ∀ c, (1 : ℝ) ≤ (K.G c).card

/-- **`RecomputeByGates`'s conclusion can fail**: at `g = 2`, above the consumer's one required gate, the gates
`S = {1}` contain more recomputed weight than `|S| / g`. -/
def RecomputeByGatesNontrivial : Prop :=
  ¬ ((2 : ℝ) * ∑ _c ∈ (univ : Finset (Fin 1)).filter (fun c => K.G c ⊆ ({1} : Finset (Fin 4))), (1 : ℝ) ≤
      (({1} : Finset (Fin 4)).card : ℝ))

/-- **`RecomputeByState`'s premises hold together** at the toy with two proof blocks: each holds one computed gate's
value, priced one bit (`h = 1`), and the consumer requiring both gates holds two. -/
def RecomputeByStateHonest : Prop :=
  HeldPriceAtMost R2 W2 1 ∧ K2.RequiresState W2 1

/-- **`RecomputeByState`'s conclusion can fail**: with one proof block holding both gates, that block contains the
consumer's required gates. -/
def RecomputeByStateNontrivial : Prop :=
  ¬ ∀ (u : Fin R1.B) (c : Fin 1), ¬ K2.G c ⊆ R1.P.gates u

/-- **`WeightedProductCap`'s premises hold together** at the toy: the product reads its slot, the use has weight `1`,
and the block holds slots priced at most one bit. -/
def WeightedProductCapHonest : Prop :=
  SlotsRead B W ∧ (∀ _j : Fin B.nt, (0 : ℝ) ≤ 1) ∧ HeldPriceAtMost R1 W 1

/-- **`WeightedProductCap`'s conclusion can fail**: at `h = 0`, below the slot the block holds, its weighted products
exceed `h` times the weight it serves. -/
def WeightedProductCapNontrivial : Prop :=
  ¬ (∑ q ∈ completedIn B (R1.P.gates 0), W.price q.1 * 1 ≤ 0 * ∑ _j ∈ served B (R1.P.gates 0), (1 : ℝ))

end Nci.Guarantees
