import Definitions.Nci.Assumptions.Capability
import Definitions.Core.Guarantees.Formalism

/-!
# Local bounds: the repertoire

The statements a local bound is assembled from, in the order an argument uses them. Write `k' = k + ⌈log₂ B⌉ + a_blk`,
Lemma 1's bound (`Verity.Guarantees.BlockDescription`).

1. **Budgets, from information.** `HeldBudget`: Lemma 1 with description hardness, every proof block holds a selection
   priced at most `k' + Λ`. `RecoveredBudget`: with a recoverable hard source, a block's eligible outputs number at most
   `(k' + δ + amb)/βv`.
2. **Caps, from structure.**
   * Incidence: `IncidenceCap`, at most `min(r |U| |V|, a_out |V|)` products from `|U|` slots and `|V|` operands;
     `OperandCapSeparate` and `OperandCapJoint` bound them over the budgets' feasible region.
   * Dependency: `ImportOrRecompute`, each served use either spends an imported gate of its region or contains its
     required gates; `RecomputeByGates` caps the second route by the block's gates, `RecomputeByState` rules it out when
     recomputing needs more updated state than the budget; `WeightedProductCap` turns served weight into a cap on
     price-weighted products.
3. **Counting.** Theorem 3 (`Verity.Guarantees.Counting`, `Verity.Guarantees.CountingSum`) turns a per-block cap on
   weighted products into `|C| ≥ c Ω / q`.
4. **Coverage.** `CapabilityFloor` (`Definitions.Accounting.Guarantees.Coverage`): below the resulting floor, success
   is rare.

Improving the structure (a finer incidence graph, better weights) never strengthens an information premise; a stronger
premise is a new hypothesis, named as one.
-/

namespace Nci.Guarantees

open Verity Finset Nci.Capability
open scoped ENNReal

/-- **The held budget** (Lemma 1 with description hardness). In a `(c, k)`-partition whose proof blocks a fixed
evaluator `e` of at most `aBlk` bits can extract within `t` (for each selection of `X`'s items held per block, the
values of their cells), description hardness at deficit `Λ` bounds what every proof block holds:
`Σ_{i ∈ U} price_i ≤ k + ⌈log₂ B⌉ + aBlk + Λ`, whether the block loads the values, decompresses them or computes them. -/
def HeldBudget : Prop :=
  ∀ (R : Registration) (c k : ℕ), R.P.IsCK R.d c k →
    ∀ {ι : Type} (X : Items R.d.D ι) (ρ : Type) (M : Decoder ρ) (r : ρ) (t aBlk : ℕ) (Λ : ℝ) (e : List Bool),
      e.length ≤ aBlk →
      (∀ (U : Fin R.B → Finset ι) (σ : ι → Finset (Fin R.d.D.N)),
        (∀ u, Selects X (U u) σ ∧ ∀ i ∈ U u, σ i ⊆ holds R.d.D (R.P.gates u)) →
        Evaluates M r t R e fun u T => R.d.D.bits T (((U u).biUnion σ).sort (· ≤ ·))) →
      Assumptions.DescriptionHard R X M r t Λ →
      HeldPriceAtMost R X ((k + Nat.clog 2 R.B + aBlk : ℕ) + Λ)

/-- **The recovered budget** (Lemma 1 with a recoverable hard source). In a `(c, k)`-partition with an evaluator `e` of at
most `aBlk` bits, source hardness at deficiency `δ` and recovery at `βv` bits per eligible output with `amb` bits of
ambiguity give, for every proof block, `βv · J u ≤ k + ⌈log₂ B⌉ + aBlk + δ + amb`. When the ambiguity grows with the
outputs, `amb ≤ ε · J u` with `ε < βv`, this is `(βv − ε) · J u ≤ k' + δ`: ambiguity lowers the information per output. -/
def RecoveredBudget : Prop :=
  ∀ (R : Registration) (c k : ℕ), R.P.IsCK R.d c k →
    ∀ (ρ : Type) (M : Decoder ρ) (r : ρ) (τ aBlk : ℕ) (e : List Bool) (z : R.d.D.Transcript → List Bool)
      (ℓz δ βv amb : ℝ) (J : Fin R.B → ℕ), e.length ≤ aBlk →
      Assumptions.SourceHard R M r τ z ℓz δ → Assumptions.Recovers R M r τ e z ℓz βv amb J →
      ∀ u, βv * J u ≤ (k + Nat.clog 2 R.B + aBlk : ℕ) + δ + amb

/-- **The incidence cap.** With at most `r` counted products per slot-operand pair and at most `aout` per operand, any
gates `S` compute at most `min(r |U| |V|, aout |V|)` counted products whole, `U` and `V` the slots and operands those
products use. The rectangle `U × V` overcounts pairs that never meet in a product; a finer count of the incidence graph is
a structural improvement, not a new premise. -/
def IncidenceCap : Prop :=
  ∀ (d : Declaration) (B : Nci.Covering.MulBlocks d) {κ : Type} [DecidableEq κ] (W : Items d.D (Fin B.nw))
    (O : Items d.D κ) (I : Incidence B W O) (r aout : ℝ),
    (∀ i o, ((B.P.filter fun q => q.1 = i ∧ I.occ q = o).card : ℝ) ≤ r) →
    (∀ o, ((B.P.filter fun q => I.occ q = o).card : ℝ) ≤ aout) →
    ∀ S : Finset (Fin d.D.N),
      ((completedIn B S).card : ℝ) ≤
        min (r * ((completedIn B S).image Prod.fst).card * ((completedIn B S).image I.occ).card)
          (aout * ((completedIn B S).image I.occ).card)

/-- **The operand cap, separate budgets.** One price `βw > 0` per slot and `βx > 0` per operand; every proof block holds
slots priced at most `hw` and operands priced at most `hx`. If `Q` bounds `min(r x y, aout y)` wherever `βw x ≤ hw` and
`βx y ≤ hx`, each proof block computes at most `Q` counted products whole. -/
def OperandCapSeparate : Prop :=
  ∀ (R : Registration) (B : Nci.Covering.MulBlocks R.d) {κ : Type} [DecidableEq κ] (W : Items R.d.D (Fin B.nw))
    (O : Items R.d.D κ) (I : Incidence B W O) (r aout βw βx hw hx Q : ℝ),
    0 < βw → 0 < βx → (∀ i, W.price i = βw) → (∀ o, O.price o = βx) →
    (∀ i o, ((B.P.filter fun q => q.1 = i ∧ I.occ q = o).card : ℝ) ≤ r) →
    (∀ o, ((B.P.filter fun q => I.occ q = o).card : ℝ) ≤ aout) →
    HeldPriceAtMost R W hw → HeldPriceAtMost R O hx →
    (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → βw * x ≤ hw → βx * y ≤ hx → min (r * x * y) (aout * y) ≤ Q) →
    ∀ u, ((completedIn B (R.P.gates u)).card : ℝ) ≤ Q

/-- **The operand cap, one joint budget.** As `OperandCapSeparate`, with every proof block holding slots and operands
priced at most `h` together (description hardness of `W.sum O`), and `Q` bounding `min(r x y, aout y)` wherever
`βw x + βx y ≤ h`. The bound's maximum is `r h² / (4 βw βx)` when `r h ≤ 2 βw aout` (the block can't obtain enough slots:
pairs limit it) and `aout (h − βw aout / r) / βx` otherwise (operand supply and fan-out limit it). -/
def OperandCapJoint : Prop :=
  ∀ (R : Registration) (B : Nci.Covering.MulBlocks R.d) {κ : Type} [DecidableEq κ] (W : Items R.d.D (Fin B.nw))
    (O : Items R.d.D κ) (I : Incidence B W O) (r aout βw βx h Q : ℝ),
    0 < βw → 0 < βx → (∀ i, W.price i = βw) → (∀ o, O.price o = βx) →
    (∀ i o, ((B.P.filter fun q => q.1 = i ∧ I.occ q = o).card : ℝ) ≤ r) →
    (∀ o, ((B.P.filter fun q => I.occ q = o).card : ℝ) ≤ aout) →
    HeldPriceAtMost R (W.sum O) h →
    (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → βw * x + βx * y ≤ h → min (r * x * y) (aout * y) ≤ Q) →
    ∀ u, ((completedIn B (R.P.gates u)).card : ℝ) ≤ Q

/-- **Import or recompute.** Under the upstream certificate and overlap weights `ω`, the weight of the uses a proof block
serves is at most its unregistered input bits plus the weight of the uses whose required gates it contains: each served
use either spends an imported gate of its region, a gate whose weight over all regions is at most 1, or computes all of
its required gates. -/
def ImportOrRecompute : Prop :=
  ∀ (R : Registration) (B : Nci.Covering.MulBlocks R.d) (Rg : UseRegions R.d B.nt) (ω : Fin B.nt → ℝ),
    Upstream B Rg → OverlapWeights Rg ω →
    ∀ u, ∑ j ∈ served B (R.P.gates u), ω j ≤
      (R.d.unregistered (R.P.gates u)).card +
        ∑ j ∈ (served B (R.P.gates u)).filter (fun j => Rg.G j ⊆ R.P.gates u), ω j

/-- **Recomputing costs gates.** With overlap weights `ω` and at least `g` required gates per use, the weight of the uses
whose required gates lie within any gates `S` is at most `|S| / g`: in a `(c, k)`-partition, `c / g` a block. -/
def RecomputeByGates : Prop :=
  ∀ (d : Declaration) (nt : ℕ) (Rg : UseRegions d nt) (ω : Fin nt → ℝ) (g : ℝ), OverlapWeights Rg ω →
    (∀ j, g ≤ (Rg.G j).card) → ∀ (J : Finset (Fin nt)) (S : Finset (Fin d.D.N)),
      g * ∑ j ∈ J.filter (fun j => Rg.G j ⊆ S), ω j ≤ S.card

/-- **Recomputing costs updated state.** If every proof block holds updated state priced at most `h` and recomputing any
use needs more, no proof block contains a use's required gates: every served use is imported. -/
def RecomputeByState : Prop :=
  ∀ (R : Registration) {ι : Type} (W : Items R.d.D ι) (nt : ℕ) (Rg : UseRegions R.d nt) (h : ℝ),
    HeldPriceAtMost R W h → RequiresState Rg W h → ∀ (u : Fin R.B) (j : Fin nt), ¬ Rg.G j ⊆ R.P.gates u

/-- **The weighted product cap.** Weight each counted product `(i, j)` by its slot's price times its use's weight. If every
proof block holds slots priced at most `h`, a proof block's weighted products are at most `h` times the weight of the uses
it serves. With `ImportOrRecompute` and `RecomputeByGates` that is `h (k + c/g)`; with `RecomputeByState`, `h k`. -/
def WeightedProductCap : Prop :=
  ∀ (R : Registration) (B : Nci.Covering.MulBlocks R.d) (W : Items R.d.D (Fin B.nw)) (ω : Fin B.nt → ℝ) (h : ℝ),
    SlotsRead B W → (∀ j, 0 ≤ ω j) → HeldPriceAtMost R W h →
    ∀ u, ∑ q ∈ completedIn B (R.P.gates u), W.price q.1 * ω q.2 ≤ h * ∑ j ∈ served B (R.P.gates u), ω j

/-- **The local bounds are stated**: each of them, equal to itself, so the lock records their text and every definition
they read until they are proved. -/
def LocalBoundsStated : Prop :=
  let s := HeldBudget ∧ RecoveredBudget ∧ IncidenceCap ∧ OperandCapSeparate ∧ OperandCapJoint ∧ ImportOrRecompute ∧
    RecomputeByGates ∧ RecomputeByState ∧ WeightedProductCap
  s = s

end Nci.Guarantees
