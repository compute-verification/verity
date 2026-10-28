import Definitions.Nci.Assumptions.Capability
import Definitions.Core.Guarantees.Formalism

/-!
# The capability argument: what each local set gives

Each local assumption set (`Definitions.Nci.Assumptions.Capability`) caps the counted products a proof block of a
`(c, k)`-partition computes whole, `|𝒥_I| ≤ q`; Theorem 3 (`Verity.Guarantees.Counting`) then gives `|C| ≥ c |𝒥| / q`.
Write `k' = k + ⌈log₂ B⌉ + a_blk`, Lemma 1's bound.

* From Lemma 1 and a description lower bound, what a block holds: `UpdatedStatePerBlock` (slots priced at most
  `k' + Λw`), `HeldUsesPerBlock` (at most `(k' + Λx)/βx` operands), `JointOperandsPerBlock` (both within one budget),
  `RecoverableCap` (Set 4: `βv |𝒥_I| ≤ k' + δ`).
* The counting each set adds, one price `βw` per slot:
  * Set 1, `UpstreamCap`: with the upstream certificate at demand `dm` and `βw · dm > k' + Λw`, each use a block serves
    costs it a handed gate of that use's own upstream computation, so `|𝒥_I| ≤ k (k' + Λw)/βw`;
  * Set 2, `OperandPairsCap`: `|𝒥_I| ≤ (k' + Λw)(k' + Λx)/(βw βx)`, and with one joint budget `JointOperandsCap`:
    `|𝒥_I| ≤ (k' + Λ)²/(4 βw βx)`;
  * Set 3, `FanOutCap`: `|𝒥_I| ≤ a_out (k' + Λx)/βx`.
* Coverage (Set 5), `CapabilityFloor`: with a floor `L` on covered computations (the smallest over the sets whose
  premises hold for the same blocks), reaching the target below `min(L, Tmax)` has probability at most `η`.
-/

namespace Nci.Guarantees

open Verity Finset Nci.Capability
open scoped ENNReal

/-- **A block holds little updated state** (Lemma 1 with local update hardness). In a `(c, k)`-partition whose proof
blocks a fixed evaluator `e` of at most `aBlk` bits can extract within `t` (for each choice of held slots, the tuple of
their versions), local update hardness at deficit `Λw` bounds the updated state every proof block holds:
`Σ_{i ∈ U} p_i ≤ k + ⌈log₂ B⌉ + aBlk + Λw`. -/
def UpdatedStatePerBlock : Prop :=
  ∀ (R : Registration) (c k : ℕ), R.P.IsCK R.d c k →
    ∀ (B : Nci.Covering.MulBlocks R.d) (W : UpdatedState B) (ρ : Type) (M : Decoder ρ) (r : ρ) (t aBlk : ℕ)
      (Λw : ℝ) (e : List Bool), e.length ≤ aBlk →
      (∀ (U : Fin R.B → Finset (Fin B.nw)) (σ : Fin B.nw → Fin R.d.D.N),
        (∀ u, ∀ i ∈ U u, σ i ∈ W.ver i ∧ σ i ∈ R.P.gates u ∪ R.d.D.into (R.P.gates u)) →
        Evaluates M r t R e fun u X => R.d.D.bits X (((U u).image σ).sort (· ≤ ·))) →
      Assumptions.LocalUpdateHardness R B W M r t Λw →
      HeldStateAtMost R W ((k + Nat.clog 2 R.B + aBlk : ℕ) + Λw)

/-- **A block holds few operands** (Lemma 1 with activation hardness at rate `βx > 0`): in a `(c, k)`-partition whose
proof blocks `e` can extract within `t` (for each choice of held uses, their operands' values), every proof block holds
at most `(k + ⌈log₂ B⌉ + aBlk + Λx)/βx` operands. -/
def HeldUsesPerBlock : Prop :=
  ∀ (R : Registration) (c k : ℕ), R.P.IsCK R.d c k →
    ∀ (B : Nci.Covering.MulBlocks R.d) (O : Operands B) (ρ : Type) (M : Decoder ρ) (r : ρ) (t aBlk : ℕ)
      (βx Λx : ℝ) (e : List Bool), 0 < βx → e.length ≤ aBlk →
      (∀ V : Fin R.B → Finset (Fin B.nt), (∀ u, ∀ j ∈ V u, O.act j ⊆ R.P.gates u ∪ R.d.D.into (R.P.gates u)) →
        Evaluates M r t R e fun u X => R.d.D.bits X (((V u).biUnion O.act).sort (· ≤ ·))) →
      Assumptions.ActivationHardness R O M r t βx Λx →
      HeldUsesAtMost R O (((k + Nat.clog 2 R.B + aBlk : ℕ) + Λx) / βx)

/-- **A block holds little of both families together** (Lemma 1 with joint operand hardness): every proof block's held
slots `U` and operands `V` satisfy `Σ_{i ∈ U} p_i + βx |V| ≤ k + ⌈log₂ B⌉ + aBlk + Λ`. -/
def JointOperandsPerBlock : Prop :=
  ∀ (R : Registration) (c k : ℕ), R.P.IsCK R.d c k →
    ∀ (B : Nci.Covering.MulBlocks R.d) (W : UpdatedState B) (O : Operands B) (ρ : Type) (M : Decoder ρ) (r : ρ)
      (t aBlk : ℕ) (βx Λ : ℝ) (e : List Bool), e.length ≤ aBlk →
      (∀ (U : Fin R.B → Finset (Fin B.nw)) (σ : Fin B.nw → Fin R.d.D.N) (V : Fin R.B → Finset (Fin B.nt)),
        (∀ u, ∀ i ∈ U u, σ i ∈ W.ver i ∧ σ i ∈ R.P.gates u ∪ R.d.D.into (R.P.gates u)) →
        (∀ u, ∀ j ∈ V u, O.act j ⊆ R.P.gates u ∪ R.d.D.into (R.P.gates u)) →
        Evaluates M r t R e fun u X => R.d.D.bits X (((U u).image σ ∪ (V u).biUnion O.act).sort (· ≤ ·))) →
      Assumptions.JointOperandHardness R W O M r t βx Λ →
      ∀ (u : Fin R.B) (U : Finset (Fin B.nw)) (σ : Fin B.nw → Fin R.d.D.N) (V : Finset (Fin B.nt)),
        (∀ i ∈ U, σ i ∈ W.ver i ∧ σ i ∈ R.P.gates u ∪ R.d.D.into (R.P.gates u)) →
        (∀ j ∈ V, O.act j ⊆ R.P.gates u ∪ R.d.D.into (R.P.gates u)) →
        ∑ i ∈ U, W.p i + βx * V.card ≤ (k + Nat.clog 2 R.B + aBlk : ℕ) + Λ

/-- **Set 1: updated state and upstream dependencies.** One price `βw > 0` per slot; every proof block holds updated
state priced at most `kb` (`UpdatedStatePerBlock`, `kb = k' + Λw`); the upstream certificate holds at demand `dm` with
`βw · dm > kb`. Then a block that computes a counted product of use `j` whole is handed a gate of `j`'s upstream
computation (else it contains all of `j`'s upstream products, over `dm` slots), these gates are distinct across uses
and unregistered, and so each proof block of a `(c, k)`-partition computes at most `k · kb / βw` counted products whole.
No operand is priced. -/
def UpstreamCap : Prop :=
  ∀ (R : Registration) (c k : ℕ), R.P.IsCK R.d c k →
    ∀ (B : Nci.Covering.MulBlocks R.d) (W : UpdatedState B) (βw kb : ℝ) (dm : ℕ), 0 < βw → (∀ i, W.p i = βw) →
      HeldStateAtMost R W kb → UpstreamCert B dm → kb < βw * dm →
      ∀ u, ((completed R B u).card : ℝ) ≤ k * (kb / βw)

/-- **Set 2: two operand families, separate bounds.** One price `βw > 0` per slot; every proof block holds updated
state priced at most `kbw` and at most `nx` operands. Each counted product is one slot-use pair, so each proof block
computes at most `(kbw / βw) · nx` counted products whole. -/
def OperandPairsCap : Prop :=
  ∀ (R : Registration) (B : Nci.Covering.MulBlocks R.d) (W : UpdatedState B) (O : Operands B) (βw kbw nx : ℝ),
    0 < βw → (∀ i, W.p i = βw) → HeldStateAtMost R W kbw → HeldUsesAtMost R O nx →
    ∀ u, ((completed R B u).card : ℝ) ≤ kbw / βw * nx

/-- **Set 2: two operand families, one joint budget.** One price `βw > 0` per slot and `βx > 0` per operand; every
proof block's held slots `U` and operands `V` satisfy `βw |U| + βx |V| ≤ kb` (`JointOperandsPerBlock`). Then each proof
block computes at most `kb² / (4 βw βx)` counted products whole. -/
def JointOperandsCap : Prop :=
  ∀ (R : Registration) (B : Nci.Covering.MulBlocks R.d) (W : UpdatedState B) (O : Operands B) (βw βx kb : ℝ),
    0 < βw → 0 < βx → (∀ i, W.p i = βw) →
    (∀ (u : Fin R.B) (U : Finset (Fin B.nw)) (σ : Fin B.nw → Fin R.d.D.N) (V : Finset (Fin B.nt)),
      (∀ i ∈ U, σ i ∈ W.ver i ∧ σ i ∈ R.P.gates u ∪ R.d.D.into (R.P.gates u)) →
      (∀ j ∈ V, O.act j ⊆ R.P.gates u ∪ R.d.D.into (R.P.gates u)) →
      βw * U.card + βx * V.card ≤ kb) →
    ∀ u, ((completed R B u).card : ℝ) ≤ kb ^ 2 / (4 * βw * βx)

/-- **Set 3: hard operands and bounded fan-out.** Every proof block holds at most `nx` operands, and each use has at
most `aout` counted products. Then each proof block computes at most `aout · nx` counted products whole. No slot is
priced. -/
def FanOutCap : Prop :=
  ∀ (R : Registration) (B : Nci.Covering.MulBlocks R.d) (O : Operands B) (aout nx : ℝ),
    (∀ j, ((B.P.filter fun q => q.2 = j).card : ℝ) ≤ aout) → HeldUsesAtMost R O nx →
    ∀ u, ((completed R B u).card : ℝ) ≤ aout * nx

/-- **Set 4: a hard source recoverable from the block's outputs.** In a `(c, k)`-partition, with an evaluator `e` of at
most `aBlk` bits, source hardness at deficiency `δ` and recovery at `βv` bits per eligible output give, for every proof
block, `βv · J u ≤ k + ⌈log₂ B⌉ + aBlk + δ`. -/
def RecoverableCap : Prop :=
  ∀ (R : Registration) (c k : ℕ), R.P.IsCK R.d c k →
    ∀ (ρ : Type) (M : Decoder ρ) (r : ρ) (τ aBlk : ℕ) (e : List Bool) (z : R.d.D.Transcript → List Bool)
      (ℓz δ βv : ℝ) (J : Fin R.B → ℕ), e.length ≤ aBlk →
      Assumptions.SourceHardness R M r τ z ℓz δ → Assumptions.Recovers R M r τ e z ℓz βv J →
      ∀ u, βv * J u ≤ (k + Nat.clog 2 R.B + aBlk : ℕ) + δ

/-- **Set 5: covered success below the floor is rare.** Under coverage of efficient methods at cost range `Tmax` and
error `η`, if every covered computation in the class `𝒜` costs at least `Lc` (the smallest floor over the local sets
whose premises hold, each from its cap and Theorem 3), then for every computation the probability of lying in `𝒜`,
reaching the target, and costing below `min Lc Tmax` is at most `η`. -/
def CapabilityFloor : Prop :=
  ∀ {In L : Type} {m : ℕ} (G : Accounting.Overhead.Task In (Fin m → List Bool)) (𝒜 : Accounting.Overhead.Class In L)
    (Core : (In → Accounting.Actual.Run L) → In → Prop) (cost : Accounting.Actual.Run L → ℕ) (Tmax Lc : ℕ)
    (η : ℝ≥0∞),
    Assumptions.CoversEfficientMethods G 𝒜 Core cost Tmax η →
    (∀ C x, (C, x) ∈ 𝒜 → Core C x → Lc ≤ cost (C x)) →
    ∀ (C : In → Accounting.Actual.Run L) (pos : Fin m → L × ℕ × ℕ),
      G.prob (fun x 𝒴 => (C, x) ∈ 𝒜 ∧ Accounting.Overhead.Succeeds (C x) pos 𝒴 ∧ cost (C x) < min Lc Tmax) ≤ η

end Nci.Guarantees
