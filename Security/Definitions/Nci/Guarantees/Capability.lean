import Definitions.Nci.Assumptions.Capability
import Definitions.Core.Guarantees.Formalism

/-!
# The capability argument: what the hypotheses give

Part 2 of the argument (`Definitions.Nci.Assumptions.Capability`), small inference blocks can perform little of the
hard-to-localize computation, and part 4, its consequence for cost:

* `UpdatedStatePerBlock`: Lemma 1 (`Verity.Guarantees.BlockDescription`) with local update hardness. A proof block can
  hold updated state priced at most `k' + Λw`, `k' = k + ⌈log₂ B⌉ + a_blk`, whether it loads it, decompresses it or
  computes it.
* `ProductsPerBlock`: with distributed dependence of width `r` and one price `βw` per slot, each proof block computes
  at most `(k / r) · (k' + Λw) / βw` counted products whole: it holds at most `(k' + Λw)/βw` slots, and each use it
  serves costs it `r` handed gates from that use's own computation. No activation is assumed to carry an information
  rate.
* `CapabilityFloor`: coverage with a cost floor `Lc` on covered computations (from `ProductsPerBlock` and Theorem 3,
  `Verity.Guarantees.Counting`) gives that reaching the target below `min(Lc, Tmax)` has probability at most `η`.

Part 3, that the protocol forces most physical compute into blocks these caps apply to, is the accounting the README's
security argument states; the slowdown's comparison against a reference is its last step.
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

/-- **A block computes few counted products** (distributed dependence of width `r ≥ 1` with a held-state budget, one
price `βw > 0` per slot). If every proof block holds updated state priced at most `kb` (`UpdatedStatePerBlock`'s
conclusion) and distributed dependence holds at width `r` and budget `kb`, each proof block of a `(c, k)`-partition
computes at most `(k / r) · (kb / βw)` counted products whole. -/
def ProductsPerBlock : Prop :=
  ∀ (R : Registration) (c k : ℕ), R.P.IsCK R.d c k →
    ∀ (B : Nci.Covering.MulBlocks R.d) (W : UpdatedState B) (r : ℕ) (βw kb : ℝ), 0 < r → 0 < βw →
      (∀ i, W.p i = βw) → HeldStateAtMost R W kb → Assumptions.DistributedDependence R B W r kb →
      ∀ u, ((B.P.filter fun q => B.m q ⊆ R.P.gates u).card : ℝ) ≤ (k : ℝ) / r * (kb / βw)

/-- **Covered success below the floor is rare** (part 4's counting step). Under coverage of efficient methods at cost
range `Tmax` and error `η`, if every covered computation in the class `𝒜` costs at least `Lc` (the per-block cap and
Theorem 3 give such a floor for compliant computations), then for every computation the probability of lying in `𝒜`,
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
