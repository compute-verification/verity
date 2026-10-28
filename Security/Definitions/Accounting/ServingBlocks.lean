import Definitions.Core.Formalism
import Mathlib.Basic.Real.Basic

/-!
# Serving blocks and the training class's counts (trusted)

A training class's counted operations sit in a circuit with its registered input gates, a declaration `d`
(`Verity.RegisteredCircuit`): at the serving-block bound, the actual circuit the fleet evaluates. Its per-block cap `q(c, k)`
bounds the weight of the operations computed wholly inside any block of at most `c` operation gates that reads at most
`k` unregistered input bits. A **serving block** is a server-time block's gates (`Verity.Block`): a server's gates
between two consecutive wipes. Each counted operation is assigned to the serving block holding its representative gate.
The serving blocks are the units `S` of a partition (`Verity.Partition`), its other units the dark remainder, and the
inference blocks are a `(c, k)`-partition (`Verity.Partition.IsCK`) refining it (`Verity.Partition.Refines`). The
serving-block bound (`Accounting.Guarantees.ServingBlocks`) is Theorem 3 (counting) applied to each serving block and
summed.

For a circuit `C` and counted operations `J` (`Counted`):
* `J.total`: `Σ_{v ∈ 𝒥} ω_v`;
* `J.assigned R`: the local count `Ω_R`, the weight of the operations whose representative gate is in `R`;
* `J.inside I`: the weight of the operations computed wholly inside `I` (`body v ⊆ I`);
* `J.outside R B`: the weight of `R`'s operations computed wholly inside none of the blocks `B i`;
* `J.split R B`: the part of `J.outside R B` whose representative is in one of the blocks, the operations split between a
  block and gates outside it.

A block's unregistered input bits are `|in(I) \ θ|` (`Verity.RegisteredCircuit.unregistered`), one bit per gate on a Boolean
circuit. Blocks here have at most `c` gates. Padding a block to exactly `c` gates (`Verity.Guarantees.Padding`) adds
only registered gates that read registered gates, so it adds no unregistered input bit and no counted operation: a cap
at exactly `c` gates, read on the padded circuit, gives the cap at most `c` on the circuit at hand.

The named premises are `PerBlockCap`, `UncertifiedCap`, `OperationBodies` (an operation's gates are unregistered and lead
to its representative) and `Overlap` (a gate is in at most `μ` operations' bodies). The last two bound the operations
assigned to a serving block but not computed wholly inside it: each has a gate outside the block that a gate of the
block reads, an unregistered input of the block, so at most `ω_max μ k_S` of `Ω_S` escapes the per-block cap.
-/

namespace Accounting.ServingBlocks

open Verity Finset

/-- **Counted operations** of a circuit `C`: a training class's `𝒥`, Theorem 3's `𝒱`. Each is named by its
representative gate `v ∈ V`, an operation gate of `C` and one of the gates that compute it (`body v`), so distinct
operations have distinct representatives. It is computed wholly inside a set `I` of gates when `body v ⊆ I`. `ω v ≥ 0`
is its counting weight. -/
structure Counted (C : Circuit) where
  V : Finset (Fin C.N)
  body : Fin C.N → Finset (Fin C.N)
  ω : Fin C.N → ℝ
  V_computed : ∀ v ∈ V, C.isInput v = false
  rep_mem : ∀ v ∈ V, v ∈ body v
  ω_nonneg : ∀ v ∈ V, 0 ≤ ω v

namespace Counted

variable {C : Circuit} (J : Counted C)

/-- `Σ_{v ∈ 𝒥} ω_v`: the counted weight. -/
def total : ℝ := ∑ v ∈ J.V, J.ω v

/-- **The local count** `Ω_R`: the weight of the counted operations whose representative gate is in `R`. -/
def assigned (R : Finset (Fin C.N)) : ℝ := ∑ v ∈ J.V with v ∈ R, J.ω v

/-- The weight of the counted operations computed wholly inside `I`. -/
def inside (I : Finset (Fin C.N)) : ℝ := ∑ v ∈ J.V with J.body v ⊆ I, J.ω v

/-- The weight of the counted operations assigned to `R` that are computed wholly inside none of the blocks `B i`. -/
def outside (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι] (B : ι → Finset (Fin C.N)) : ℝ :=
  ∑ v ∈ J.V with v ∈ R ∧ ∀ i, ¬ J.body v ⊆ B i, J.ω v

/-- The weight of the counted operations assigned to `R` whose representative is in one of the blocks `B i` but that are
computed wholly inside none of them: the operations split between a block and gates outside it. -/
def split (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι] (B : ι → Finset (Fin C.N)) : ℝ :=
  ∑ v ∈ J.V with v ∈ R ∧ (∃ i, v ∈ B i) ∧ ∀ i, ¬ J.body v ⊆ B i, J.ω v

end Counted

/-- **A per-block cap** `q(c, k)`: every block `I` of at most `c` operation gates of `d` that reads at most `k`
unregistered input bits (`|in(I) \ θ|`) holds counted weight at most `q c k` computed wholly inside it. A property of the
training class at the block size `(c, k)` (Lemma 1, `Verity.Guarantees.BlockDescription`), read on the circuit at
hand. -/
def PerBlockCap (d : RegisteredCircuit) (J : Counted d.D) (q : ℕ → ℕ → ℝ) (c k : ℕ) : Prop :=
  ∀ I ⊆ d.D.computed, I.card ≤ c → (d.unregistered I).card ≤ k → J.inside I ≤ q c k

/-- **An uncertified cap** `q_U`: the weight of `R`'s counted operations computed wholly inside none of the blocks `B i`
is at most `qU`. At a serving block with the inference blocks inside it, this is `q_U(S)`; at the dark remainder, with no
blocks, `q_U(L_dark)`. -/
def UncertifiedCap {C : Circuit} (J : Counted C) (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι]
    (B : ι → Finset (Fin C.N)) (qU : ℝ) : Prop :=
  J.outside R B ≤ qU

/-- **Operation bodies**: every gate of a counted operation is unregistered (outside `θ`), and each one but its
representative is read by another gate of the operation, so the operation's gates lead to its representative inside the
operation. It allows an operation split across a wipe or between servers; such an operation then has a gate outside its
serving block that a gate of the block reads. -/
def OperationBodies (d : RegisteredCircuit) (J : Counted d.D) : Prop :=
  ∀ v ∈ J.V, ∀ u ∈ J.body v, u ∉ d.θ ∧ (u ≠ v → ∃ g ∈ J.body v, u ∈ d.D.reads g)

/-- **Overlap** `μ`: every gate is in the bodies of at most `μ` counted operations (`μ = 1` when no two operations share
a gate). -/
def Overlap {C : Circuit} (J : Counted C) (μ : ℕ) : Prop :=
  ∀ g, (J.V.filter fun v => g ∈ J.body v).card ≤ μ

end Accounting.ServingBlocks
