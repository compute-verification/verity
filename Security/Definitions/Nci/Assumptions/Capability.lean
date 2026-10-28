import Definitions.Nci.Capability
import Definitions.Accounting.Overhead
import Mathlib.Basic.ENNReal.Real
import Mathlib.Basic.Real.ENatENNReal

/-!
# The capability argument: its three scientific hypotheses

Why a protocol that splits declared computation into small inference blocks should slow the acquisition of a new
capability. The argument has four parts, and only the first two are about learning:

1. **Acquiring the target requires hard-to-localize computation**: coverage of efficient methods
   (`CoversEfficientMethods`).
2. **Small inference blocks can perform little of it**: local update hardness (`LocalUpdateHardness`) and distributed
   dependence (`DistributedDependence`), with Lemma 1 (`Verity.Guarantees.BlockDescription`) and Theorem 3
   (`Verity.Guarantees.Counting`).
3. **The protocol forces most physical compute into those blocks**: protocol soundness and the accounting, stated in
   `Security/README.md` and not here.
4. **So acquiring the target needs more fleet-time**: the floor and its counting (`Nci.Guarantees.CapabilityFloor`).

The protocol is not what makes training difficult; these hypotheses are. The protocol makes that difficulty matter.

Each hypothesis is a named `Prop`, and each keeps compression, recomputation and other algorithms possible rather
than excluding them by vocabulary:

* `LocalUpdateHardness`: a joint, time-bounded description lower bound for updated state, at prices per parameter slot
  that measure joint description cost, not storage width. A low-rank factorization, a predictor, a seed, a previous
  checkpoint or a replay of the training prefix within the decoder's budget lowers the prices; the premise must reflect
  them, and it fails where they make the state cheap.
* `DistributedDependence`: a block that completes a counted product of token `j` while handed fewer than `r` values
  from `j`'s own computation must hold more updated state than its budget pays for. With update hardness it forces `r`
  handed gates per use, with no information rate assumed for any activation. At width 1 it is a property of the
  wiring, checkable on a named circuit; a larger width is measured.
* `CoversEfficientMethods`: the task-level hypothesis. Computationally competitive ways of reaching the target can't
  escape every analysed structure. It is the least certain of the three, and the one a cheap compact-state method
  refutes.

None of these is "the final model is large", "every activation carries `β` new bits", "weights and activations pay
independent additive costs", "a whole active path doesn't fit, so the local bound holds", "the reference's product rate
is an adversarial ceiling", or "nothing derives the capability within the claimed time". Each of those would restate the
conclusion.
-/

namespace Nci.Assumptions

open Verity Finset Nci.Capability
open scoped ENNReal

/-- NEW PREMISE. **Local update hardness** ([LOCAL-UPDATE-HARDNESS]) with deficit `Λw`, within `t` steps of the decoder
`M` given the registered information `r`: in every honest run of the registered circuit, for every set `U` of updated
parameter slots and every choice `σ` of one eligible version of each, the tuple of those versions' values has local
description length at least `Σ_{i ∈ U} p_i − Λw`:
`LDL_{t,r}(w_U) ≥ Σ_{i ∈ U} p_i − Λw`.

It covers every collection a block can choose, not a typical tensor or a random subset, and it charges a slot once
however often a block reuses it. It is time-bounded: a short program that rebuilds the state by replaying training
counts only if it finishes within `t`.

It fails, or holds only at lower prices, when a block can replay the training prefix within `t`, when the update has a
small effective representation (low rank, few distinct values, a short seed), or when the chosen slots are cheaply
predictable from the registration; those cases mark where it applies. A deficit `Λw` that cancels every useful price
makes it empty. Each decoder found gives an upper bound and can refute a price; failing to find a shorter decoder is
evidence, not proof. -/
def LocalUpdateHardness (R : Registration) (B : Nci.Covering.MulBlocks R.d) (W : UpdatedState B)
    {ρ : Type} (M : Decoder ρ) (r : ρ) (t : ℕ) (Λw : ℝ) : Prop :=
  ∀ a, R.Honest a → ∀ (U : Finset (Fin B.nw)) (σ : Fin B.nw → Fin R.d.D.N), (∀ i ∈ U, σ i ∈ W.ver i) →
    ENNReal.ofReal (∑ i ∈ U, W.p i - Λw) ≤
      (LDL M t r (R.d.D.bits (R.d.D.eval a) ((U.image σ).sort (· ≤ ·))) : ℝ≥0∞)

/-- NEW PREMISE. **Distributed dependence** ([DISTRIBUTED-DEPENDENCE]) of width `r` at the description budget `kb`
(Lemma 1's `k + ⌈log₂ B⌉ + a_blk` plus update hardness's deficit `Λw`): a proof block of the registration that computes
a counted product of token `j` whole, while reading fewer than `r` gates of `j`'s computation `T j`, holds versions,
among its gates and what it reads, of updated slots priced above `kb`.

In words: unless a block is handed `r` values from a use's own upstream computation, reproducing the operand that use
needs forces it to obtain more updated state than one block can describe. The token computations are disjoint and hold
no registered gate (`Nci.Covering.MulBlocks.T_disjoint`, `T_θ`), so a block serves at most `k / r` uses.

At width 1 it is a property of the wiring alone, checkable on a named circuit. A larger width is the empirical part:
how many bits of a use's upstream values a block must import before rebuilding the rest within its budget gets cheap,
which the operand's local description length given the held state measures. A first layer, whose activations are a
function of the token id, fails it at every width, and so does a layer of fixed features `a_j = h_θ(x_j)` that a block
computes from registered information; their products are not counted. That alternative implementations keep a
comparable structure is the coverage question (`CoversEfficientMethods`), not this one. -/
def DistributedDependence (R : Registration) (B : Nci.Covering.MulBlocks R.d) (W : UpdatedState B) (r : ℕ)
    (kb : ℝ) : Prop :=
  ∀ (u : Fin R.B) (j : Fin B.nt), (∃ i, (i, j) ∈ B.P ∧ B.m (i, j) ⊆ R.P.gates u) →
    (R.d.D.into (R.P.gates u) ∩ B.T j).card < r →
    ∃ (U : Finset (Fin B.nw)) (σ : Fin B.nw → Fin R.d.D.N),
      (∀ i ∈ U, σ i ∈ W.ver i ∧ σ i ∈ R.P.gates u ∪ R.d.D.into (R.P.gates u)) ∧ kb < ∑ i ∈ U, W.p i

/-- NEW PREMISE. **Coverage of efficient methods** ([COVERAGE]) for the task `G` in the class `𝒜`, at the cost range
`Tmax`, with error `η`: for every computation and output positions, the probability that its run lies in `𝒜`, reaches
the target, costs below `Tmax`, and lacks the analysed structure `Core` is at most `η`:
`Pr[(C, x) ∈ 𝒜 ∧ C(x) ∈ 𝒴 ∧ cost < Tmax ∧ ¬ Core(C, x)] ≤ η`.

`Core C x` is that the run's computation contains a counted structure large enough for the target, each part of it
meeting `LocalUpdateHardness` and `DistributedDependence` at the block budgets; each method may have its own structure,
from a product argument, an accumulation argument or another, and none needs to reproduce a reference circuit.

It doesn't say that every successful method, however slow, has that structure: only that methods cheap enough to matter
do. A method that reduces the restricted state or dependence and reaches the target within the range refutes it at that
target and operating point. Full-parameter training being the unrestricted optimum is not enough: a compact-state
method at twice the cost bounds the slowdown by two. -/
def CoversEfficientMethods {In L : Type} {m : ℕ} (G : Accounting.Overhead.Task In (Fin m → List Bool))
    (𝒜 : Accounting.Overhead.Class In L) (Core : (In → Accounting.Actual.Run L) → In → Prop)
    (cost : Accounting.Actual.Run L → ℕ) (Tmax : ℕ) (η : ℝ≥0∞) : Prop :=
  ∀ (C : In → Accounting.Actual.Run L) (pos : Fin m → L × ℕ × ℕ),
    G.prob (fun x 𝒴 => (C, x) ∈ 𝒜 ∧ Accounting.Overhead.Succeeds (C x) pos 𝒴 ∧ cost (C x) < Tmax ∧ ¬ Core C x) ≤ η

end Nci.Assumptions
