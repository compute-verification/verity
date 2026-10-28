import Definitions.Nci.Capability
import Definitions.Accounting.Overhead
import Mathlib.Basic.ENNReal.Real
import Mathlib.Basic.Real.ENatENNReal

/-!
# The capability argument: four local assumption sets and coverage

Why splitting declared computation into small inference blocks should slow the acquisition of a new capability. A
proof block has a short description of everything it can extract (Lemma 1, `Verity.Guarantees.BlockDescription`:
`LDL_{t,R}(y_I) ≤ k' = k + ⌈log₂ B⌉ + a_blk`). Each local set below is one way to turn that into a cap on the counted
computation a block does; they are alternatives, not a conjunction, and differ in where the hard information lives:

1. **Updated state and upstream dependencies** (the primary hypothesis): `LocalUpdateHardness` with the graph
   certificate `Nci.Capability.UpstreamCert`. A block can hold little updated state, and serving a use without being
   handed a value from its upstream computation would need more (`Nci.Guarantees.UpstreamCap`). No activation is
   priced.
2. **Two operand families**: `LocalUpdateHardness` and `ActivationHardness`, each bounding its own family
   (`Nci.Guarantees.OperandPairsCap`), or the stronger `JointOperandHardness`, one budget for both
   (`Nci.Guarantees.JointOperandsCap`). The joint form's factor 4 is a new hypothesis, not better algebra.
3. **Hard activations and bounded fan-out**: `ActivationHardness` alone, each operand feeding at most `a_out` counted
   products (`Nci.Guarantees.FanOutCap`).
4. **A hard source recoverable from the block's outputs**: `SourceHardness` with `Recovers`
   (`Nci.Guarantees.RecoverableCap`), which counts outputs that reveal the source rather than multiplications.

and, separately,

5. **Coverage of efficient methods** (`CoversEfficientMethods`): computations cheap enough to matter that reach the target
   contain enough computation covered by one of the four.

The rates and deficits belong to the decoder budget, the registration, the arithmetic and the computation evaluated;
none is a universal bits-per-value constant. Each premise keeps compression, recomputation and other algorithms
possible: a low-rank factorization, a predictor, a seed, a checkpoint or a replay within the decoder's budget lowers a
price, and the premise must reflect it. Protocol soundness and the device model are outside these sets: they establish
where the block restrictions apply and bound the rest, not that learning is hard.
-/

namespace Nci.Assumptions

open Verity Finset Nci.Capability
open scoped ENNReal

/-- NEW PREMISE. **Local update hardness** ([LOCAL-UPDATE-HARDNESS]) with deficit `Λw`, within `t` steps of the decoder
`M` given the registered information `r`: in every honest run of the registered circuit, for every set `U` of updated
parameter slots and every choice `σ` of one eligible version of each, the tuple of those versions' values has local
description length at least `Σ_{i ∈ U} p_i − Λw`: `LDL_{t,r}(w_U) ≥ Σ_{i ∈ U} p_i − Λw`.

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

/-- NEW PREMISE. **Activation hardness** ([ACTIVATION-HARDNESS]) at rate `βx` with deficit `Λx`, within `t` steps: in
every honest run, for every set `V` of operand occurrences, their gates' values have local description length at least
`βx · |V| − Λx`: `LDL_{t,r}(a_V) ≥ βx |V| − Λx`.

It covers operands a block produces itself, from registered weights and a few imported values, and mixtures of layers
and steps: every operand collection an admissible block can hold, after it exploits everything it holds. A block that
imports updated weights and cheaply generates many operands refutes it, and so do exact decoders the experiments have
found for several uniform rates; a code rate measured for one tensor is not such a bound. It is the older
`Nci.Ldl.Assumptions.ActivationRate` on core's registration. -/
def ActivationHardness (R : Registration) {B : Nci.Covering.MulBlocks R.d} (O : Operands B)
    {ρ : Type} (M : Decoder ρ) (r : ρ) (t : ℕ) (βx Λx : ℝ) : Prop :=
  ∀ a, R.Honest a → ∀ V : Finset (Fin O.no),
    ENNReal.ofReal (βx * V.card - Λx) ≤
      (LDL M t r (R.d.D.bits (R.d.D.eval a) ((V.biUnion O.act).sort (· ≤ ·))) : ℝ≥0∞)

/-- NEW PREMISE. **Joint operand hardness** ([JOINT-OPERAND-HARDNESS]) with deficit `Λ`, within `t` steps: in every
honest run, for every set `U` of slots with versions `σ` and every set `V` of operand occurrences, their values together
have local
description length at least `Σ_{i ∈ U} p_i + βx · |V| − Λ`.

It is stronger than `LocalUpdateHardness` and `ActivationHardness` together: it says the two families don't share
description, when the same imported bits can help rebuild both (an operand computed from held weights, a weight inferred
from held operands). It is what the joint cap's factor 4 rests on. -/
def JointOperandHardness (R : Registration) {B : Nci.Covering.MulBlocks R.d} (W : UpdatedState B) (O : Operands B)
    {ρ : Type} (M : Decoder ρ) (r : ρ) (t : ℕ) (βx Λ : ℝ) : Prop :=
  ∀ a, R.Honest a → ∀ (U : Finset (Fin B.nw)) (σ : Fin B.nw → Fin R.d.D.N) (V : Finset (Fin O.no)),
    (∀ i ∈ U, σ i ∈ W.ver i) →
    ENNReal.ofReal (∑ i ∈ U, W.p i + βx * V.card - Λ) ≤
      (LDL M t r (R.d.D.bits (R.d.D.eval a) ((U.image σ ∪ V.biUnion O.act).sort (· ≤ ·))) : ℝ≥0∞)

/-- NEW PREMISE. **Source hardness** ([SOURCE-HARDNESS]) of the source `z` (a string the run determines, such as the
update's canonical encoding) of length `ℓz`, with deficiency `δ`, within `τ` steps: in every honest run,
`LDL_{τ,r}(z) ≥ ℓz − δ`. The deficiency is paid in every block's bound, so a `δ` far above `k'` leaves Set 4 empty;
a compressed representation of a structured update does not by itself give a source with small deficiency. -/
def SourceHardness (R : Registration) {ρ : Type} (M : Decoder ρ) (r : ρ) (τ : ℕ) (z : R.d.D.Transcript → List Bool)
    (ℓz δ : ℝ) : Prop :=
  ∀ a, R.Honest a → ENNReal.ofReal (ℓz - δ) ≤ (LDL M τ r (z (R.d.D.eval a)) : ℝ≥0∞)

/-- NEW PREMISE. **The source is recoverable from each block** ([SOURCE-RECOVERABLE]) at `βv` bits per eligible output:
the decoder, on proof block `u`'s Lemma 1 program (`blockProgram`, whose evaluator `e` also performs the recovery)
followed by a complement `s` of at most `ℓz − βv · J u` bits, outputs the source within `τ` steps. `J u` counts block
`u`'s eligible outputs.

It is a capability, like Lemma 1's `Evaluates`, and the structural heart of Set 4: correlated outputs need not shorten
the complement one coordinate each (for an exact linear map the rank counts, not the number of scalars), and inputs
the recovery needs must be among what the source-hardness premise's decoder is given. -/
def Recovers (R : Registration) {ρ : Type} (M : Decoder ρ) (r : ρ) (τ : ℕ) (e : List Bool)
    (z : R.d.D.Transcript → List Bool) (ℓz βv : ℝ) (J : Fin R.B → ℕ) : Prop :=
  ∀ a, R.Honest a → ∀ u, ∃ s : List Bool, (s.length : ℝ) ≤ ℓz - βv * J u ∧
    M.run r (blockProgram R e (R.d.D.eval a) u ++ s) τ = some (z (R.d.D.eval a))

/-- NEW PREMISE. **Coverage of efficient methods** ([COVERAGE]) for the task `G` in the class `𝒜`, at the cost range
`Tmax`, with error `η`: for every computation and output positions, the probability that its run lies in `𝒜`, reaches
the target, costs below `Tmax`, and has none of the analysed structures `Core` is at most `η`:
`Pr[(C, x) ∈ 𝒜 ∧ C(x) ∈ 𝒴 ∧ cost < Tmax ∧ ¬ Core(C, x)] ≤ η`.

`Core C x` is the disjunction, over the local sets, of: the run's computation meets that set's premises and contains at
least its minimum counted computation. Each method may have its own covered structure; none has to reproduce a
reference circuit.

It doesn't say that every successful method, however slow, has such a structure: only that methods cheap enough to
matter do. A method that reduces the restricted state or dependence and reaches the target within the range refutes it
at that target and operating point. Full-parameter training being the unrestricted optimum is not enough: a
compact-state method at twice the cost bounds the slowdown by two. It is the largest scientific conjecture here; a
first result states it for an explicit implementation family. -/
def CoversEfficientMethods {In L : Type} {m : ℕ} (G : Accounting.Overhead.Task In (Fin m → List Bool))
    (𝒜 : Accounting.Overhead.Class In L) (Core : (In → Accounting.Actual.Run L) → In → Prop)
    (cost : Accounting.Actual.Run L → ℕ) (Tmax : ℕ) (η : ℝ≥0∞) : Prop :=
  ∀ (C : In → Accounting.Actual.Run L) (pos : Fin m → L × ℕ × ℕ),
    G.prob (fun x 𝒴 => (C, x) ∈ 𝒜 ∧ Accounting.Overhead.Succeeds (C x) pos 𝒴 ∧ cost (C x) < Tmax ∧ ¬ Core C x) ≤ η

end Nci.Assumptions
