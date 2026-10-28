import Definitions.Nci.Work
import Mathlib.Analysis.SpecialFunctions.Log.Base

/-!
# NCI over threads, each value charged once over its copies: the vocabulary

`WeightsSupportFmt` charges `b_w` per weight gate a subcircuit uses and bounds a value's charge per thread through a
margin `μ` derived nowhere. Both fail on copies: a weight quantized once
for fprop and once for dgrad is two weight gates carrying one weight's information, and an activation's copies feed
counted multiplications that a margin of two copies does not pay for. `Threading.Sources` names the value each gate is
a copy of, so that weights are charged per source and fan-out is counted over all of a value's copies, and how many
optimizer steps each source is from the baseline.

* `Sources.src g`: the value `g` holds a copy of (a quantized, transposed, cast or recomputed copy maps to the value it
  copies; any other gate to itself). Each source is used at most `reuse` times per thread, over all its copies.
* `Sources.step s`: the optimizer steps between the baseline `θ` and the source `s`, `0` for one `θ` holds.
* `Sources.sourcesUsed R`: the sources of the weights `R` uses.
* `Sources.fanOut s`: the counted multiplications whose activation is a copy of `s`, over the whole circuit.
* `Sources.FanOutBound b_a e n`: every value's fan-out, charged at `b_a` per `e` counted multiplications, is at most
  the `log₂ n` bits a value of the activations' format carries: `b_a · fanOut s ≤ e · log₂ n`.
* `Sources.OffFresh k₀`: every counted multiplication reads a weight at least `k₀` optimizer steps from `θ`.
-/

namespace Nci.Threading

open Verity

/-- **The source of each value**, for a threading `T` of the reference circuit `c`: the value each gate is a copy of,
with each source used at most `T.reuse` times per thread over all its copies, and the optimizer steps between the
baseline and each source. -/
structure Sources {c : Registered} (T : Threading c) where
  /-- The value a gate holds a copy of: a weight quantized for fprop and for dgrad maps to the weight it quantizes,
  an activation's quantized, transposed, cast or recomputed copy to the activation; any other gate to itself. -/
  src : Fin c.D.N → Fin c.D.N
  /-- Each source is used at most `reuse` times per thread, over all its copies. -/
  reuse_le : ∀ s t, (T.M.filter fun g => src (T.weight g) = s ∧ T.thread (T.act g) = some t).card ≤ T.reuse
  /-- The optimizer steps between the baseline `θ` and a source. -/
  step : Fin c.D.N → ℕ
  step_θ : ∀ g ∈ c.θ, step (src g) = 0

namespace Sources

variable {c : Registered} {T : Threading c} (Src : T.Sources)

/-- The sources of the weights `R` uses. -/
def sourcesUsed (R : Finset (Fin c.D.N)) : Finset (Fin c.D.N) := (T.weightsUsed R).image Src.src

/-- The fan-out of a value over all its copies: the counted multiplications whose activation is a copy of `s`. -/
def fanOut (s : Fin c.D.N) : ℕ := (T.M.filter fun g => Src.src (T.act g) = s).card

/-- **Fan-out within a value's information**: charged at `b_a` per `e` counted multiplications, no value's fan-out over
its copies exceeds the `log₂ n` bits a value of the activations' format carries. -/
def FanOutBound (b_a : ℝ) (e n : ℕ) : Prop :=
  ∀ s, b_a * (Src.fanOut s : ℝ) ≤ (e : ℝ) * Real.logb 2 n

/-- **Off fresh weights**: every counted multiplication reads a weight at least `k₀` optimizer steps from `θ`; at
`k₀ ≥ 1`, none reads a weight `θ` holds or a copy of one. -/
def OffFresh (k₀ : ℕ) : Prop := ∀ g ∈ T.M, k₀ ≤ Src.step (Src.src (T.weight g))

end Sources

end Nci.Threading
