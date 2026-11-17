module

public import Definitions.Slowdown.Witness
public import Definitions.InferenceOnly.Assumptions
public import Definitions.Slowdown.Threads
public import Definitions.Core.Circuit
public import Definitions.Core.RegisteredCircuit
public import Definitions.Core.Partition
public import Definitions.InferenceOnly
public import Definitions.InferenceOnly.Reads
public import Definitions.Slowdown.Assumptions.Threads
public import Definitions.InferenceOnly.Assumptions.Theta
public import Definitions.Slowdown.Assumptions.Layered

/-!
# The inference-only policy's guarantees: their witnesses

Each guarantee over the inference-only policy that concludes a bound is of the lock's `exclusion` kind, and has two
witnesses at the toy circuit (`Slowdown.Capability.Toy`, with the instances of `Slowdown.Witness`), listed under the lock's
`witnesses`. `Inference` concludes the policy at `(c, k)` (`Complies`); every other guarantee takes it at a rate
(`CompliesRate Γ α`), and its witnesses are at `Γ = 2` and `α = 1/2`, so that the read cap `α Γ` is `1`:

* `X_honest`: the guarantee's premises hold together at the toy declaration `Toy.D`, complying with its one proof block
  of two gates, every premise a definition states written out (a bare order of numbers, such as `0 ≤ α`, is left out);
* `X_nontrivial`: at an explicit instance, each other premise holds, one named premise fails, and so does the bound:
  without that premise the bound says something false.

The premise each `X_nontrivial` drops is the one the bound rests on: the coverage `κ` when the bound is a share of the
reference training algorithm (at the threading `T0` of no counted multiplication, every bound is `0`), `ThetaRegistered` when it is
in the read cap `k` (at `Dθ`, the declaration registering the token too, the toy complies at `k = 0`).
-/

@[expose] public section

open InferenceOnly

namespace Slowdown.Guarantees

open Verity Finset Slowdown.Capability.Toy Slowdown.Witness

/-- **`ThreadedTraining`'s and `ThreadedTrainingOn`'s premises hold together** at the toy, with `Γ = 2`, `α = 1/2`,
`b_w = b_a = 1/2`, `κ = 1`, `r = d = Toy.D` through the containment of the toy in itself, the threading `T1` and
`(dd, e) = (1, 0)`: a small subcircuit holding gate `2` reads the token, one fresh bit, and is charged half a bit for the
weight and half for the token. -/
def ThreadedTraining_honest : Prop :=
  CompliesRate 2 (1 / 2) D ∧ Assumptions.ThetaRegistered D D M1 ∧
    T1.Threaded 2 1 1 0 ∧ Assumptions.WeightsIncompressible D T1 2 1 (1 / 2) (1 / 2) ∧
    Assumptions.Coverage 1 D (2 * T1.M.card)

/-- **`ThreadedTraining` without coverage**: at the threading `T0` of no counted multiplication, `κ = 1` is no coverage,
and the bound `κ b_a size(r) ≤ 2 α Λ size(d)` fails, `Λ` being `0`. -/
def ThreadedTraining_nontrivial : Prop :=
  (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ CompliesRate 2 (1 / 2) D ∧
    Assumptions.ThetaRegistered D D M1 ∧
    (T0 D).Threaded 2 1 1 0 ∧ Assumptions.WeightsIncompressible D (T0 D) 2 1 (1 / 2) (1 / 2) ∧
    ¬ Assumptions.Coverage 1 D (2 * (T0 D).M.card) ∧
    ¬ (1 * (1 / 2) * D.D.size D.D.computed ≤
        2 * (1 / 2 : ℝ) * (T0 D).bound 1 (1 / 2) 1 0 * D.D.size D.D.computed)

/-- **`ThreadedTrainingOn` without coverage**: `ThreadedTraining_nontrivial`'s instance, through the containment with no
dark gate. -/
def ThreadedTrainingOn_nontrivial : Prop :=
  (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ CompliesRate 2 (1 / 2) D ∧
    Assumptions.ThetaRegistered D D M1 ∧
    (T0 D).Threaded 2 1 1 0 ∧ Assumptions.WeightsIncompressible D (T0 D) 2 1 (1 / 2) (1 / 2) ∧
    ¬ Assumptions.Coverage 1 D (2 * (T0 D).M.card) ∧
    ¬ (1 * (1 / 2) * D.D.size D.D.computed ≤
        2 * (1 / 2) * M1.dark.card +
          2 * (1 / 2 : ℝ) * (T0 D).bound 1 (1 / 2) 1 0 * D.D.size D.D.computed)

/-- **`ThreadedThrottled`'s, `ThreadedThrottledOn`'s and `ThreadedThrottledDark`'s premises hold together**:
`ThreadedTraining_honest`'s instance, which needs no coverage. -/
def ThreadedThrottled_honest : Prop :=
  CompliesRate 2 (1 / 2) D ∧ Assumptions.ThetaRegistered D D M1 ∧
    T1.Threaded 2 1 1 0 ∧ Assumptions.WeightsIncompressible D T1 2 1 (1 / 2) (1 / 2)

/-- **`ThreadedThrottled` without `ThetaRegistered`**: `Dθ` complies at `α = 0`, so at read cap `α Γ = 0`, and the bound
`b_a #M ≤ α Λ size(d)` fails, its right side `0` and gate `2` the counted multiplication of `T1`. -/
def ThreadedThrottled_nontrivial : Prop :=
  (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ CompliesRate 2 0 Dθ ∧
    ¬ Assumptions.ThetaRegistered D Dθ Mθ ∧
    T1.Threaded 2 0 1 0 ∧ Assumptions.WeightsIncompressible D T1 2 0 (1 / 2) (1 / 2) ∧
    ¬ ((1 / 2 : ℝ) * T1.M.card ≤ 0 * T1.bound 0 (1 / 2) 1 0 * Dθ.D.size Dθ.D.computed)

/-- **`ThreadedThrottledOn` without `ThetaRegistered`**: `ThreadedThrottled_nontrivial`'s instance, where `Dθ` computes
gate `2`, so that it is in `M ∩ S`. -/
def ThreadedThrottledOn_nontrivial : Prop :=
  (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ CompliesRate 2 0 Dθ ∧
    ¬ Assumptions.ThetaRegistered D Dθ Mθ ∧
    T1.Threaded 2 0 1 0 ∧ Assumptions.WeightsIncompressible D T1 2 0 (1 / 2) (1 / 2) ∧
    ¬ ((1 / 2 : ℝ) * (T1.M ∩ Mθ.S).card ≤ 0 * T1.bound 0 (1 / 2) 1 0 * Dθ.D.size Dθ.D.computed)

/-- **`ThreadedThrottledDark` without `ThetaRegistered`**: `ThreadedThrottled_nontrivial`'s instance, with no dark gate:
`#M = 1` exceeds `#dark + α Λ size(d) / b_a = 0`. -/
def ThreadedThrottledDark_nontrivial : Prop :=
  (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ CompliesRate 2 0 Dθ ∧
    ¬ Assumptions.ThetaRegistered D Dθ Mθ ∧
    T1.Threaded 2 0 1 0 ∧ Assumptions.WeightsIncompressible D T1 2 0 (1 / 2) (1 / 2) ∧
    ¬ ((T1.M.card : ℝ) ≤ Mθ.dark.card + 0 * T1.bound 0 (1 / 2) 1 0 * Dθ.D.size Dθ.D.computed / (1 / 2))

/-- **`ThreadedThrottledSlots`', `ThreadedThrottledOnSlots`' and `ThreadedThrottledDarkSlots`' premises hold
together**: `ThreadedThrottled_honest`'s instance, the weight in its own parameter slot (`σ1`). -/
def ThreadedThrottledSlots_honest : Prop :=
  CompliesRate 2 (1 / 2) D ∧ Assumptions.ThetaRegistered D D M1 ∧
    T1.Threaded 2 1 1 0 ∧ Assumptions.WeightsIncompressibleSlots D T1 σ1 2 1 (1 / 2) (1 / 2)

/-- **`ThreadedThrottledSlots` without `ThetaRegistered`**: `ThreadedThrottled_nontrivial`'s instance, the weight in its
own parameter slot (`σ1`). -/
def ThreadedThrottledSlots_nontrivial : Prop :=
  (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ CompliesRate 2 0 Dθ ∧
    ¬ Assumptions.ThetaRegistered D Dθ Mθ ∧
    T1.Threaded 2 0 1 0 ∧ Assumptions.WeightsIncompressibleSlots D T1 σ1 2 0 (1 / 2) (1 / 2) ∧
    ¬ ((1 / 2 : ℝ) * T1.M.card ≤ 0 * T1.bound 0 (1 / 2) 1 0 * Dθ.D.size Dθ.D.computed)

/-- **`ThreadedThrottledOnSlots` without `ThetaRegistered`**: `ThreadedThrottledOn_nontrivial`'s instance, the weight
in its own parameter slot (`σ1`). -/
def ThreadedThrottledOnSlots_nontrivial : Prop :=
  (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ CompliesRate 2 0 Dθ ∧
    ¬ Assumptions.ThetaRegistered D Dθ Mθ ∧
    T1.Threaded 2 0 1 0 ∧ Assumptions.WeightsIncompressibleSlots D T1 σ1 2 0 (1 / 2) (1 / 2) ∧
    ¬ ((1 / 2 : ℝ) * (T1.M ∩ Mθ.S).card ≤ 0 * T1.bound 0 (1 / 2) 1 0 * Dθ.D.size Dθ.D.computed)

/-- **`ThreadedThrottledDarkSlots` without `ThetaRegistered`**: `ThreadedThrottledDark_nontrivial`'s instance, the
weight in its own parameter slot (`σ1`). -/
def ThreadedThrottledDarkSlots_nontrivial : Prop :=
  (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ CompliesRate 2 0 Dθ ∧
    ¬ Assumptions.ThetaRegistered D Dθ Mθ ∧
    T1.Threaded 2 0 1 0 ∧ Assumptions.WeightsIncompressibleSlots D T1 σ1 2 0 (1 / 2) (1 / 2) ∧
    ¬ ((T1.M.card : ℝ) ≤ Mθ.dark.card + 0 * T1.bound 0 (1 / 2) 1 0 * Dθ.D.size Dθ.D.computed / (1 / 2))

/-- **`LayeredTraining`'s premises hold together** at the toy, with `Γ = 2`, `α = 1/2`, `b_w = b_a = 1/2`, `κ = 1`, the
threading `T1` keyed with no roots (`layered1`, `d = e = 1`, `eLk = 0`, `b = 2`) and `E = 0`. -/
def LayeredTraining_honest : Prop :=
  Nonempty (Layered T1 1 1 0 2) ∧ CompliesRate 2 (1 / 2) D ∧
    Assumptions.ThetaRegistered D D M1 ∧
    Assumptions.CapBelowColumns 1 0 2 ∧ Assumptions.WeightsIncompressible D T1 2 1 (1 / 2) (1 / 2) ∧
    Assumptions.Coverage 1 D (2 * T1.M.card)

/-- **`LayeredTraining` without coverage**: at the threading `T0` keyed with no roots (`layered0`), `κ = 1` is no
coverage, and the bound fails, `Λ` being `0`. -/
def LayeredTraining_nontrivial : Prop :=
  Nonempty (Layered (T0 D) 1 0 0 2) ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧
    CompliesRate 2 (1 / 2) D ∧
    Assumptions.ThetaRegistered D D M1 ∧
    Assumptions.CapBelowColumns 1 0 2 ∧ Assumptions.WeightsIncompressible D (T0 D) 2 1 (1 / 2) (1 / 2) ∧
    ¬ Assumptions.Coverage 1 D (2 * (T0 D).M.card) ∧
    ¬ (1 * (1 / 2) * D.D.size D.D.computed ≤
        2 * (1 / 2 : ℝ) * (T0 D).bound 1 (1 / 2) (1 - 0) (0 + 0 * 0) * D.D.size D.D.computed)

end Slowdown.Guarantees
