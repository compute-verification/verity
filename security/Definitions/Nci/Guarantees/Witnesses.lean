module

public import Definitions.Nci.Witness
public import Definitions.Nci.Assumptions
public import Definitions.Nci.Threads
public import Definitions.Core.Circuit
public import Definitions.Core.RegisteredCircuit
public import Definitions.Core.Partition
public import Definitions.Nci
public import Definitions.Nci.Dampener
public import Definitions.Nci.Reads
public import Definitions.Nci.Assumptions.Threads
public import Definitions.Nci.Assumptions.Baseline
public import Definitions.Nci.Assumptions.Layered
public import Definitions.Nci.Assumptions.Dampener
public import Definitions.Nci.Guarantees.CoverTraining
public import Definitions.Nci.Guarantees.Remote
public import Definitions.Nci.Cut.Toy

/-!
# The inference-only policy's guarantees: their witnesses

Each guarantee over the inference-only policy that concludes a bound is of the lock's `exclusion` kind, and has two
witnesses at the toy circuit (`Nci.Capability.Toy`, with the instances of `Nci.Witness`), listed under the lock's
`witnesses`. `Inference` concludes the policy at `(c, k)` (`Complies`); every other guarantee takes it at a rate
(`CompliesRate Γ α`), and its witnesses are at `Γ = 2` and `α = 1/2`, so that the read cap `α Γ` is `1`:

* `X_honest`: the guarantee's premises hold together at the toy declaration `Toy.D`, complying with its one proof block
  of two gates, every premise a definition states written out (a bare order of numbers, such as `0 ≤ α`, is left out);
* `X_nontrivial`: at an explicit instance, each other premise holds, one named premise fails, and so does the bound:
  without that premise the bound says something false.

The premise each `X_nontrivial` drops is the one the bound rests on: the coverage `κ` when the bound is a share of the
reference circuit (at the threading `T0` of no counted multiplication, every bound is `0`), `ThetaRegistered` when it is
in the read cap `k` (at `Dθ`, the declaration registering the token too, the toy complies at `k = 0`), `WindowsHold`
for the dampener, and PoUW's soundness (`PouwSound`) for the remote dampener, whose witnesses take its declared
computation at the min-cut toy (`Nci.Cut.Toy`), where the count verdict `CompliesAtMost` holds under (H) and (E). The cover theorem's join
(`Nci.Cover.Guarantees.CoverTraining`) is stated at the multiplication blocks `mb0` of no product, where coverage
`α₀ > 0` fails.
-/

@[expose] public section

open Verity.Game renaming prPMFReal → prob

namespace Nci.Guarantees

open Verity Finset Nci.Capability.Toy Nci.Witness Nci.Dampener

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

/-- **`DampenerThreaded`'s premises hold together** at the toy, with `Γ = 2`, `α = 1/2`, `b_w = b_a = 1/2`, the
threading `T1` at `(dd, e) = (1, 0)`, `κ = 1`, monitored compute `C = 2` all certified (`W = 2`), no unmonitored
compute or fault, one counted multiplication (`M = 1`, the declared one), one server, `F = 2`, `H = L = R = 1`. -/
def DampenerThreaded_honest : Prop :=
  CompliesRate 2 (1 / 2) D ∧ Assumptions.ThetaRegistered D D M1 ∧
    T1.Threaded 2 1 1 0 ∧ Assumptions.WeightsIncompressible D T1 2 1 (1 / 2) (1 / 2) ∧
    (D.D.size D.D.computed : ℝ) ≤ 2 ∧ 1 ≤ 0 + 0 + T1.M.card + 0 ∧ Assumptions.WindowLength (1 / 2) 2 1 1 ∧
    ((0 : ℕ) : ℝ) ≤ T1.reuse * (2 * 1) / (4 * (1 : ℕ) * (1 / 2)) ∧
    Assumptions.WindowsHold T1 0 1 (1 / 2) 2 1 1 1 (1 / 2) (1 / 2) 1 0

/-- **`DampenerThreaded` without `WindowsHold`**: at the threading `T0`, with no state (`R = 0`) and no rate (`F = 0`),
one counted multiplication in the residual (`W = 0`, `M = M_r = 1`) is held by no window, and the efficiency `1`
exceeds the bound `0`. -/
def DampenerThreaded_nontrivial : Prop :=
  (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 2 ∧
    (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ CompliesRate 2 (1 / 2) D ∧
    Assumptions.ThetaRegistered D D M1 ∧
    (T0 D).Threaded 2 1 1 0 ∧ Assumptions.WeightsIncompressible D (T0 D) 2 1 (1 / 2) (1 / 2) ∧
    (D.D.size D.D.computed : ℝ) ≤ 2 ∧ 1 ≤ 0 + 1 + (T0 D).M.card + 0 ∧
    ((0 : ℕ) : ℝ) ≤ 1 * 0 / 2 ∧ ((1 : ℕ) : ℝ) ≤ 1 * (2 - 0) / 2 ∧ ((0 : ℕ) : ℝ) ≤ 1 * 0 * (2 + 0) / 2 ∧
    ((1 : ℕ) : ℝ) * 0 * 1 ≤ 2 ∧ Assumptions.WindowLength (1 / 2) 0 1 0 ∧
    ((0 : ℕ) : ℝ) ≤ (T0 D).reuse * (2 * 0) / (4 * (1 : ℕ) * (1 / 2)) ∧
    ¬ Assumptions.WindowsHold (T0 D) 1 1 (1 / 2) 0 1 1 0 (1 / 2) (1 / 2) 1 0 ∧
    ¬ (efficiency 1 1 2 0 ≤
        unregisteredShare 2 0 +
            min (residualShare 0 2) (memoryCap (T0 D).reuse 1 0 ((1 : ℕ) / (2 : ℕ) : ℝ) 1 (1 / 2) (1 / 2) * (1 + 1 / 1)) +
          declaredShare 1 (rhoT ((1 : ℕ) / (2 : ℕ) : ℝ) (1 / 2) ((T0 D).bound 1 (1 / 2) 1 0)) 2 0 + 0)

/-- **`DampenerThreadedSlots`' premises hold together**: `DampenerThreaded_honest`'s instance, the weight in its own
parameter slot (`σ1`). -/
def DampenerThreadedSlots_honest : Prop :=
  CompliesRate 2 (1 / 2) D ∧ Assumptions.ThetaRegistered D D M1 ∧
    T1.Threaded 2 1 1 0 ∧ Assumptions.WeightsIncompressibleSlots D T1 σ1 2 1 (1 / 2) (1 / 2) ∧
    (D.D.size D.D.computed : ℝ) ≤ 2 ∧ 1 ≤ 0 + 0 + T1.M.card + 0 ∧ Assumptions.WindowLength (1 / 2) 2 1 1 ∧
    ((0 : ℕ) : ℝ) ≤ T1.reuse * (2 * 1) / (4 * (1 : ℕ) * (1 / 2)) ∧
    Assumptions.WindowsHold T1 0 1 (1 / 2) 2 1 1 1 (1 / 2) (1 / 2) 1 0

/-- **`DampenerThreadedSlots` without `WindowsHold`**: `DampenerThreaded_nontrivial`'s instance, at `T0`'s one parameter
slot (`σ0`). -/
def DampenerThreadedSlots_nontrivial : Prop :=
  (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 2 ∧
    (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ CompliesRate 2 (1 / 2) D ∧
    Assumptions.ThetaRegistered D D M1 ∧
    (T0 D).Threaded 2 1 1 0 ∧ Assumptions.WeightsIncompressibleSlots D (T0 D) (σ0 D) 2 1 (1 / 2) (1 / 2) ∧
    (D.D.size D.D.computed : ℝ) ≤ 2 ∧ 1 ≤ 0 + 1 + (T0 D).M.card + 0 ∧
    ((0 : ℕ) : ℝ) ≤ 1 * 0 / 2 ∧ ((1 : ℕ) : ℝ) ≤ 1 * (2 - 0) / 2 ∧ ((0 : ℕ) : ℝ) ≤ 1 * 0 * (2 + 0) / 2 ∧
    ((1 : ℕ) : ℝ) * 0 * 1 ≤ 2 ∧ Assumptions.WindowLength (1 / 2) 0 1 0 ∧
    ((0 : ℕ) : ℝ) ≤ (T0 D).reuse * (2 * 0) / (4 * (1 : ℕ) * (1 / 2)) ∧
    ¬ Assumptions.WindowsHold (T0 D) 1 1 (1 / 2) 0 1 1 0 (1 / 2) (1 / 2) 1 0 ∧
    ¬ (efficiency 1 1 2 0 ≤
        unregisteredShare 2 0 +
            min (residualShare 0 2) (memoryCap (T0 D).reuse 1 0 ((1 : ℕ) / (2 : ℕ) : ℝ) 1 (1 / 2) (1 / 2) * (1 + 1 / 1)) +
          declaredShare 1 (rhoT ((1 : ℕ) / (2 : ℕ) : ℝ) (1 / 2) ((T0 D).bound 1 (1 / 2) 1 0)) 2 0 + 0)

/-- **`DampenerRemote`'s premises hold together** over the outcomes `Bool` sampled uniformly: one PoUW window of capacity
`C = 2`, its certificate never wrong (`p = 0`), all of it certified (`W = 2`), no incorrect proof block, so that every
detection holds at `q = 1` and the residual is `0`, and the declaration the toy `Nci.Cut.Toy`, complying with
`CompliesAtMost 1 2 2` under (H) at unit costs and (E) at `b = 2`, whose two MACs are the counted multiplications. -/
def DampenerRemote_honest : Prop :=
  Assumptions.WithinCompute OneWindow.cap 2 ∧ Assumptions.PouwSound (PMF.uniformOfFintype Bool) OneWindow.right 0 ∧
    (∀ ω, Assumptions.TerminalDetection (OneWindow.bad ω) (OneWindow.bad ω) OneWindow.ident OneWindow.time 0) ∧
    Assumptions.SampledDetection (PMF.uniformOfFintype Bool) OneWindow.bad OneWindow.bad OneWindow.ident
      OneWindow.time 1 ∧
    CompliesAtMost 1 2 2 Cut.Toy.d ∧ Cut.ValuesInjective Cut.Toy.W Cut.Toy.val ∧
    Cut.NoSharedPair Cut.Toy.W Cut.Toy.val ∧ Cut.Assumptions.Hardness Cut.Toy.d Cut.Toy.arrays Cut.Toy.val ![1, 1] 2 ∧
    Cut.Assumptions.Exports Cut.Toy.W 2 ∧ Assumptions.SourcesCover 2 0 0 Cut.Toy.W.N 0 ∧
    (∀ ω, Assumptions.ResidualAtMost 0 2 2 0 2 (wrongCapacity OneWindow.cap (OneWindow.right ω))
      (∑ b ∈ OneWindow.bad ω, OneWindow.time b))

/-- **`DampenerRemote` without PoUW's soundness**: one window of capacity `C = 100`, its certificate wrong at every
outcome, so the residual holds its capacity, `M_r = 100` counted multiplications, all of `C` certified, `κ = 2`,
`ε' = 1/2` and `θ = 1/4`, beside `DampenerRemote_honest`'s declaration, whose bound `X = 8` is `8/100` of the compute.
The efficiency `1` exceeds the bound `ε' + θ + 8/100 = 83/100` at every outcome, so the event has probability `0`,
below `1 − δ = 1`. -/
def DampenerRemote_nontrivial : Prop :=
  (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 2 ∧ (∀ i, 0 ≤ (fun _ : Fin 1 => (100 : ℝ)) i) ∧
    Assumptions.WithinCompute (fun _ : Fin 1 => (100 : ℝ)) 100 ∧
    ¬ Assumptions.PouwSound (PMF.uniformOfFintype Bool) OneWindow.wrong 0 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) * 1 ≤ 100 ∧
    (∀ ω, ∀ b ∈ OneWindow.bad ω, OneWindow.ident b ∈ (∅ : Finset Unit)) ∧
    (∀ ω, Assumptions.TerminalDetection (OneWindow.bad ω) (OneWindow.bad ω) OneWindow.ident OneWindow.time 0) ∧
    (∀ ω j t, ∑ b ∈ OneWindow.bad ω with OneWindow.ident b = j ∧ t ≤ OneWindow.time b ∧ OneWindow.time b ≤ t + 0,
      OneWindow.time b ≤ 0 * 0) ∧
    (0 : ℝ) < 1 / 4 ∧ (∀ b, 0 ≤ OneWindow.time b) ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧
    (∀ ω, ∀ b ∈ OneWindow.bad ω, OneWindow.time b ≤ 0) ∧
    Assumptions.SampledDetection (PMF.uniformOfFintype Bool) OneWindow.bad OneWindow.bad OneWindow.ident
      OneWindow.time 1 ∧
    (0 : ℝ) < 2 ∧ (0 : ℝ) < 100 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 2 ∧ CompliesAtMost 1 2 2 Cut.Toy.d ∧ Cut.ValuesInjective Cut.Toy.W Cut.Toy.val ∧
    Cut.NoSharedPair Cut.Toy.W Cut.Toy.val ∧ Cut.Assumptions.Hardness Cut.Toy.d Cut.Toy.arrays Cut.Toy.val ![1, 1] 2 ∧
    Cut.Assumptions.Exports Cut.Toy.W 2 ∧
    0 < 100 ∧ Assumptions.SourcesCover 100 0 100 Cut.Toy.W.N 0 ∧ ((0 : ℕ) : ℝ) ≤ 1 * 2 * 0 / 2 ∧
    (∀ ω, Assumptions.ResidualAtMost 100 2 100 0 100 (wrongCapacity (fun _ : Fin 1 => (100 : ℝ)) (OneWindow.wrong ω))
      (∑ b ∈ OneWindow.bad ω, OneWindow.time b)) ∧
    ((0 : ℕ) : ℝ) ≤ 2 * 0 * (100 + 0) / 2 ∧
    ¬ (1 - (0 / (1 / 2) + ((∅ : Finset Unit).card : ℝ) * ((1 - 1) / 1 * 0) / (1 / 4 * 100)) ≤
      prob (PMF.uniformOfFintype Bool) (fun _ => efficiency 2 100 100 0 ≤
          unregisteredShare 100 0 +
              (1 - unregisteredShare 100 0) *
                (residualShare 100 100 + 0 * 100 / 100 + 1 / 2 + lagShare ((∅ : Finset Unit).card) 0 1 + 1 / 4) +
            efficiency 2 (Cut.workHE 2 2 (1 * 1) 2 Cut.Toy.W.O) 100 0 + 0 ∧
        remoteDampener 1 (unregisteredShare 100 0)
            ((1 - unregisteredShare 100 0) *
                (residualShare 100 100 + 0 * 100 / 100 + 1 / 2 + lagShare ((∅ : Finset Unit).card) 0 1 + 1 / 4) +
              efficiency 2 (Cut.workHE 2 2 (1 * 1) 2 Cut.Toy.W.O) 100 0 + 0) ≤
          hardwareMultiple 1 2 100 100 0))

end Nci.Guarantees

namespace Nci.Cover.Guarantees

open Verity Nci.Capability.Toy Nci.Witness

/-- **`CoverTraining`'s premises hold together** at the toy, with `Γ = 2`, `α = 1/2`, `b = 1`, `r = d = Toy.D` through
the containment of the toy in itself, the multiplication blocks `mb0` of no product at depth `D = 2`, block size
`s = 4 = w₀ Γ + α Γ + width(θ)` at `w₀ = 1`, and `α₀ = 0`, the share of the toy's gates that are counted products. -/
def CoverTraining_honest : Prop :=
  Nci.CompliesRate 2 (1 / 2) D ∧ Nci.Assumptions.ThetaRegistered D D M1 ∧ Assumptions.UpdateHardness (mb0 D) 4 1 ∧
    (mb0 D).Upstream (fun _ => ∅) 2 ∧ Assumptions.CapBelowDepth 1 1 2 ∧ Assumptions.BlockFits D 2 1 1 4 ∧
    Nci.Assumptions.Coverage 0 D (mb0 D).P.card

/-- **`CoverTraining` without coverage**: at `CoverTraining_honest`'s instance, `α₀ = 2` is no coverage of the toy by
`mb0`'s no product, and `α₀ b size(r) = 4` exceeds `(α² Γ + b α) size(d) = 2`. -/
def CoverTraining_nontrivial : Prop :=
  (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ Nci.CompliesRate 2 (1 / 2) D ∧ Nci.Assumptions.ThetaRegistered D D M1 ∧
    Assumptions.UpdateHardness (mb0 D) 4 1 ∧ (mb0 D).Upstream (fun _ => ∅) 2 ∧ Assumptions.CapBelowDepth 1 1 2 ∧
    Assumptions.BlockFits D 2 1 1 4 ∧ ¬ Nci.Assumptions.Coverage 2 D (mb0 D).P.card ∧
    ¬ (2 * 1 * D.D.size D.D.computed ≤ ((1 / 2 : ℝ) * (1 / 2) * 2 + 1 * (1 / 2)) * D.D.size D.D.computed)

end Nci.Cover.Guarantees
