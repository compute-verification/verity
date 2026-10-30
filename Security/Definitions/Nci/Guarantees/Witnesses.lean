import Definitions.Nci.Witness
import Definitions.Nci.Guarantees
import Definitions.Nci.Dampener
import Definitions.Nci.Reads
import Definitions.Nci.Assumptions.Threads
import Definitions.Nci.Assumptions.Baseline
import Definitions.Nci.Assumptions.Activations
import Definitions.Nci.Assumptions.Layered
import Definitions.Nci.Assumptions.Dampener
import Definitions.Nci.Assumptions.Training
import Definitions.Nci.Assumptions.PackedTraining
import Definitions.Nci.Support

/-!
# The inference-only policy's guarantees in `(c, k)`: their witnesses

Each guarantee over the policy `Complies c k` that concludes a bound is of the lock's `exclusion` kind, and has two
witnesses at the toy circuit (`Nci.Capability.Toy`, with the instances of `Nci.Witness`), listed under the lock's
`witnesses`:

* `X_honest`: the guarantee's premises hold together at the toy declaration `Toy.D`, complying at `(2, 1)` with its one
  proof block, every premise a definition states written out (a bare order of numbers, such as `0 < c`, is left out);
* `X_nontrivial`: at an explicit instance, each other premise holds, one named premise fails, and so does the bound:
  without that premise the bound says something false.

The premise each `X_nontrivial` drops is the one the bound rests on: the coverage `κ` when the bound is a share of the
reference circuit (at the threading `T0` of no counted multiplication, every bound is `0`), `ThetaRegistered` when it is
in the read cap `k` (at `Dθ`, the declaration registering the token too, the toy complies at `k = 0`), the size of the
reference circuit for `Training`, the sessions for `Inference`, and `WindowsHold` for the dampener.
-/

namespace Nci.Guarantees

open Verity Finset Nci.Capability.Toy Nci.Witness Nci.Dampener

/-- **`ThreadedTraining`'s and `ThreadedTrainingOn`'s premises hold together** at the toy, with `c = 2`, `k = 1`,
`b_w = b_a = 1/2`, `κ = 1`, `r = d = Toy.D` through the containment of the toy in itself, the threading `T1` and
`(dd, e) = (1, 0)`: a small subcircuit holding gate `2` reads the token, one fresh bit, and is charged half a bit for the
weight and half for the token. -/
def ThreadedTraining_honest : Prop :=
  Complies 2 1 D ∧ Assumptions.ThetaRegistered D D M1 ∧
    T1.Threaded 2 1 1 0 ∧ Assumptions.WeightsIncompressible D T1 2 1 (1 / 2) (1 / 2) ∧
    Assumptions.Coverage 1 D (2 * T1.M.card)

/-- **`ThreadedTraining` without coverage**: at the threading `T0` of no counted multiplication, `κ = 1` is no coverage,
and the bound `κ b_a size(r) ≤ 2 (k / c) Λ size(d)` fails, `Λ` being `0`. -/
def ThreadedTraining_nontrivial : Prop :=
  0 < 2 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ Complies 2 1 D ∧
    Assumptions.ThetaRegistered D D M1 ∧
    (T0 D).Threaded 2 1 1 0 ∧ Assumptions.WeightsIncompressible D (T0 D) 2 1 (1 / 2) (1 / 2) ∧
    ¬ Assumptions.Coverage 1 D (2 * (T0 D).M.card) ∧
    ¬ (1 * (1 / 2) * D.D.size D.D.computed ≤
        2 * (1 / 2 : ℝ) * (T0 D).bound 1 (1 / 2) 1 0 * D.D.size D.D.computed)

/-- **`ThreadedTrainingOn` without coverage**: `ThreadedTraining_nontrivial`'s instance, through the containment with no
dark gate. -/
def ThreadedTrainingOn_nontrivial : Prop :=
  0 < 2 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ Complies 2 1 D ∧
    Assumptions.ThetaRegistered D D M1 ∧
    (T0 D).Threaded 2 1 1 0 ∧ Assumptions.WeightsIncompressible D (T0 D) 2 1 (1 / 2) (1 / 2) ∧
    ¬ Assumptions.Coverage 1 D (2 * (T0 D).M.card) ∧
    ¬ (1 * (1 / 2) * D.D.size D.D.computed ≤
        2 * (1 / 2) * M1.dark.card +
          2 * (1 / 2 : ℝ) * (T0 D).bound 1 (1 / 2) 1 0 * D.D.size D.D.computed)

/-- **`ActivationsBoundOn`'s premises hold together** at the toy, with `c = 2`, `k = K = 1`, `b_w = b_a = 1/2`,
`r = d = Toy.D`, the threading `T1` with no links, `dd = 1` and `e = 1`: the token enters one counted multiplication. -/
def ActivationsBoundOn_honest : Prop :=
  Complies 2 1 D ∧ Assumptions.ThetaRegistered D D M1 ∧
    Assumptions.EntersAtMost T1 1 ∧ Assumptions.DrawsAtLeast (noLinks T1) 1 ∧
    Assumptions.ActivationsIncompressibleAll D T1 (noLinks T1) 2 1 (1 / 2) (1 / 2)

/-- **`ActivationsBoundOn` without `ThetaRegistered`**: `Dθ`, the toy with the token registered, complies at `k = 0`,
and the bound `b_a #(M ∩ S) ≤ (k / c) Λ size(d)` fails, its right side `0` and gate `2` a counted multiplication `Dθ`
computes. -/
def ActivationsBoundOn_nontrivial : Prop :=
  0 < 2 ∧ ((0 : ℕ) : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ Complies 2 0 Dθ ∧
    ¬ Assumptions.ThetaRegistered D Dθ Mθ ∧
    Assumptions.EntersAtMost T1 1 ∧ Assumptions.DrawsAtLeast (noLinks T1) 1 ∧
    Assumptions.ActivationsIncompressibleAll D T1 (noLinks T1) 2 1 (1 / 2) (1 / 2) ∧
    ¬ ((1 / 2 : ℝ) * (T1.M ∩ Mθ.S).card ≤
        ((0 : ℕ) / (2 : ℕ) : ℝ) * T1.bound 1 (1 / 2) 1 1 * Dθ.D.size Dθ.D.computed)

/-- **`ActivationsTrainingOn`'s premises hold together**: `ActivationsBoundOn_honest`'s instance at coverage `κ = 1`. -/
def ActivationsTrainingOn_honest : Prop :=
  Complies 2 1 D ∧ Assumptions.ThetaRegistered D D M1 ∧
    Assumptions.EntersAtMost T1 1 ∧ Assumptions.DrawsAtLeast (noLinks T1) 1 ∧
    Assumptions.ActivationsIncompressibleAll D T1 (noLinks T1) 2 1 (1 / 2) (1 / 2) ∧
    Assumptions.Coverage 1 D (2 * T1.M.card)

/-- **`ActivationsTrainingOn` without coverage**: at the threading `T0`, with `e = 0`, `κ = 1` is no coverage and the
bound fails, `Λ` being `0` and no gate dark. -/
def ActivationsTrainingOn_nontrivial : Prop :=
  0 < 2 ∧ ((1 : ℕ) : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ Complies 2 1 D ∧
    Assumptions.ThetaRegistered D D M1 ∧
    Assumptions.EntersAtMost (T0 D) 0 ∧ Assumptions.DrawsAtLeast (noLinks (T0 D)) 1 ∧
    Assumptions.ActivationsIncompressibleAll D (T0 D) (noLinks (T0 D)) 2 1 (1 / 2) (1 / 2) ∧
    ¬ Assumptions.Coverage 1 D (2 * (T0 D).M.card) ∧
    ¬ (1 * (1 / 2) * D.D.size D.D.computed ≤
        2 * (1 / 2) * M1.dark.card +
          2 * (1 / 2 : ℝ) * (T0 D).bound 1 (1 / 2) 1 0 * D.D.size D.D.computed)

/-- **`LayeredTraining`'s premises hold together** at the toy, with `c = 2`, `k = 1`, `b_w = b_a = 1/2`, `κ = 1`, the
threading `T1` keyed with no roots (`layered1`, `d = e = 1`, `eLk = 0`, `b = 2`) and `E = 0`. -/
def LayeredTraining_honest : Prop :=
  Nonempty (Layered T1 1 1 0 2) ∧ Complies 2 1 D ∧
    Assumptions.ThetaRegistered D D M1 ∧
    Assumptions.CapBelowColumns 1 0 2 ∧ Assumptions.WeightsIncompressible D T1 2 1 (1 / 2) (1 / 2) ∧
    Assumptions.Coverage 1 D (2 * T1.M.card)

/-- **`LayeredTraining` without coverage**: at the threading `T0` keyed with no roots (`layered0`), `κ = 1` is no
coverage, and the bound fails, `Λ` being `0`. -/
def LayeredTraining_nontrivial : Prop :=
  Nonempty (Layered (T0 D) 1 0 0 2) ∧ 0 < 2 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ Complies 2 1 D ∧
    Assumptions.ThetaRegistered D D M1 ∧
    Assumptions.CapBelowColumns 1 0 2 ∧ Assumptions.WeightsIncompressible D (T0 D) 2 1 (1 / 2) (1 / 2) ∧
    ¬ Assumptions.Coverage 1 D (2 * (T0 D).M.card) ∧
    ¬ (1 * (1 / 2) * D.D.size D.D.computed ≤
        2 * (1 / 2 : ℝ) * (T0 D).bound 1 (1 / 2) (1 - 0) (0 + 0 * 0) * D.D.size D.D.computed)

/-- **`DampenerThreaded`'s premises hold together** at the toy, with `c = 2`, `k = 1`, `b_w = b_a = 1/2`, the
threading `T1` at `(dd, e) = (1, 0)`, `κ = 1`, monitored compute `C = 2` all certified (`W = 2`), no unmonitored
compute or fault, one counted multiplication (`M = 1`, the declared one), one server, `F = 2`, `H = L = R = 1`. -/
def DampenerThreaded_honest : Prop :=
  Complies 2 1 D ∧ Assumptions.ThetaRegistered D D M1 ∧
    T1.Threaded 2 1 1 0 ∧ Assumptions.WeightsIncompressible D T1 2 1 (1 / 2) (1 / 2) ∧
    (D.D.size D.D.computed : ℝ) ≤ 2 ∧ 1 ≤ 0 + 0 + T1.M.card + 0 ∧ Assumptions.WindowLength 2 1 2 1 1 ∧
    ((0 : ℕ) : ℝ) ≤ T1.reuse * (2 * 1) / (4 * (1 : ℕ) * (1 / 2)) ∧
    Assumptions.WindowsHold T1 2 1 0 1 2 1 1 1 (1 / 2) (1 / 2) 1 0

/-- **`DampenerThreaded` without `WindowsHold`**: at the threading `T0`, with no state (`R = 0`) and no rate (`F = 0`),
one counted multiplication in the residual (`W = 0`, `M = M_r = 1`) is held by no window, and the efficiency `1`
exceeds the bound `0`. -/
def DampenerThreaded_nontrivial : Prop :=
  (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ 0 < 2 ∧ (0 : ℝ) < 1 / 2 ∧
    (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ Complies 2 1 D ∧
    Assumptions.ThetaRegistered D D M1 ∧
    (T0 D).Threaded 2 1 1 0 ∧ Assumptions.WeightsIncompressible D (T0 D) 2 1 (1 / 2) (1 / 2) ∧
    (D.D.size D.D.computed : ℝ) ≤ 2 ∧ 1 ≤ 0 + 1 + (T0 D).M.card + 0 ∧
    ((0 : ℕ) : ℝ) ≤ 1 * 0 / 2 ∧ ((1 : ℕ) : ℝ) ≤ 1 * (2 - 0) / 2 ∧ ((0 : ℕ) : ℝ) ≤ 1 * 0 * (2 + 0) / 2 ∧
    ((1 : ℕ) : ℝ) * 0 * 1 ≤ 2 ∧ Assumptions.WindowLength 2 1 0 1 0 ∧
    ((0 : ℕ) : ℝ) ≤ (T0 D).reuse * (2 * 0) / (4 * (1 : ℕ) * (1 / 2)) ∧
    ¬ Assumptions.WindowsHold (T0 D) 2 1 1 1 0 1 1 0 (1 / 2) (1 / 2) 1 0 ∧
    ¬ (efficiency 1 1 2 0 ≤
        unmonitoredShare 2 0 +
            min (residualShare 0 2) (memoryCap (T0 D).reuse 1 0 ((1 : ℕ) / (2 : ℕ) : ℝ) 1 (1 / 2) (1 / 2) * (1 + 1 / 1)) +
          declaredShare 1 (rhoT ((1 : ℕ) / (2 : ℕ) : ℝ) (1 / 2) ((T0 D).bound 1 (1 / 2) 1 0)) 2 0 + 0)

/-- **`DampenerThreadedSlots`' premises hold together**: `DampenerThreaded_honest`'s instance, the weight in its own
parameter slot (`σ1`). -/
def DampenerThreadedSlots_honest : Prop :=
  Complies 2 1 D ∧ Assumptions.ThetaRegistered D D M1 ∧
    T1.Threaded 2 1 1 0 ∧ Assumptions.WeightsIncompressibleSlots D T1 σ1 2 1 (1 / 2) (1 / 2) ∧
    (D.D.size D.D.computed : ℝ) ≤ 2 ∧ 1 ≤ 0 + 0 + T1.M.card + 0 ∧ Assumptions.WindowLength 2 1 2 1 1 ∧
    ((0 : ℕ) : ℝ) ≤ T1.reuse * (2 * 1) / (4 * (1 : ℕ) * (1 / 2)) ∧
    Assumptions.WindowsHold T1 2 1 0 1 2 1 1 1 (1 / 2) (1 / 2) 1 0

/-- **`DampenerThreadedSlots` without `WindowsHold`**: `DampenerThreaded_nontrivial`'s instance, at `T0`'s one parameter
slot (`σ0`). -/
def DampenerThreadedSlots_nontrivial : Prop :=
  (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ 0 < 2 ∧ (0 : ℝ) < 1 / 2 ∧
    (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ Complies 2 1 D ∧
    Assumptions.ThetaRegistered D D M1 ∧
    (T0 D).Threaded 2 1 1 0 ∧ Assumptions.WeightsIncompressibleSlots D (T0 D) (σ0 D) 2 1 (1 / 2) (1 / 2) ∧
    (D.D.size D.D.computed : ℝ) ≤ 2 ∧ 1 ≤ 0 + 1 + (T0 D).M.card + 0 ∧
    ((0 : ℕ) : ℝ) ≤ 1 * 0 / 2 ∧ ((1 : ℕ) : ℝ) ≤ 1 * (2 - 0) / 2 ∧ ((0 : ℕ) : ℝ) ≤ 1 * 0 * (2 + 0) / 2 ∧
    ((1 : ℕ) : ℝ) * 0 * 1 ≤ 2 ∧ Assumptions.WindowLength 2 1 0 1 0 ∧
    ((0 : ℕ) : ℝ) ≤ (T0 D).reuse * (2 * 0) / (4 * (1 : ℕ) * (1 / 2)) ∧
    ¬ Assumptions.WindowsHold (T0 D) 2 1 1 1 0 1 1 0 (1 / 2) (1 / 2) 1 0 ∧
    ¬ (efficiency 1 1 2 0 ≤
        unmonitoredShare 2 0 +
            min (residualShare 0 2) (memoryCap (T0 D).reuse 1 0 ((1 : ℕ) / (2 : ℕ) : ℝ) 1 (1 / 2) (1 / 2) * (1 + 1 / 1)) +
          declaredShare 1 (rhoT ((1 : ℕ) / (2 : ℕ) : ℝ) (1 / 2) ((T0 D).bound 1 (1 / 2) 1 0)) 2 0 + 0)

/-- **`Inference`'s premises hold together** at the toy: its circuit is Boolean, and its one proof block is a session of
`c = 2` gates reading the weight, registered, and its prompt, the token, of `k = 1` bit. -/
def Inference_honest : Prop :=
  D.D.Boolean ∧ Assumptions.Sessions D P1 (fun _ => {1}) 2 1

/-- **`Inference` without sessions**: with no prompt, the toy's proof block is no session at `k = 0`, and the toy does
not comply at `(2, 0)`, its block reading the token. -/
def Inference_nontrivial : Prop :=
  D.D.Boolean ∧ ¬ Assumptions.Sessions D P1 (fun _ => ∅) 2 0 ∧ ¬ Complies 2 0 D

/-- **`Training`'s premises hold together** at the toy, at coverage `κ = 0`: every declaration that implements it and
complies runs the chain of no multiplication. -/
def Training_honest : Prop :=
  0 < D.D.size D.D.computed ∧ Assumptions.InClass D 2 1 1 1 0 1

/-- **`Training` without a computed gate**: the declaration of no gates is in the class at `κ = 1`, every chain's
coverage of no gate being complete, and its slowdown factor is `0`, below `κ F > 0`: it implements itself and complies
with no proof block, at `0 / 0 = 0`. -/
def Training_nontrivial : Prop :=
  ¬ 0 < E0.D.size E0.D.computed ∧ 0 < 2 ∧ 0 < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ Assumptions.InClass E0 2 1 1 1 1 1 ∧
    ¬ (ENNReal.ofReal (1 * F ((1 : ℕ) / (2 : ℕ) : ℝ) 1 1 1 1) ≤ slowdown (Complies 2 1) E0)

/-- **`UpdatedMatMul`'s premises hold together** at the toy, with `c = 2`, `k = 1`, `b_w = b_a = 1/2` and the
one-product chain `ch1` of inner dimension `d = 1`: an inference block holding the product reads the token, one fresh bit,
and is charged half a bit for its weight entry and half for its input entry. -/
def UpdatedMatMul_honest : Prop :=
  Complies 2 1 D ∧ Assumptions.InnerAtLeast ch1 1 ∧ Assumptions.ActivationsIncompressible D ch1 2 1 (1 / 2) (1 / 2) ∧
    Assumptions.PartialSumsIncompressible D ch1 2 1 (1 / 2) (1 / 2)

/-- **`UpdatedMatMul` without incompressible activations**: charging `b_w = 1` and `b_a = 2`, more than the one bit an
inference block holding the product reads, `F ≥ 3/2` and the bound `F · 2G ≤ size(D)` fails, `G = 1` and
`size(D) = 2`. -/
def UpdatedMatMul_nontrivial : Prop :=
  0 < 2 ∧ 0 < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ Assumptions.InnerAtLeast ch1 1 ∧ Complies 2 1 D ∧
    ¬ Assumptions.ActivationsIncompressible D ch1 2 1 1 2 ∧ Assumptions.PartialSumsIncompressible D ch1 2 1 1 2 ∧
    ¬ (F ((1 : ℕ) / (2 : ℕ) : ℝ) 1 1 2 1 * (2 * ch1.G) ≤ D.D.size D.D.computed)

/-- **`UpdatedMatMulReads`' premises hold together** at the toy's one proof block, with `c = 2`, `k = 1`,
`b_w = b_a = 1/2` and the chain `ch1`. -/
def UpdatedMatMulReads_honest : Prop :=
  Assumptions.InnerAtLeast ch1 1 ∧ Assumptions.SmallBlocks D P1 2 1 ∧
    Assumptions.ActivationsIncompressible D ch1 2 1 (1 / 2) (1 / 2) ∧
    Assumptions.PartialSumsIncompressible D ch1 2 1 (1 / 2) (1 / 2)

/-- **`UpdatedMatMulReads` without incompressible activations**: `UpdatedMatMul_nontrivial`'s charges, at which the read
factor is at least `3/2`, above the one bit the toy's proof block reads. -/
def UpdatedMatMulReads_nontrivial : Prop :=
  (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ Assumptions.InnerAtLeast ch1 1 ∧ Assumptions.SmallBlocks D P1 2 1 ∧
    ¬ Assumptions.ActivationsIncompressible D ch1 2 1 1 2 ∧ Assumptions.PartialSumsIncompressible D ch1 2 1 1 2 ∧
    ¬ (ch1.G * readFactor 1 1 2 1 ≤ ∑ u, (novelReads D (P1.gates u) : ℝ))

/-- **`ContainedTraining`'s premises hold together** at the toy, with `c = 2`, `k = 1`, `b_w = b_a = 1/2`, `κ = 1`, the
chain `ch1` at `dd = 1` and the run of the toy with every input `1` as the registered values: the product's value is the
token's, two values for one bit of charge. -/
def ContainedTraining_honest : Prop :=
  Complies 2 1 D ∧ Assumptions.ThetaRegistered D D M1 ∧
    Assumptions.ActivationsSupport D ones ch1 (1 + 1 / 2 + 1 / 2) (1 / 2) (1 / 2) ∧
    Assumptions.PartialSumsSupport D ones ch1 2 1 (1 / 2) (1 / 2) ∧ Assumptions.Coverage 1 D (2 * ch1.G)

/-- **`ContainedTraining` without coverage**: at the chain `ch0` of no multiplication, charging `b_w = 1` and
`b_a = 2`, `κ = 1` is no coverage, and `κ F size(r) ≤ size(d)` fails, `F ≥ 3/2`. -/
def ContainedTraining_nontrivial : Prop :=
  0 < 2 ∧ 0 < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (∀ ℓ, 1 ≤ (ch0 C).d ℓ) ∧ Complies 2 1 D ∧
    Assumptions.ThetaRegistered D D M1 ∧ Assumptions.ActivationsSupport D ones (ch0 C) (1 + 1 + 2) 1 2 ∧
    Assumptions.PartialSumsSupport D ones (ch0 C) 2 1 1 2 ∧ ¬ Assumptions.Coverage 1 D (2 * (ch0 C).G) ∧
    ¬ (1 * F ((1 : ℕ) / (2 : ℕ) : ℝ) 1 1 2 1 * D.D.size D.D.computed ≤ D.D.size D.D.computed)

/-- **`ContainedMatMulOn`'s premises hold together** at the toy, with `c = 2`, `k = K = 1`, `b_w = b_a = 1/2`, `μ = 1`,
the chain `ch1` at `dd = emax = 1`, and no dark gate. -/
def ContainedMatMulOn_honest : Prop :=
  Complies 2 1 D ∧ Assumptions.ThetaRegistered D D M1 ∧ Assumptions.DarkReads M1 1 ∧
    Assumptions.ActivationsSupport D ones ch1 (1 + 1 / 2 + 1 / 2) (1 / 2) (1 / 2) ∧
    Assumptions.PartialSumsSupport D ones ch1 2 1 (1 / 2) (1 / 2)

/-- **`ContainedMatMulOn` without `ThetaRegistered`**: `Dθ` complies at `k = 0`, and the bound fails, its right side `0`
and the product at gate `2`, which `Dθ` computes, charged `μ b_a / max(emax, K/(4 b_w dd)) = 1/2`. -/
def ContainedMatMulOn_nontrivial : Prop :=
  (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ 0 < 2 ∧ ((0 : ℕ) : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
    (∀ ℓ, 1 ≤ ch1.d ℓ) ∧ (∀ ℓ, ch1.e ℓ ≤ 1) ∧ Complies 2 0 Dθ ∧ ¬ Assumptions.ThetaRegistered D Dθ Mθ ∧
    Assumptions.DarkReads Mθ 1 ∧ Assumptions.ActivationsSupport D ones ch1 (1 + 1 / 2 + 1 / 2) (1 / 2) (1 / 2) ∧
    Assumptions.PartialSumsSupport D ones ch1 2 1 (1 / 2) (1 / 2) ∧
    ¬ (max (min ((1 + √1) ^ 2 * (1 / 2 * (1 / 2) * 1 / 1)) (√2 * (1 + √1) * √(1 - 1) * (1 / 2 * √(1 / 2 / 1))))
          (1 * (1 / 2) / max (1 : ℝ) (1 / (4 * (1 / 2) * 1))) *
        (univ.filter fun p : (ℓ : Fin ch1.L) × Fin ch1.N × Fin (ch1.e ℓ) × Fin (ch1.d ℓ) =>
          ch1.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ Mθ.S).card ≤
      ((0 : ℕ) / (2 : ℕ) : ℝ) * Dθ.D.size Dθ.D.computed + (1 - 1) * D.D.width Mθ.dark)

/-- **`ContainedTrainingOn`'s premises hold together**: `ContainedMatMulOn_honest`'s instance at coverage `κ = 1`. -/
def ContainedTrainingOn_honest : Prop :=
  Complies 2 1 D ∧ Assumptions.ThetaRegistered D D M1 ∧ Assumptions.DarkReads M1 1 ∧
    Assumptions.ActivationsSupport D ones ch1 (1 + 1 / 2 + 1 / 2) (1 / 2) (1 / 2) ∧
    Assumptions.PartialSumsSupport D ones ch1 2 1 (1 / 2) (1 / 2) ∧ Assumptions.Coverage 1 D (2 * ch1.G)

/-- **`ContainedTrainingOn` without `ThetaRegistered`**: `ContainedMatMulOn_nontrivial`'s instance at coverage `κ = 1`;
the bound's left side is `1/2 · κ size(r) = 1` and its right side `0`. -/
def ContainedTrainingOn_nontrivial : Prop :=
  (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ 0 < 2 ∧ ((0 : ℕ) : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
    (∀ ℓ, 1 ≤ ch1.d ℓ) ∧ (∀ ℓ, ch1.e ℓ ≤ 1) ∧ Complies 2 0 Dθ ∧ ¬ Assumptions.ThetaRegistered D Dθ Mθ ∧
    Assumptions.DarkReads Mθ 1 ∧ Assumptions.ActivationsSupport D ones ch1 (1 + 1 / 2 + 1 / 2) (1 / 2) (1 / 2) ∧
    Assumptions.PartialSumsSupport D ones ch1 2 1 (1 / 2) (1 / 2) ∧ Assumptions.Coverage 1 D (2 * ch1.G) ∧
    ¬ (max (min ((1 + √1) ^ 2 * (1 / 2 * (1 / 2) * 1 / 1)) (√2 * (1 + √1) * √(1 - 1) * (1 / 2 * √(1 / 2 / 1))))
          (1 * (1 / 2) / max (1 : ℝ) (1 / (4 * (1 / 2) * 1))) *
        (1 * D.D.size D.D.computed - 2 * Mθ.dark.card) ≤
      2 * (((0 : ℕ) / (2 : ℕ) : ℝ) * Dθ.D.size Dθ.D.computed + (1 - 1) * D.D.width Mθ.dark))

/-- **`IncompressibleContainedTraining`'s premises hold together** at the toy, with `c = 2`, `k = 1`,
`b_w = b_a = 1/2`, `κ = 1`, the chain `ch1` and the threading `T1`. -/
def IncompressibleContainedTraining_honest : Prop :=
  Complies 2 1 D ∧ Assumptions.ThetaRegistered D D M1 ∧
    Assumptions.TrainingIncompressible D ch1 T1 2 1 (1 / 2) (1 / 2) ∧ Assumptions.Coverage 1 D (2 * ch1.G)

/-- **`IncompressibleContainedTraining` without coverage**: `ContainedTraining_nontrivial`'s instance, at the threading
`T0`. -/
def IncompressibleContainedTraining_nontrivial : Prop :=
  0 < 2 ∧ 0 < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (∀ ℓ, 1 ≤ (ch0 C).d ℓ) ∧ Complies 2 1 D ∧
    Assumptions.ThetaRegistered D D M1 ∧ Assumptions.TrainingIncompressible D (ch0 C) (T0 D) 2 1 1 2 ∧
    ¬ Assumptions.Coverage 1 D (2 * (ch0 C).G) ∧
    ¬ (1 * F ((1 : ℕ) / (2 : ℕ) : ℝ) 1 1 2 1 * D.D.size D.D.computed ≤ D.D.size D.D.computed)

/-- **`IncompressibleContainedMatMulOn`'s premises hold together**: `ContainedMatMulOn_honest`'s instance, with
`TrainingIncompressible` at the threading `T1`. -/
def IncompressibleContainedMatMulOn_honest : Prop :=
  Complies 2 1 D ∧ Assumptions.ThetaRegistered D D M1 ∧ Assumptions.DarkReads M1 1 ∧
    Assumptions.TrainingIncompressible D ch1 T1 2 1 (1 / 2) (1 / 2)

/-- **`IncompressibleContainedMatMulOn` without `ThetaRegistered`**: `ContainedMatMulOn_nontrivial`'s instance. -/
def IncompressibleContainedMatMulOn_nontrivial : Prop :=
  (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ 0 < 2 ∧ ((0 : ℕ) : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
    (∀ ℓ, 1 ≤ ch1.d ℓ) ∧ (∀ ℓ, ch1.e ℓ ≤ 1) ∧ Complies 2 0 Dθ ∧ ¬ Assumptions.ThetaRegistered D Dθ Mθ ∧
    Assumptions.DarkReads Mθ 1 ∧ Assumptions.TrainingIncompressible D ch1 T1 2 1 (1 / 2) (1 / 2) ∧
    ¬ (max (min ((1 + √1) ^ 2 * (1 / 2 * (1 / 2) * 1 / 1)) (√2 * (1 + √1) * √(1 - 1) * (1 / 2 * √(1 / 2 / 1))))
          (1 * (1 / 2) / max (1 : ℝ) (1 / (4 * (1 / 2) * 1))) *
        (univ.filter fun p : (ℓ : Fin ch1.L) × Fin ch1.N × Fin (ch1.e ℓ) × Fin (ch1.d ℓ) =>
          ch1.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ Mθ.S).card ≤
      ((0 : ℕ) / (2 : ℕ) : ℝ) * Dθ.D.size Dθ.D.computed + (1 - 1) * ∑ v ∈ Mθ.dark, (novelReads D {v} : ℝ))

/-- **`IncompressibleContainedTrainingOn`'s premises hold together**: `IncompressibleContainedMatMulOn_honest`'s
instance at coverage `κ = 1`. -/
def IncompressibleContainedTrainingOn_honest : Prop :=
  Complies 2 1 D ∧ Assumptions.ThetaRegistered D D M1 ∧ Assumptions.DarkReads M1 1 ∧
    Assumptions.TrainingIncompressible D ch1 T1 2 1 (1 / 2) (1 / 2) ∧ Assumptions.Coverage 1 D (2 * ch1.G)

/-- **`IncompressibleContainedTrainingOn` without `ThetaRegistered`**: `ContainedTrainingOn_nontrivial`'s instance. -/
def IncompressibleContainedTrainingOn_nontrivial : Prop :=
  (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ 0 < 2 ∧ ((0 : ℕ) : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
    (∀ ℓ, 1 ≤ ch1.d ℓ) ∧ (∀ ℓ, ch1.e ℓ ≤ 1) ∧ Complies 2 0 Dθ ∧ ¬ Assumptions.ThetaRegistered D Dθ Mθ ∧
    Assumptions.DarkReads Mθ 1 ∧ Assumptions.TrainingIncompressible D ch1 T1 2 1 (1 / 2) (1 / 2) ∧
    Assumptions.Coverage 1 D (2 * ch1.G) ∧
    ¬ (max (min ((1 + √1) ^ 2 * (1 / 2 * (1 / 2) * 1 / 1)) (√2 * (1 + √1) * √(1 - 1) * (1 / 2 * √(1 / 2 / 1))))
          (1 * (1 / 2) / max (1 : ℝ) (1 / (4 * (1 / 2) * 1))) *
        (1 * D.D.size D.D.computed - 2 * Mθ.dark.card) ≤
      2 * (((0 : ℕ) / (2 : ℕ) : ℝ) * Dθ.D.size Dθ.D.computed + (1 - 1) * ∑ v ∈ Mθ.dark, (novelReads D {v} : ℝ)))

/-- **`PackedIncompressibleContainedMatMulOn`'s premises hold together**: `IncompressibleContainedMatMulOn_honest`'s
instance, `ch1` as a packed chain of one product to a gate. -/
def PackedIncompressibleContainedMatMulOn_honest : Prop :=
  Complies 2 1 D ∧ Assumptions.ThetaRegistered D D M1 ∧ Assumptions.DarkReads M1 1 ∧
    Assumptions.PackedTrainingIncompressible D ch1.toPacked T1 2 1 (1 / 2) (1 / 2)

/-- **`PackedIncompressibleContainedMatMulOn` without `ThetaRegistered`**: `IncompressibleContainedMatMulOn_nontrivial`'s
instance, `ch1` packed. -/
def PackedIncompressibleContainedMatMulOn_nontrivial : Prop :=
  (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ 0 < 2 ∧ ((0 : ℕ) : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
    (∀ ℓ, 1 ≤ ch1.toPacked.d ℓ) ∧ (∀ ℓ, ch1.toPacked.e ℓ ≤ 1) ∧ Complies 2 0 Dθ ∧
    ¬ Assumptions.ThetaRegistered D Dθ Mθ ∧ Assumptions.DarkReads Mθ 1 ∧
    Assumptions.PackedTrainingIncompressible D ch1.toPacked T1 2 1 (1 / 2) (1 / 2) ∧
    ¬ (max (min ((1 + √1) ^ 2 * (1 / 2 * (1 / 2) * 1 / 1)) (√2 * (1 + √1) * √(1 - 1) * (1 / 2 * √(1 / 2 / 1))))
          (1 * (1 / 2) / max (1 : ℝ) (1 / (4 * (1 / 2) * 1))) *
        (univ.filter fun p : (ℓ : Fin ch1.toPacked.L) × Fin ch1.toPacked.N × Fin (ch1.toPacked.e ℓ) ×
            Fin (ch1.toPacked.d ℓ) => ch1.toPacked.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ Mθ.S).card ≤
      ((0 : ℕ) / (2 : ℕ) : ℝ) * Dθ.D.size Dθ.D.computed + (1 - 1) * ∑ v ∈ Mθ.dark, (novelReads D {v} : ℝ))

/-- **`PackedIncompressibleContainedTrainingOn`'s premises hold together**:
`PackedIncompressibleContainedMatMulOn_honest`'s instance at coverage `κ = 1`. -/
def PackedIncompressibleContainedTrainingOn_honest : Prop :=
  Complies 2 1 D ∧ Assumptions.ThetaRegistered D D M1 ∧ Assumptions.DarkReads M1 1 ∧
    Assumptions.PackedTrainingIncompressible D ch1.toPacked T1 2 1 (1 / 2) (1 / 2) ∧
    Assumptions.Coverage 1 D (2 * ch1.toPacked.G)

/-- **`PackedIncompressibleContainedTrainingOn` without `ThetaRegistered`**:
`IncompressibleContainedTrainingOn_nontrivial`'s instance, `ch1` packed. -/
def PackedIncompressibleContainedTrainingOn_nontrivial : Prop :=
  (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 2 ∧ 0 < 1 ∧ 0 < 2 ∧ ((0 : ℕ) : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
    (∀ ℓ, 1 ≤ ch1.toPacked.d ℓ) ∧ (∀ ℓ, ch1.toPacked.e ℓ ≤ 1) ∧ Complies 2 0 Dθ ∧
    ¬ Assumptions.ThetaRegistered D Dθ Mθ ∧ Assumptions.DarkReads Mθ 1 ∧
    Assumptions.PackedTrainingIncompressible D ch1.toPacked T1 2 1 (1 / 2) (1 / 2) ∧
    Assumptions.Coverage 1 D (2 * ch1.toPacked.G) ∧
    ¬ (max (min ((1 + √1) ^ 2 * (1 / 2 * (1 / 2) * 1 / 1)) (√2 * (1 + √1) * √(1 - 1) * (1 / 2 * √(1 / 2 / 1))))
          (1 * (1 / 2) / max (1 : ℝ) (1 / (4 * (1 / 2) * 1))) *
        (1 * D.D.size D.D.computed - 2 * (1 * Mθ.dark.card)) ≤
      2 * (((0 : ℕ) / (2 : ℕ) : ℝ) * Dθ.D.size Dθ.D.computed + (1 - 1) * ∑ v ∈ Mθ.dark, (novelReads D {v} : ℝ)))

end Nci.Guarantees
