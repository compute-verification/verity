import Mathlib.Algebra.Order.Field.Rat
import Mathlib.Algebra.Order.Ring.NNRat
import Mathlib.Tactic.DeriveFintype

/-!
# A device's figures, and registration's check of them against the auditor's estimates (trusted)

**What is registered.** Each registered device's leaf in the registration log carries the figures the developer states
for it (`DeviceFigures`): its SKU, its tensor clock f̂, and for each operand class `c` of a tensor-core MAC
(`OperandClass`: random, 2-bit and all-zero operands) its peak rate R̂_c and its energy floor ε̂_c, the least energy one
such MAC costs, and its power cap P̂. The floor is per operand class because zero operands cost less than random ones
and the prover picks its own encoding.

**What is public.** The auditor's estimate of each SKU's figures and the tolerance registration allows around them
(`DeviceEstimate`, `FigureTolerance`) are public verifier parameters, not registered: the SKU exactly, the clock, each
peak rate and the power cap within one relative tolerance, and each energy floor within another.

**The rule** (`DeviceFigures.Accepted`). Registration refuses a device unless its SKU is the estimate's and every figure
is within its tolerance of the estimate's in both directions, `(1 - τ)·e ≤ d ≤ (1 + τ)·e` (`WithinTolerance`). Stating
less capacity than estimated is refused too: it would make verified compute look a larger share of the device's
capacity.

**What the bound reads** (`DeviceFigures.bounding`). Figure by figure, the larger capacity of the stated figure and the
estimate's: the larger clock, peak rates and power cap, and the smaller energy floor, since a lower floor is more
capacity (`DeviceFigures.CapacityLE` orders figures by capacity).

Figures and tolerances are nonnegative rationals, so the check is exact. That a device runs within its figures (no
faster than its clock and peak rates, at least its floor per MAC, at most its cap) is a premise of the hardware model,
stated with its assumptions over these definitions.
-/

namespace Verity

open scoped NNRat

/-- The operand class of a tensor-core MAC: what its operands hold, which bounds below the energy the MAC costs. -/
inductive OperandClass
  /-- random operands -/
  | random
  /-- 2-bit operands: each operand takes one of four values -/
  | twoBit
  /-- all-zero operands -/
  | zero
  deriving DecidableEq, Fintype

/-- A device's figures, which its leaf in the registration log carries. The SKU is a string because the registration
log names a card by its SKU and the check only compares it for equality. -/
structure DeviceFigures where
  /-- the SKU, as the registration log names the device's card -/
  sku : String
  /-- the tensor clock f̂, in Hz -/
  clock : ℚ≥0
  /-- the peak rate R̂_c of tensor-core MACs of operand class `c`, in MAC/s; the classes share one pipe -/
  peakRate : OperandClass → ℚ≥0
  /-- the energy floor ε̂_c, the least energy one tensor-core MAC of operand class `c` costs, in pJ per MAC -/
  energyFloor : OperandClass → ℚ≥0
  /-- the power cap P̂, the most power the device draws, in W -/
  powerCap : ℚ≥0

/-- Relative tolerances around an estimate's figures. -/
structure FigureTolerance where
  /-- the clock's, each peak rate's and the power cap's -/
  capacity : ℚ≥0
  /-- each energy floor's -/
  floor : ℚ≥0

/-- The auditor's estimate of one SKU's figures, with the tolerance registration allows around them: public verifier
parameters, not registered. -/
structure DeviceEstimate where
  /-- the estimated figures, the SKU among them -/
  figures : DeviceFigures
  /-- the tolerance around them -/
  tolerance : FigureTolerance

/-- `x` is within relative tolerance `τ` of `e`, in both directions: `(1 - τ)·e ≤ x ≤ (1 + τ)·e`. `1 - τ` truncates at
0, so a tolerance of 1 or more sets no lower bound. -/
def WithinTolerance (τ e x : ℚ≥0) : Prop := (1 - τ) * e ≤ x ∧ x ≤ (1 + τ) * e

instance instDecidableWithinTolerance (τ e x : ℚ≥0) : Decidable (WithinTolerance τ e x) := by
  unfold WithinTolerance; infer_instance

namespace DeviceFigures

/-- **Registration's check of a device's figures** `d` against the auditor's estimate `est`: the SKU is the estimate's,
and every figure is within its tolerance of the estimate's, in both directions. -/
def Accepted (est : DeviceEstimate) (d : DeviceFigures) : Prop :=
  d.sku = est.figures.sku ∧
    WithinTolerance est.tolerance.capacity est.figures.clock d.clock ∧
    (∀ c, WithinTolerance est.tolerance.capacity (est.figures.peakRate c) (d.peakRate c)) ∧
    (∀ c, WithinTolerance est.tolerance.floor (est.figures.energyFloor c) (d.energyFloor c)) ∧
    WithinTolerance est.tolerance.capacity est.figures.powerCap d.powerCap

instance instDecidableAccepted (est : DeviceEstimate) (d : DeviceFigures) : Decidable (d.Accepted est) := by
  unfold Accepted; infer_instance

/-- **The figures the bound reads** for a device that stated `d` against the estimate `est`: the larger clock, peak
rates and power cap of the two, and the smaller energy floor. The SKU is the stated one. -/
def bounding (est : DeviceEstimate) (d : DeviceFigures) : DeviceFigures where
  sku := d.sku
  clock := max d.clock est.figures.clock
  peakRate c := max (d.peakRate c) (est.figures.peakRate c)
  energyFloor c := min (d.energyFloor c) (est.figures.energyFloor c)
  powerCap := max d.powerCap est.figures.powerCap

/-- `a`'s capacity is at most `b`'s, figure by figure: `b`'s clock, peak rates and power cap are at least `a`'s, and
its energy floors at most `a`'s. The SKU isn't compared. -/
def CapacityLE (a b : DeviceFigures) : Prop :=
  a.clock ≤ b.clock ∧ (∀ c, a.peakRate c ≤ b.peakRate c) ∧ (∀ c, b.energyFloor c ≤ a.energyFloor c) ∧
    a.powerCap ≤ b.powerCap

end DeviceFigures

end Verity
