import Definitions.Core.Accel.Isa
import Mathlib.Algebra.Order.Field.Rat
import Mathlib.Algebra.Order.Ring.NNRat
import Mathlib.Data.Fintype.Defs

/-!
# A device's figures, and registration's check of them against the auditor's estimates (trusted)

**What is registered.** Each registered accelerator's leaf in the registration log carries the figures the developer
states for it (`DeviceFigures`): its SKU, its tensor clock f̂, and for each operand class `c` of a tensor-core MAC
(`OperandClass`: int8 operands drawn uniformly over a range of values) its peak rate R̂_c. Host CPUs carry no compute,
so only accelerators state figures. The rate is per operand class because a tensor-core MAC's true peak rate differs
with its operands' range.

**What is public.** The auditor's estimate of each SKU's figures and the tolerance registration allows around them
(`DeviceEstimate`, `FigureTolerance`) are public verifier parameters, not registered: the SKU exactly, and the clock and
each peak rate within one relative tolerance.

**The rule** (`DeviceFigures.Accepted`). Registration refuses a device unless its SKU is the estimate's, its clock and
every peak rate are within the tolerance of the estimate's in both directions, `(1 - τ)·e ≤ d ≤ (1 + τ)·e`
(`WithinTolerance`), and every peak rate it states is positive, so a statement that divides by a rate never goes
vacuous. Stating less capacity than estimated is refused too: it would make verified compute look a larger share of the
device's capacity.

**What the bound reads** (`DeviceFigures.bounding`). Figure by figure, the larger capacity of the stated figure and the
estimate's: the larger clock and the larger peak rate of each class (`DeviceFigures.CapacityLE` orders figures by
capacity).

Figures and tolerances are nonnegative rationals, so the check is exact. That a device runs within its figures (no
faster than its clock and its peak rates) is a premise of the hardware model, stated with its assumptions over these
definitions.
-/

namespace Verity

open scoped NNRat

/-- The operand class of a tensor-core MAC: int8 operands drawn uniformly over the range `[lo, hi]`. A MAC counts at its
class's rate only where the protocol fixes the operands' distribution; anywhere else it counts at the highest rate. -/
inductive OperandClass
  /-- uniform over `[-128, 127]`, every int8 value -/
  | uniformS8
  /-- uniform over `[-2, 1]` -/
  | uniformS2
  /-- uniform over `[0, 3]` -/
  | uniformU2
  /-- uniform over `[0, 15]` -/
  | uniformU4
  /-- the one value `0` -/
  | zero
  deriving DecidableEq

-- `deriving Fintype` fails on an enumeration at the pinned Mathlib.
instance instFintypeOperandClass : Fintype OperandClass :=
  ⟨⟨{.uniformS8, .uniformS2, .uniformU2, .uniformU4, .zero}, by decide⟩, fun c => by cases c <;> decide⟩

namespace OperandClass

/-- The least value of the class's range. -/
def lo : OperandClass → ℤ
  | .uniformS8 => -128
  | .uniformS2 => -2
  | .uniformU2 => 0
  | .uniformU4 => 0
  | .zero => 0

/-- The greatest value of the class's range. -/
def hi : OperandClass → ℤ
  | .uniformS8 => 127
  | .uniformS2 => 1
  | .uniformU2 => 3
  | .uniformU4 => 15
  | .zero => 0

/-- The tensor format of the class's operands. -/
def fmt : OperandClass → Accel.Fmt
  | .uniformS8 => .s8
  | .uniformS2 => .s8
  | .uniformU2 => .s8
  | .uniformU4 => .s8
  | .zero => .s8

end OperandClass

/-- A device's figures, which its leaf in the registration log carries. The SKU is a string because the registration
log names a card by its SKU and the check only compares it for equality. -/
structure DeviceFigures where
  /-- the SKU, as the registration log names the device's card -/
  sku : String
  /-- the tensor clock f̂, in Hz -/
  clock : ℚ≥0
  /-- the peak rate R̂_c of tensor-core MACs of operand class `c`, in MAC/s; the classes share one pipe -/
  peakRate : OperandClass → ℚ≥0

/-- The relative tolerance around an estimate's figures. -/
structure FigureTolerance where
  /-- the clock's and each peak rate's -/
  capacity : ℚ≥0

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
the clock and every peak rate are within the tolerance of the estimate's in both directions, and every peak rate is
positive. -/
def Accepted (est : DeviceEstimate) (d : DeviceFigures) : Prop :=
  d.sku = est.figures.sku ∧
    WithinTolerance est.tolerance.capacity est.figures.clock d.clock ∧
    (∀ c, WithinTolerance est.tolerance.capacity (est.figures.peakRate c) (d.peakRate c)) ∧
    ∀ c, 0 < d.peakRate c

instance instDecidableAccepted (est : DeviceEstimate) (d : DeviceFigures) : Decidable (d.Accepted est) := by
  unfold Accepted; infer_instance

/-- **The figures the bound reads** for a device that stated `d` against the estimate `est`: the larger clock and the
larger peak rate of each class, of the two. The SKU is the stated one. -/
def bounding (est : DeviceEstimate) (d : DeviceFigures) : DeviceFigures where
  sku := d.sku
  clock := max d.clock est.figures.clock
  peakRate c := max (d.peakRate c) (est.figures.peakRate c)

/-- `a`'s capacity is at most `b`'s: `b`'s clock and every peak rate are at least `a`'s. The SKU isn't compared. -/
def CapacityLE (a b : DeviceFigures) : Prop :=
  a.clock ≤ b.clock ∧ ∀ c, a.peakRate c ≤ b.peakRate c

end DeviceFigures

end Verity
