module

public import Definitions.Core.Accel.Isa
public import Mathlib.Algebra.Order.Field.Rat
public import Mathlib.Algebra.Order.Ring.NNRat
public import Mathlib.Data.Fintype.Defs
public import Mathlib.Data.NNRat.Defs
import Mathlib.Data.Finset.Attr

/-! A device's registered figures and registration's check of them against the auditor's public estimate
(`DeviceFigures.Accepted`). -/

@[expose] public section

namespace Verity

open scoped NNRat

/-- The operand class of a tensor-core MAC: int8 operands sampled uniformly over the range `[lo, hi]`. A MAC counts at its
class's rate only where the protocol fixes the operands' distribution; anywhere else it counts at the highest rate. -/
inductive OperandClass
  | uniformS8
  | uniformS2
  | uniformU2
  | uniformU4
  | zero
  deriving DecidableEq

-- `deriving Fintype` fails on an enumeration at the pinned Mathlib.
instance instFintypeOperandClass : Fintype OperandClass :=
  ⟨⟨{.uniformS8, .uniformS2, .uniformU2, .uniformU4, .zero}, by decide⟩, fun c => by cases c <;> decide⟩

namespace OperandClass

def lo : OperandClass → ℤ
  | .uniformS8 => -128
  | .uniformS2 => -2
  | .uniformU2 => 0
  | .uniformU4 => 0
  | .zero => 0

def hi : OperandClass → ℤ
  | .uniformS8 => 127
  | .uniformS2 => 1
  | .uniformU2 => 3
  | .uniformU4 => 15
  | .zero => 0

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
  figures : DeviceFigures
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
