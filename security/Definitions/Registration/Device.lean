module

public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
public import Mathlib.Order.Interval.Finset.Nat

/-! The device model the compute, bandwidth and capacity audits share: its limits, and schedules that charge each unit of a resource to one account. -/

@[expose] public section

namespace ResourceAccounting

/-- The op classes a step charges. -/
inductive OpClass
  /-- a tensor-core multiply-add -/
  | tc
  /-- a CUDA-core op -/
  | cuda
  /-- an oracle query: a call to a hash that a protocol models as a random oracle -/
  | oracle
  deriving DecidableEq

instance instFintypeOpClass : Fintype OpClass where
  elems := {.tc, .cuda, .oracle}
  complete o := by cases o <;> simp

/-- A device with side channels `ι`, by its limits. -/
structure Device (ι : Type) where
  /-- bits of on-chip state (S_chip) -/
  chip : ℕ
  /-- bits of device memory (M) -/
  mem : ℕ
  /-- ops of class `o` per tick (`rate .tc` is F) -/
  rate : OpClass → ℕ
  /-- bits from device memory to the chip per tick (BW) -/
  bw : ℕ
  /-- bits in over side channel `i` per tick -/
  cap : ι → ℕ

/-- One tick of a schedule, its charges split by account `α`. Each op is charged once, to the account that runs it,
not once per use of its result: PoUW's outputs are useful by design, so the residual counts ops, not information. -/
structure Step (ι α : Type) where
  ops : α → OpClass → ℕ
  /-- the bits account `a` read from device memory -/
  read : α → ℕ
  /-- the bits account `a` took in over side channel `i` -/
  side : α → ι → ℕ
  /-- the on-chip state account `a` holds at the end of the tick, in bits -/
  chip : α → ℕ
  /-- the device memory account `a` holds at the end of the tick, in bits -/
  mem : α → ℕ

/-- A schedule: the device's step at each tick. -/
abbrev Schedule (ι α : Type) : Type := ℕ → Step ι α

/-- A window of `len` ticks from tick `start`; `len` is its duration T. -/
structure Window where
  start : ℕ
  len : ℕ

/-- The ticks of a window. -/
def Window.ticks (w : Window) : Finset ℕ := Finset.Ico w.start (w.start + w.len)

/-- A price per op of each class: the cost unit a work certificate counts in (PoUW's counts RTX 4090 multiply-adds and
CUDA-core integer ops). -/
abbrev Price : Type := OpClass → ℕ

/-- The priced work the device can do in one tick (F when `p` prices a tensor-core multiply-add at 1 and nothing
else). -/
def Device.priced {ι : Type} (d : Device ι) (p : Price) : ℕ := ∑ o, p o * d.rate o

variable {ι α : Type} [Fintype α]

/-- A step fits the device: for every resource, the charges of all accounts together are within its limit. -/
def Step.Fits (d : Device ι) (st : Step ι α) : Prop :=
  (∀ o, ∑ a, st.ops a o ≤ d.rate o) ∧ ∑ a, st.read a ≤ d.bw ∧ (∀ i, ∑ a, st.side a i ≤ d.cap i) ∧
    ∑ a, st.chip a ≤ d.chip ∧ ∑ a, st.mem a ≤ d.mem

/-- A schedule fits the device: every step does. -/
def Schedule.Fits (d : Device ι) (sc : Schedule ι α) : Prop := ∀ t, (sc t).Fits d

namespace Schedule

variable (sc : Schedule ι α) (w : Window) (a : α)

/-- The ops of class `o` account `a` ran over window `w`. -/
def ops (o : OpClass) : ℕ := ∑ t ∈ w.ticks, (sc t).ops a o

/-- Account `a`'s work over window `w` at price `p`. -/
def work (p : Price) : ℕ := ∑ o, p o * sc.ops w a o

/-- The bits account `a` read from device memory over window `w`. -/
def read : ℕ := ∑ t ∈ w.ticks, (sc t).read a

def side (i : ι) : ℕ := ∑ t ∈ w.ticks, (sc t).side a i

/-- The device memory account `a` held over window `w`, in bit-ticks: the bits it holds at the end of each of the
window's ticks, summed. One bit held for one tick is one bit-tick, so `held / len` is its average in bits. -/
def held : ℕ := ∑ t ∈ w.ticks, (sc t).mem a

/-- The on-chip state account `a` held over window `w`, in bit-ticks. -/
def chipHeld : ℕ := ∑ t ∈ w.ticks, (sc t).chip a

end Schedule

end ResourceAccounting
