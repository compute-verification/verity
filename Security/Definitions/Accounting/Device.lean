import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Order.Interval.Finset.Nat

/-!
# The device model (trusted)

The one device that the compute, bandwidth and capacity audits are all about: what it holds, what flows into it, what
it computes, and when.

* **State.** On-chip state of at most `chip` bits (S_chip: registers, shared memory, caches) and device memory of at
  most `mem` bits (M).
* **Ingress.** Bits reach the chip from device memory at most `bw` per tick (BW), and over each side channel `i : ι` at
  most `cap i` per tick: PCIe from NVMe and the network (there is no host RAM, Daniel's ruling of 3 Oct).
* **Ops.** Each op class (`Op`) runs at most `rate o` times per tick: tensor-core multiply-adds (`rate .tc` is F),
  CUDA-core ops and oracle queries.
* **Time.** A tick is the device's unit of time, and a window is `len` consecutive ticks from `start`, so its duration
  T is `len`.

A schedule is the device's step at each tick. A step charges every resource it uses to an account (`α`): the ops of
each class, the bits read from device memory and over each side channel, and the on-chip state and device memory held
at the end of the tick. A step fits the device (`Step.Fits`) when, for every resource, the charges of all accounts
together are within the device's limit. So each unit of a resource (one tensor-core cycle, one bit read, one bit held
for a tick) is charged once, to one account, and no two accounts can both claim it. A window's totals (`Schedule.ops`,
`Schedule.read`, `Schedule.held`, ...) sum an account's charges over the window's ticks. Memory held is counted in
bit-ticks: one bit held for one tick is one bit-tick, so `Schedule.held` over a window of `len` ticks is `len` times the
account's average in bits.
-/

namespace Accounting

/-- The op classes a step charges. -/
inductive Op
  /-- a tensor-core multiply-add -/
  | tc
  /-- a CUDA-core op -/
  | cuda
  /-- an oracle query: a call to a hash that a protocol models as a random oracle -/
  | oracle
  deriving DecidableEq

instance instFintypeOp : Fintype Op where
  elems := {.tc, .cuda, .oracle}
  complete o := by cases o <;> simp

/-- A device with side channels `ι`, by its limits. -/
structure Device (ι : Type) where
  /-- bits of on-chip state (S_chip) -/
  chip : ℕ
  /-- bits of device memory (M) -/
  mem : ℕ
  /-- ops of class `o` per tick (`rate .tc` is F) -/
  rate : Op → ℕ
  /-- bits from device memory to the chip per tick (BW) -/
  bw : ℕ
  /-- bits in over side channel `i` per tick -/
  cap : ι → ℕ

/-- One tick of a schedule, its charges split by account `α`. Each op is charged once, to the account that runs it,
not once per use of its result: PoUW's outputs are useful by design, so the residual counts ops, not information. -/
structure Step (ι α : Type) where
  /-- the ops of class `o` that account `a` ran -/
  ops : α → Op → ℕ
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
abbrev Price : Type := Op → ℕ

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
def ops (o : Op) : ℕ := ∑ t ∈ w.ticks, (sc t).ops a o

/-- Account `a`'s work over window `w` at price `p`. -/
def work (p : Price) : ℕ := ∑ o, p o * sc.ops w a o

/-- The bits account `a` read from device memory over window `w`. -/
def read : ℕ := ∑ t ∈ w.ticks, (sc t).read a

/-- The bits account `a` took in over side channel `i` over window `w`. -/
def side (i : ι) : ℕ := ∑ t ∈ w.ticks, (sc t).side a i

/-- The device memory account `a` held over window `w`, in bit-ticks: the bits it holds at the end of each of the
window's ticks, summed. One bit held for one tick is one bit-tick, so `held / len` is its average in bits. -/
def held : ℕ := ∑ t ∈ w.ticks, (sc t).mem a

/-- The on-chip state account `a` held over window `w`, in bit-ticks. -/
def chipHeld : ℕ := ∑ t ∈ w.ticks, (sc t).chip a

end Schedule

end Accounting
