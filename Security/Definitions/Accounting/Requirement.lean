import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Definitions.Accounting.Device

/-!
# A declared circuit's requirement (trusted)

What a declared circuit `W` needs of a device's memory, read from its wiring and its schedule (the IR, never a kernel):
the values its execution keeps live when each step begins (`live`) and the values each step reads (`reads`), in bits,
in all, by block of the store and over a window; and when the audit's store and plans are `W`'s, block for block
(`StoreMatches`, `PlanMatches`), so that what the audit credits counts toward this requirement.

* **Wiring.** `W`'s values are its gates in the IR's canonical layout (`verity/ml/circuits/PROTOCOL.md` §6), each a
  `Value<w>` leaf of `width` bits. A value is registered (a `RegisteredInput`: the weights), a run input (an `Input`: a
  request's tokens) or a gate's output. A gate's operands are `ops` (`layout.Circuit.gate(i).operands`). The operands it
  uses when the circuit's values are `x` are `uses x`: all of `ops`, except that a gather uses its index and the one
  word it selects (an embedding's row), not its whole column.
* **Execution.** A scheduled execution is the values it produces (`ran`: the executed prefix, a request's rows up to
  its end and the engine's lag), the step at which it produces each (`step`: an engine step, a request's step `s` being
  its arrival step plus `s`), and the values (`value`, as bits). Registered values are there before step 0.
* **Live.** A value is live when step `t` begins if it was produced before `t` (or registered) and is an operand of a
  gate the execution runs at step `t` or later: the boundary values `out(S_<t) ∩ in(S_≥t)` of
  `verity/ml/circuits/boundary.py`, for `S_<t` the gates of the steps before `t`. A value no later gate reads is not
  live: `liveness.py`'s dead gates, and a step's outputs that leave the device. Liveness is structural (`ops`): a later
  gather may select any word of its column, so the whole column stays live.
* **Reads.** Step `t` reads the values produced before it that its gates use: `in(S_t)`, the boundary inputs of step
  `t`'s gates, less each gather's unselected words, so `reads t ⊆ live t`. A value produced and used within one step
  is neither: it is the step's working state, not memory the step reads or holds. A run input is used at the step it
  arrives, so it is neither either: it comes over a side channel (the property's `R_in`).
* **Blocks.** A layout places each value at a bit offset in a block of the store, by its store index `j`: registered
  (`j < N`, the weights) or appended by the memory map (`j ≥ N`, KV). Static and dynamic blocks are counted alike: a
  block is in `liveBlocks t` while it holds a live value, whatever wrote it.
* **Store.** The registered store with the memory map beside it (I3, `memory-map/v1` of
  `verity/core/protocols/remote_bandwidth/memory_map.py`): `N` registered blocks, the fresh indices each serving step
  wrote and those it freed, and each block's payload. A block is written once and complete, so its payload is a
  function of its index.

Bits, as `Device.lean` counts them: a value's bytes are its bits / 8, and the deployed store's block is `B = 32768` bits
(4,096 bytes). Steps are the engine's: which device ticks a step spans is the audit's timing, not this file's.
-/

namespace Accounting.Requirement

/-- A declared circuit's wiring over its values `G`. -/
structure Wiring (G : Type) where
  /-- a value's width in bits (`Value<w>`) -/
  width : G → ℕ
  /-- the registration's values (`RegisteredInput`): the weights -/
  registered : Finset G
  /-- a gate's operands, the values its Definition reads; none for an input -/
  ops : G → Finset G
  /-- the operands gate `g` uses when the circuit's values are `x`: `ops g`, except a gather's index and the word it
  selects -/
  uses : (G → ℕ) → G → Finset G

/-- A scheduled execution of a circuit over values `G`. -/
structure Execution (G : Type) where
  /-- the values it produces, the registered ones among them -/
  ran : Finset G
  /-- the engine step at which it produces each value -/
  step : G → ℕ
  /-- each value, as its bits -/
  value : G → ℕ

variable {G : Type} [DecidableEq G]

/-- An execution follows `W`'s wiring: it produces every registered value, and each gate it runs uses only its operands,
which it produced too and at no later step. Causality only: that the values are `W`'s is what the proofs of the one-stage
record show. -/
def Execution.Follows (W : Wiring G) (e : Execution G) : Prop :=
  W.registered ⊆ e.ran ∧
    ∀ g ∈ e.ran, W.uses e.value g ⊆ W.ops g ∧ ∀ v ∈ W.ops g, v ∈ e.ran ∧ (v ∈ W.registered ∨ e.step v ≤ e.step g)

section Values

variable (W : Wiring G) (e : Execution G)

/-- **The values live when step `t` begins**: produced before `t` (or registered), and an operand of a gate the execution
runs at step `t` or later. -/
def live (t : ℕ) : Finset G :=
  e.ran.filter fun v => (v ∈ W.registered ∨ e.step v < t) ∧ ∃ g ∈ e.ran, t ≤ e.step g ∧ v ∈ W.ops g

/-- **The values step `t` reads**: produced before `t` (or registered), and used by a gate the execution runs at
step `t`. -/
def reads (t : ℕ) : Finset G :=
  e.ran.filter fun v => (v ∈ W.registered ∨ e.step v < t) ∧ ∃ g ∈ e.ran, e.step g = t ∧ v ∈ W.uses e.value g

/-- The bits of a set of values. -/
def bits (S : Finset G) : ℕ := ∑ v ∈ S, W.width v

/-- The bits `W` keeps live when step `t` begins. -/
def liveBits (t : ℕ) : ℕ := bits W (live W e t)

/-- The bits step `t` reads. -/
def readBits (t : ℕ) : ℕ := bits W (reads W e t)

/-- The bits `W` holds over the steps of window `w`, in bit-steps: its live bits as each step begins, summed (as
`Schedule.held` counts bit-ticks). -/
def held (w : Window) : ℕ := ∑ t ∈ w.ticks, liveBits W e t

/-- The bits `W` reads over the steps of window `w` (as `Schedule.read` counts them). -/
def read (w : Window) : ℕ := ∑ t ∈ w.ticks, readBits W e t

end Values

/-- A layout of values in the store: each value's block, a store index, and its bit offset in the block. -/
structure Layout (G : Type) where
  /-- the store index of the value's block -/
  block : G → ℕ
  /-- the value's first bit in its block -/
  offset : G → ℕ

section Blocks

variable (W : Wiring G) (e : Execution G) (ℓ : Layout G)

/-- The blocks that hold values of `S`. -/
def blocks (S : Finset G) : Finset ℕ := S.image ℓ.block

/-- The bits of the values of `S` that lie in the blocks `H`. -/
def bitsAt (S : Finset G) (H : Finset ℕ) : ℕ := bits W (S.filter fun v => ℓ.block v ∈ H)

/-- The bits of the values of `S` in block `j`. -/
def bitsIn (S : Finset G) (j : ℕ) : ℕ := bits W (S.filter fun v => ℓ.block v = j)

/-- **`W`'s live blocks when step `t` begins**: the blocks that hold a live value, registered and appended alike. -/
def liveBlocks (t : ℕ) : Finset ℕ := blocks ℓ (live W e t)

/-- The bits `W` keeps live in block `j` when step `t` begins. -/
def liveIn (t j : ℕ) : ℕ := bitsIn W ℓ (live W e t) j

/-- **The blocks step `t` reads**: those that hold a value it reads. -/
def readBlocks (t : ℕ) : Finset ℕ := blocks ℓ (reads W e t)

/-- The bits step `t` reads from block `j`. -/
def readIn (t j : ℕ) : ℕ := bitsIn W ℓ (reads W e t) j

/-- The bits `W` reads from block `j` over the steps of window `w`. -/
def readOf (w : Window) (j : ℕ) : ℕ := ∑ t ∈ w.ticks, readIn W e ℓ t j

/-- `ℓ` packs the values `S` into blocks of `B` bits: each lies within its block, and two in one block don't overlap. -/
def Packs (B : ℕ) (S : Finset G) : Prop :=
  ∀ u ∈ S, ℓ.offset u + W.width u ≤ B ∧
    ∀ v ∈ S, u ≠ v → ℓ.block u = ℓ.block v →
      ℓ.offset u + W.width u ≤ ℓ.offset v ∨ ℓ.offset v + W.width v ≤ ℓ.offset u

end Blocks

/-- The audit's store: the registered store and the memory map beside it (I3). -/
structure Store where
  /-- the registered store's blocks `N`, store indices `0 … N - 1` -/
  registered : ℕ
  /-- the fresh store indices serving step `t` wrote (memory-map/v1's `written`) -/
  written : ℕ → Finset ℕ
  /-- the store indices serving step `t` freed (memory-map/v1's `freed`, by the indices their slots held) -/
  freed : ℕ → Finset ℕ
  /-- each block's payload, as its bits -/
  payload : ℕ → ℕ

/-- **The blocks resident when step `t` begins**: the registered ones and those written at steps before `t`, less those
freed at steps before `t`. -/
def Store.resident (s : Store) (t : ℕ) : Finset ℕ :=
  (Finset.range s.registered ∪ (Finset.range t).biUnion s.written) \ (Finset.range t).biUnion s.freed

section Link

variable (W : Wiring G) (e : Execution G) (ℓ : Layout G)

/-- Each value of `S` is in its block's payload: the payload's bits from the value's offset are the value. -/
def Carries (s : Store) (S : Finset G) : Prop :=
  ∀ v ∈ S, s.payload (ℓ.block v) / 2 ^ ℓ.offset v % 2 ^ W.width v = e.value v

/-- **The store is `W`'s when step `t` begins**: its resident blocks are `W`'s live blocks, block for block (no block
`W` doesn't keep live is resident, and every block holding a live value is), the layout packs the live values into
blocks of `B` bits, and each live value is in its block's payload. -/
def StoreMatches (s : Store) (B t : ℕ) : Prop :=
  s.resident t = liveBlocks W e ℓ t ∧ Packs W ℓ B (live W e t) ∧ Carries W e ℓ s (live W e t)

/-- **Step `t`'s plan is `W`'s reads**: the blocks the audit's pass bound to step `t` reads are the blocks `W` reads at
step `t` (each receipted block is one `W` reads then, and every block `W` reads then is receipted), and the layout packs
the values read into blocks of `B` bits. -/
def PlanMatches (plan : Finset ℕ) (B t : ℕ) : Prop :=
  plan = readBlocks W e ℓ t ∧ Packs W ℓ B (reads W e t)

end Link

end Accounting.Requirement
