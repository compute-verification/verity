import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Definitions.Accounting.Device
import Definitions.Core.Requirement

/-!
# A declared circuit's requirement (trusted)

What a declared circuit `W` needs of a device's memory, read from its wiring and its schedule (the IR, never a kernel):
the values its execution keeps live when each step begins (`live`) and the values each step reads (`reads`), in bits,
in all, by block of the store and over a window; the live values in complete blocks, which the audit credits
(`credited`); and when the audit's store and plans hold `W`'s credited blocks (`StoreMatches`, `PlanMatches`), so that
what the audit credits counts toward this requirement. Placed on devices
(`Placement`), the execution's gates are also what PoUW credits and the bits they read from other devices
(`region`, `ingress`, `work`; `Definitions/Accounting/Requirement/Placed.lean` makes it an actual run).

A step is always an **engine step**: the engine's `s`-th step (a request's step `s` being its arrival step plus `s`),
never a device tick. The ticks a step spans on each device are the placement's (`Placement.ticks`), and a window
(`Window`) here is a run of engine steps (`Window.steps`), never of ticks.

* **Wiring.** `W`'s values are its gates in the IR's canonical layout (`verity/ml/circuits/PROTOCOL.md` §6), each a
  `Value<w>` leaf of `width` bits. A value is registered (a `RegisteredInput`: the weights), a run input (an `Input` not
  bound by registration: a request's tokens, `inputs`) or a gate's output. A gate's operands are `ops` (`layout.Circuit.gate(i).operands`). The operands it
  uses when the circuit's values are `x` are `uses x`: all of `ops`, except that a gather uses its index and the one
  word it selects (an embedding's row), not its whole column.
* **Execution.** A scheduled execution is the values it produces (`ran`: the executed prefix, a request's rows up to
  its end and the engine's lag), the engine step at which it produces each (`step`), and the values (`value`, as bits).
  Registered values are there before step 0.
* **Placement.** A placed execution also says at which device each value is produced or, an input, stored (`leaf`: a
  leaf of the actual circuit's tree of locations, `Accounting.Placed`, the device, never an SM, an SMSP or a unit), and
  at which tick each device begins each engine step (`ticks`). A value is placed at the tick its device begins its step
  (`Placement.tick`): within a step the schedule has no order, so a window of ticks is bounded by the whole steps
  (`stepRegion`) that begin inside it on every device of a location, and by those whose span covers it. A run input is
  an input gate wherever it is placed: it arrives over a side channel, and is no device's work.
* **Live.** A value is live when step `s` begins if it was produced before `s` (or registered) and is an operand of a
  gate the execution runs at step `s` or later: the boundary values `out(S_<s) ∩ in(S_≥s)` of
  `verity/ml/circuits/boundary.py`, for `S_<s` the gates of the steps before `s`. A value no later gate reads is not
  live: `liveness.py`'s dead gates, and a step's outputs that leave the device. Liveness is structural (`ops`): a later
  gather may select any word of its column, so the whole column stays live.
* **Reads.** Step `s` reads the values produced before it that its gates use: `in(S_s)`, the boundary inputs of step
  `s`'s gates, less each gather's unselected words, so `reads s ⊆ live s`. A value produced and used within one step
  is neither: it is the step's working state, not memory the step reads or holds. A run input is used at the step it
  arrives, so it is neither either: it comes over a side channel (the property's `R_in`).
* **Blocks.** A store placement (`Store.Placement`) puts each value at a bit offset in a block of the store, by its
  store index `j`: registered (`j < N`, the weights) or appended by the memory map (`j ≥ N`, KV). Static and dynamic
  blocks are counted alike: a block is in `liveBlocks s` while it holds a live value, whatever wrote it.
* **Store.** The registered store with the memory map beside it (I3, `memory-map/v1` of
  `verity/core/protocols/remote_bandwidth/memory_map.py`): `N` registered blocks, the fresh indices each serving step
  wrote and those it freed, and each block's payload. A block is written once, complete and encoded, so its payload is
  a function of its index. Step `s`'s record takes effect after step `s`'s reads: a block is written in the record of
  the step before its first encoded read (deferred residency: the step that completes it when its append's coin is back
  before the next step's reads, a later one when not), readable from the step after, and freed in the record of a step
  no earlier than its values' last structural reader.
* **Credited.** A block is complete when step `s` begins once every value the store placement puts in it was produced
  before `s` and the device has encoded it under its append's coin, `s` being at or after its first encoded read
  (`Store.Placement.encoded`, `incomplete`); a block of registered values is complete from the start. An incomplete
  block is in no record, so a live value in one (a sequence's KV page that later steps fill, or a filled page whose coin
  is still out) is in no resident block, and the audit credits only the live values in complete blocks (`credited`). A
  block's values are credited from its first encoded read to their last structural reader (`live`), so a block whose
  values' last structural reader comes before its first encoded read is credited at no step, and memory-map/v1
  withdraws it. The link is one-sided: every block holding a credited value is resident (`StoreMatches`), and every
  complete block a step reads is in its pass's plan (`PlanMatches`), but other blocks may be resident or planned too
  (filler, dead or another workload's), and they hold none of `W`'s credited values. The share left out is a block per
  sequence, layer and plane when the store placement fills each one's pages in turn, and one more while the page it
  completed last waits for its coin (`OneBlockEach`, the page's parity in the group).

The execution, its placement and the store (`Execution`, `Placement`, `Store`) are Core's
(`Definitions/Core/Requirement.lean`), stated there as the bullets above describe them.

Bits, as `Device.lean` counts them: a value's bytes are its bits / 8, and the deployed store's block is `b = 32768` bits
(4,096 bytes).
-/

namespace Accounting

/-- **The engine steps of window `w`**: a window of the requirement is a run of engine steps, never of a device's ticks
(`Requirement.Placement.ticks`). -/
def Window.steps (w : Window) : Finset ℕ := Finset.Ico w.start (w.start + w.len)

end Accounting

namespace Accounting.Requirement

/-- A declared circuit's wiring over its values `G`. -/
structure Wiring (G : Type) where
  /-- a value's width in bits (`Value<w>`) -/
  width : G → ℕ
  /-- the registration's values (`RegisteredInput`): the weights -/
  registered : Finset G
  /-- the run's inputs (an `Input` not bound by registration): a request's tokens, input gates wherever they're placed -/
  inputs : Finset G
  /-- a gate's operands, the values its Definition reads; none for an input -/
  ops : G → Finset G
  /-- the operands gate `g` uses when the circuit's values are `x`: `ops g`, except a gather's index and the word it
  selects -/
  uses : (G → ℕ) → G → Finset G

/-- **`W`'s wiring is topological**: a gate's operands come before it (the IR's canonical order, §6). -/
def Wiring.Topological {G : Type} [LT G] (W : Wiring G) : Prop := ∀ g, ∀ v ∈ W.ops g, v < g

variable {G : Type} [DecidableEq G]

/-- An execution follows `W`'s wiring: it produces every registered value, and each gate it runs uses only its operands,
which it produced too and at no later step. Causality only: that the values are `W`'s is what the proofs of the one-stage
record show. -/
def Execution.Follows (W : Wiring G) (e : Execution G) : Prop :=
  W.registered ⊆ e.ran ∧
    ∀ g ∈ e.ran, W.uses e.value g ⊆ W.ops g ∧ ∀ v ∈ W.ops g, v ∈ e.ran ∧ (v ∈ W.registered ∨ e.step v ≤ e.step g)

section Values

variable (W : Wiring G) (e : Execution G)

/-- **The values live when step `s` begins**: produced before `s` (or registered), and an operand of a gate the execution
runs at step `s` or later. -/
def live (s : ℕ) : Finset G :=
  e.ran.filter fun v => (v ∈ W.registered ∨ e.step v < s) ∧ ∃ g ∈ e.ran, s ≤ e.step g ∧ v ∈ W.ops g

/-- **The values step `s` reads**: produced before `s` (or registered), and used by a gate the execution runs at
step `s`. -/
def reads (s : ℕ) : Finset G :=
  e.ran.filter fun v => (v ∈ W.registered ∨ e.step v < s) ∧ ∃ g ∈ e.ran, e.step g = s ∧ v ∈ W.uses e.value g

/-- The bits of a set of values. -/
def bits (S : Finset G) : ℕ := ∑ v ∈ S, W.width v

/-- The bits `W` keeps live when step `s` begins. -/
def liveBits (s : ℕ) : ℕ := bits W (live W e s)

/-- The bits step `s` reads. -/
def readBits (s : ℕ) : ℕ := bits W (reads W e s)

/-- The bits `W` holds over the engine steps of window `w`, in bit-steps: its live bits as each step begins, summed (as
`Schedule.held` counts bit-ticks). -/
def held (w : Window) : ℕ := ∑ s ∈ w.steps, liveBits W e s

/-- The bits `W` reads over the engine steps of window `w` (as `Schedule.read` counts them). -/
def read (w : Window) : ℕ := ∑ s ∈ w.steps, readBits W e s

end Values

section Placed

variable {L : Type} (W : Wiring G) (p : Placement L G)

open Classical in
/-- **`W`'s region** `W[X, I]` (`Accounting.Placed.region`): the gates the execution runs on the devices `X` at the
ticks `I`, the values it produces but the registered ones and the run's inputs. -/
noncomputable def region (X : Finset L) (I : Set ℕ) : Finset G :=
  p.ran.filter fun g => g ∉ W.registered ∧ g ∉ W.inputs ∧ p.leaf g ∈ X ∧ p.tick g ∈ I

open Classical in
/-- **`W`'s gates on `X` over the engine steps of window `w`**: those the execution runs on the devices `X` at a step of
`w`, the registered values and the run's inputs aside. -/
noncomputable def stepRegion (X : Finset L) (w : Window) : Finset G :=
  p.ran.filter fun g => g ∉ W.registered ∧ g ∉ W.inputs ∧ p.leaf g ∈ X ∧ p.step g ∈ w.steps

/-- `in(S)` (`Verity.Circuit.into`): the values outside `S` that a gate of `S` uses. -/
def into (S : Finset G) : Finset G := S.biUnion (W.uses p.value) \ S

open Classical in
/-- **`W`'s ingress** `in_X(I)` (`Accounting.Placed.ingress`): the bits of the values placed outside `X` that `W[X, I]`
uses. -/
noncomputable def ingress (X : Finset L) (I : Set ℕ) : ℕ :=
  bits W ((into W p (region W p X I)).filter fun a => p.leaf a ∉ X)

/-- **Work** `π(R)` (`Accounting.Placed.work`): the total price of the gates of `R` at a pricing `π`, a gate's credit
(the target multiply-adds its class stands for). -/
def work (π : G → ℕ) (R : Finset G) : ℕ := ∑ g ∈ R, π g

end Placed

/-- A placement of values in the store: each value's block, a store index, its bit offset in the block, and each block's
first encoded read. -/
structure Store.Placement (G : Type) where
  /-- the store index of the value's block -/
  block : G → ℕ
  /-- the value's first bit in its block -/
  offset : G → ℕ
  /-- the engine step of block `j`'s first encoded read: the first step whose reads see it encoded under its append's
  coin, one after the record that writes it, or would have for a block withdrawn first (memory-map/v1's deferred
  residency). One past the step that completes it when the coin is back before the next step's reads; a block is
  complete no earlier than every value in it is produced, whatever this says, and a block of registered values from
  the start -/
  encoded : ℕ → ℕ

section Blocks

variable (W : Wiring G) (e : Execution G) (ℓ : Store.Placement G)

/-- The blocks that hold values of `S`. -/
def blocks (S : Finset G) : Finset ℕ := S.image ℓ.block

/-- The bits of the values of `S` that lie in the blocks `H`. -/
def bitsAt (S : Finset G) (H : Finset ℕ) : ℕ := bits W (S.filter fun v => ℓ.block v ∈ H)

/-- The bits of the values of `S` in block `j`. -/
def bitsIn (S : Finset G) (j : ℕ) : ℕ := bits W (S.filter fun v => ℓ.block v = j)

/-- **`W`'s live blocks when step `s` begins**: the blocks that hold a live value, registered and appended alike. -/
def liveBlocks (s : ℕ) : Finset ℕ := blocks ℓ (live W e s)

/-- The bits `W` keeps live in block `j` when step `s` begins. -/
def liveIn (s j : ℕ) : ℕ := bitsIn W ℓ (live W e s) j

/-- **The blocks step `s` reads**: those that hold a value it reads. -/
def readBlocks (s : ℕ) : Finset ℕ := blocks ℓ (reads W e s)

/-- The bits step `s` reads from block `j`. -/
def readIn (s j : ℕ) : ℕ := bitsIn W ℓ (reads W e s) j

/-- The bits `W` reads from block `j` over the engine steps of window `w`. -/
def readOf (w : Window) (j : ℕ) : ℕ := ∑ s ∈ w.steps, readIn W e ℓ s j

/-- `ℓ` packs the values `S` into blocks of `b` bits: each lies within its block, and two in one block don't overlap. -/
def Packs (b : ℕ) (S : Finset G) : Prop :=
  ∀ u ∈ S, ℓ.offset u + W.width u ≤ b ∧
    ∀ v ∈ S, u ≠ v → ℓ.block u = ℓ.block v →
      ℓ.offset u + W.width u ≤ ℓ.offset v ∨ ℓ.offset v + W.width v ≤ ℓ.offset u

/-- **One block per group**: the values of `S` in each group `k` of `g` lie in one block at most. With `g` a KV value's
sequence, layer, plane (K or V) and its page's parity, and `S` the live values in incomplete blocks, it is a KV pool's
page structure under deferred residency: a sequence fills its pages of each layer and plane one at a time, and each
page's coin is back before the next page is complete, so only the block it is filling and the one it completed last,
while that one's encode is out, are incomplete. -/
def OneBlockEach {K : Type} [DecidableEq K] (g : G → K) (S : Finset G) : Prop :=
  ∀ k, ((S.filter fun v => g v = k).image ℓ.block).card ≤ 1

end Blocks

section Credited

variable (W : Wiring G) (e : Execution G) (ℓ : Store.Placement G)

/-- **The blocks incomplete when step `s` begins**: those the store placement puts a value in that the execution
produces at step `s` or later, or a value it produces and whose first encoded read (`Store.Placement.encoded`) is after
`s`. A block is complete once every value in it is produced and the device has encoded it under its append's coin, so a
block of registered values is complete from the start, and memory-map/v1 writes a block in the record of the step
before its first encoded read (deferred residency): until then it is in no record, like a block still being filled. -/
def incomplete (s : ℕ) : Finset ℕ :=
  (e.ran.filter fun v => v ∉ W.registered ∧ (s ≤ e.step v ∨ s < ℓ.encoded (ℓ.block v))).image ℓ.block

/-- **The live values the audit credits when step `s` begins**: those in complete blocks. A live value in a block that
later steps fill, or whose encode hasn't landed, is in no record yet, so no audit credits it: a block's values are
credited from its first encoded read to their last structural reader (`live`). -/
def credited (s : ℕ) : Finset G := (live W e s).filter fun v => ℓ.block v ∉ incomplete W e ℓ s

/-- The values step `s` reads from complete blocks. -/
def creditedReads (s : ℕ) : Finset G := (reads W e s).filter fun v => ℓ.block v ∉ incomplete W e ℓ s

/-- The live bits the audit credits when step `s` begins. -/
def creditedBits (s : ℕ) : ℕ := bits W (credited W e ℓ s)

/-- The bits step `s` reads from complete blocks. -/
def creditedReadBits (s : ℕ) : ℕ := bits W (creditedReads W e ℓ s)

end Credited

section Link

variable (W : Wiring G) (e : Execution G) (ℓ : Store.Placement G)

/-- Each value of `S` is in its block's payload: the payload's bits from the value's offset are the value. -/
def Carries (σ : Store) (S : Finset G) : Prop :=
  ∀ v ∈ S, σ.payload (ℓ.block v) / 2 ^ ℓ.offset v % 2 ^ W.width v = e.value v

/-- **The store holds `W`'s credited values when step `s` begins**: every block holding a live value the audit credits
is resident, the store placement packs the live values into blocks of `b` bits, and each credited value is in its
block's payload. Other blocks may be resident too, and hold none of `W`'s credited values: the registered store's filler
(a KV pool's pages, resident until a write first reaches them), a block whose values are all dead, or another
workload's. A live value in an incomplete block is in no record, so no audit credits it. Liveness is structural (`ops`),
so a block holding a credited value stays resident through its values' last structural reader, a later gather that may
select it included, and a free at their last use before that breaks the link. -/
def StoreMatches (σ : Store) (b s : ℕ) : Prop :=
  blocks ℓ (credited W e ℓ s) ⊆ σ.resident s ∧ Packs W ℓ b (live W e s) ∧ Carries W e ℓ σ (credited W e ℓ s)

/-- **Step `s`'s plan covers `W`'s reads**: the blocks the audit's pass bound to step `s` reads include every complete
block `W` reads at step `s`, and the store placement packs the values read into blocks of `b` bits. The plan may hold
other blocks too (a deployed plan is topped up with filler reads), and those hold none of `W`'s reads. -/
def PlanMatches (plan : Finset ℕ) (b s : ℕ) : Prop :=
  blocks ℓ (creditedReads W e ℓ s) ⊆ plan ∧ Packs W ℓ b (reads W e s)

end Link

end Accounting.Requirement
