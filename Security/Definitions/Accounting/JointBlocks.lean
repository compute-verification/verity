import Definitions.Accounting.JointPeriod

/-!
# Server-time blocks audited by every protocol, and the wipes between them (trusted)

A server's life is cut into server-time blocks, each opened by a certified wipe of its device memory. Over a horizon of
`B` blocks on one device, one circuit and one coin space (`BlocksRun`):

* **Block `b`** is a period of the node's run (`Accounting.PeriodRun`) decided by the five audits, sampled proofs, PoUW,
  PoUS, PoUB and the network certifier, each on its own coordinate of the coins. Every block's schedule and committed
  transcript are functions of all the coins, so a block may adapt to the coins of the blocks before it.
* **The wipe that opens block `b`** (`BlockWipe`): the memory challenger fills `fill` bits of the device memory with
  labels derived from a fresh seed and runs timed spot reads. Its verdict, and `kept`, the bits of foreign content
  block `b` can read back from what the wipe leaves beyond the committed carried state, are functions of the coins. Its
  survivor bound at a fraction `f` (`BlockWipe.Survives`) is PoUS's secure-erasure requirement's
  (`Pous.EraseMeets`, which `Pous.Guarantees.SecureErasureMeetsFamily` proves at `f = 1/19`, `ε = 2^-128`): at most
  `f · fill + 2λ` bits (`λ = 128`) survive, plus the state the fill leaves out (`BlockWipe.unfilled`, the on-chip
  state and the device memory the fill doesn't reach).
* **The memory term of block `b`** (`BlocksRun.MemoryTerm`), PoUS over the weights alone at a share of `0.07` (Daniel's
  ruling, ASK-132): at each of the block's PoUS challenge times, the block's undeclared memory, the bits of its `U`
  bits of weights that the capacity account doesn't hold (`BlocksRun.undeclared`), is at most `0.07 · U + S_chip + c`,
  where `c = ∑_i cap_i · T_read` is what the side channels bring in over the read deadline `T_read`. It is stated at
  the challenge times only, until the continuous audit (`Pous.Guarantees.ContinuousAudit`) is tied in.

There is one wipe per block, the one that opens it (so block 0 has one too, which bounds what survives from before the
horizon), as in DEVICE-WIPE's `Verity.Blocks` and `Verity.Wipe`. Blocks and wipes are fields here until that core
vocabulary lands; the wipe's verdict and `kept` are fields as the audits' verdicts are (`PeriodRun.accepts`), and a
bridge from `Verity.Wipe.SurvivorBound` to the wipe's event is to give `kept` its meaning.
-/

namespace Accounting

/-- The wipe that opens a server-time block on device `d`, on the coins `(i : I) → Ω i`: the bits of device memory its
fill covers, the memory challenger's verdict, and `kept`, the bits of foreign content (beyond the committed value of
the carried state) that the block can read back from what the wipe leaves, at each value of the coins. -/
structure BlockWipe {ι : Type} (d : Device ι) {I : Type} (Ω : I → Type) where
  /-- the bits of device memory the fill covers -/
  fill : ℕ
  fill_le : fill ≤ d.mem
  /-- the memory challenger's verdict on the wipe, at the coins -/
  accepts : ((i : I) → Ω i) → Prop
  /-- the bits of foreign content beyond the committed carried state that survive the wipe into the block, at the
  coins -/
  kept : ((i : I) → Ω i) → ℕ

namespace BlockWipe

variable {ι : Type} {d : Device ι} {I : Type} {Ω : I → Type} (w : BlockWipe d Ω)

/-- The bits of the device's state the fill leaves out: its on-chip state and the device memory the fill doesn't reach. -/
def unfilled : ℕ := d.chip + (d.mem - w.fill)

/-- **The wipe's survivor bound** at fraction `f`, at the coins `ξ`: at most `f · fill + 2λ` bits of foreign content
(`λ = 128`), plus the unfilled state, survive beyond the committed carried state. -/
def Survives (f : ℝ) (ξ : (i : I) → Ω i) : Prop :=
  (w.kept ξ : ℝ) ≤ f * w.fill + 2 * 128 + w.unfilled

end BlockWipe

/-- A horizon of `B` server-time blocks of a node's run on device `d` with accounts `α`, over the circuit `C`, on the
coins `(i : I) → Ω i`: each block's period, the wipe that opens it, the bits of weights its PoUS store holds (`U`),
its PoUS challenge times at each value of the coins, and the read deadline `T_read` in ticks. -/
structure BlocksRun {ι : Type} (d : Device ι) (α : Type) [Fintype α] (C : Verity.Circuit) {I : Type}
    (Ω : I → Type) (B : ℕ) where
  /-- block `b`'s period, decided by the five audits -/
  block : Fin B → PeriodRun d α C Ω
  /-- the wipe that opens block `b` -/
  wipe : Fin B → BlockWipe d Ω
  /-- the bits of weights block `b`'s PoUS store holds (U) -/
  weights : Fin B → ℕ
  /-- block `b`'s PoUS challenge times, at the coins -/
  chal : Fin B → ((i : I) → Ω i) → Finset ℕ
  /-- the read deadline, in ticks (T_read) -/
  readDeadline : ℕ

namespace BlocksRun

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type} {B : ℕ}
  (s : BlocksRun d α C Ω B)

/-- **Block `b`'s undeclared memory at tick `t`**: the bits of its weights that its capacity account doesn't hold in
device memory at the end of the tick. -/
def undeclared (b : Fin B) (ξ : (i : I) → Ω i) (t : ℕ) : ℝ :=
  (s.weights b : ℝ) - ((s.block b).schedule ξ t).mem ((s.block b).acct .capacity)

/-- What the side channels bring in over the read deadline (c): `∑_i cap_i · T_read` bits. -/
def sideBits [Fintype ι] : ℕ := ∑ i, d.cap i * s.readDeadline

/-- **Block `b`'s memory term** (ASK-132, PoUS over the weights at a share of `0.07`), at the coins `ξ`: at each of its
PoUS challenge times, its undeclared memory is at most `0.07 · U + S_chip + c`. -/
def MemoryTerm [Fintype ι] (b : Fin B) (ξ : (i : I) → Ω i) : Prop :=
  ∀ t ∈ s.chal b ξ, s.undeclared b ξ t ≤ 7 / 100 * s.weights b + d.chip + s.sideBits

end BlocksRun

end Accounting
