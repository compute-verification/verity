import Definitions.Accounting.JointPeriod

/-!
# Server-time blocks audited by every protocol (trusted)

A server's life is cut into server-time blocks, each opened by a certified wipe of its device memory. Over a horizon of
`B` blocks on one device, one circuit and one coin space (`BlocksRun`):

* **Block `b`** is a period of the node's run (`Accounting.PeriodRun`) decided by the five audits, sampled proofs, PoUW,
  PoUS, PoUB and the network certifier, each on its own coordinate of the coins. Every block's schedule and committed
  transcript are functions of all the coins, so a block may adapt to the coins of the blocks before it.
* **The memory term of block `b` at share `σ`** (`BlocksRun.MemoryTerm`), what PoUS certifies: at each of the block's
  PoUS challenge times, the block's undeclared memory, the bits of the device's memory `M` (`Device.mem`) that its
  capacity account doesn't hold (`BlocksRun.undeclared`), is at most `σ · M + S_chip + c`, where
  `c = ∑_i cap_i · T_read` is what the side channels bring in over a round, the read deadline `T_read`. The share is
  the one PoUS's guarantee certifies for the block's store on that device, not a policy of the joint statements; at the
  demo's store it is `0.1211` (`demoMemoryShare`, `Pous.Guarantees.ObligationDemo`'s). It is stated at the challenge
  times only, until the continuous audit (`Pous.Guarantees.ContinuousAudit`) is tied in.

The wipe that opens block `b` is core's (`Verity.Wipe`, at boundary `b` of a schedule of server-time blocks,
`Verity.Blocks`), so block 0 has one too, which bounds what survives from before the horizon. What it guarantees is its
survivor bound (`Verity.Wipe.SurvivorBound`) at the pair its verdict publishes, a bound in the wipe's own game over a
uniform foreign content, the seed and the spot reads. The joint guarantee reads it on the run through a map of the run
into the wipe's game at each boundary (`Verity.Wipe.OfRun`), on a run whose coins there are the wipe's
(`Accounting.Guarantees.JointHorizonBlocksLive`). The blocks are fields here, not read off that core vocabulary.
-/

namespace Accounting

/-- A horizon of `B` server-time blocks of a node's run on device `d` with accounts `α`, over the circuit `C`, on the
coins `(i : I) → Ω i`: each block's period, its PoUS challenge times at each value of the coins, and the read deadline
`T_read` in ticks. -/
structure BlocksRun {ι : Type} (d : Device ι) (α : Type) [Fintype α] (C : Verity.Circuit) {I : Type}
    (Ω : I → Type) (B : ℕ) where
  /-- block `b`'s period, decided by the five audits -/
  block : Fin B → PeriodRun d α C Ω
  /-- block `b`'s PoUS challenge times, at the coins -/
  chal : Fin B → ((i : I) → Ω i) → Finset ℕ
  /-- the read deadline, in ticks (T_read) -/
  readDeadline : ℕ

namespace BlocksRun

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type} {B : ℕ}
  (s : BlocksRun d α C Ω B)

/-- **Block `b`'s undeclared memory at tick `t`**: the bits of device memory its capacity account doesn't hold at the
end of the tick, `M` less what it holds. -/
def undeclared (b : Fin B) (ξ : (i : I) → Ω i) (t : ℕ) : ℝ :=
  (d.mem : ℝ) - ((s.block b).schedule ξ t).mem ((s.block b).acct .capacity)

/-- What the side channels bring in over a round, the read deadline (c): `∑_i cap_i · T_read` bits. -/
def sideBits [Fintype ι] : ℕ := ∑ i, d.cap i * s.readDeadline

/-- **Block `b`'s memory term at share `σ`** (what PoUS certifies), at the coins `ξ`: at each of its PoUS challenge
times, its undeclared memory is at most `σ · M + S_chip + c`, `M` the device's memory. -/
def MemoryTerm [Fintype ι] (σ : ℝ) (b : Fin B) (ξ : (i : I) → Ω i) : Prop :=
  ∀ t ∈ s.chal b ξ, s.undeclared b ξ t ≤ σ * d.mem + d.chip + s.sideBits

end BlocksRun

/-- **The demo's memory share**, `0.1211`: at the demo's store, `Pous.Guarantees.ObligationDemo` bounds the bits of the
device's memory `M` above the state bound `S₀` by `0.1211 · M`, so the demo's memory term is
`BlocksRun.MemoryTerm demoMemoryShare`. -/
noncomputable def demoMemoryShare : ℝ := 1211 / 10000

end Accounting
