import Definitions.Accounting.Remote.Held
import RemoteAudit.Session
import RemoteAudit.Line

/-!
# The line of record's inputs, read as the memory model's (`Properties.RemoteAudit.Line`)

What a gate run's memory line (`RemoteAudit.Line.lineOfRecord`) reads of one session, and the paced session
(`Accounting.Remote.Held.Pace`) the memory model reads it as:

* `Node`: what is fixed before the coins: the session's parameters and the registration it runs on (so its windows,
  one group each, `Node.windows`, and its registered blocks, `Node.blocks`), the groups a stretch, a block's bytes, the
  node's on-chip storage and memory in bytes, its rates (`RemoteAudit.Line.Rates`) and the δ the line reads its pooled
  bound at. `Node.squarings` … `Node.chipBlocks` are those rates in the model's units: a tick is a nanosecond on the
  record's clock and a block is `blockBytes`. Its δ is a nonnegative rational (`ℚ≥0`): the line computes at no δ
  outside `(0, 1)`.
* `Record`: what one outcome gives: the verifier's inputs (its coins, the earliest coin release and the session's
  events in order) and what the auditor saw of each window (`RemoteAudit.Line.Seen`: its coin's release and last root,
  and its passes' reads by kind).
* `Verified`: the Lean verifier accepts the record's session (`Session.run` from `State.init` at the node's
  parameters and registration, then `outcome`).
* `Record.summary`: the Lean verifier's record of the session (`Session.State.summary`), which is some summary
  exactly where it accepts. The line reads it (`lineOf`, `lineOfRecord`): each window's passes, its receipts and draws,
  and the registered blocks; `Record.groups` and `Record.receipts` are its groups and its windows' receipts.
* `ofRecord`: a trace with the Lean verifier's verdict on the record as its own, the trace the line's premises read.
* `recordPace`: the record's groups as the model's paced session, group `g`'s first pass the passes before it, its coin
  and its end `e_g` the record's as `stretches` reads them (`ended`); past the last group its coin stands, so the coins
  stay in order.
* `stretchPace`: stretch `m`'s groups `m·per … (m + 1)·per` alone as a paced session, `per` credited groups and the
  closing group `(m + 1)·per`; `spanStart` and `spanStop` its span `[u, v)` as `stretches` reads it, from group 0's end
  for the first stretch and from group `m·per`'s coin for the rest, to group `(m + 1)·per`'s coin.
-/

namespace Properties.RemoteAudit.Line

open Accounting.Remote.Held _root_.RemoteAudit.Line

/-- The node and the session's structure the line reads, fixed before the coins. -/
structure Node where
  /-- the session's parameters: its windows, one group each, and the receipts each draws -/
  params : RemoteAudit.Session.Params
  /-- the registration the session runs on, its registered blocks among it -/
  reg : RemoteAudit.Registration
  /-- the groups a stretch -/
  per : ℕ
  blockBytes : ℕ
  chipBytes : ℕ
  nodeBytes : ℕ
  rates : Rates
  /-- the δ the line reads its pooled bound at, a nonnegative rational: the line computes at none outside `(0, 1)`
  (`Pooled.pooled` refuses it, `line_delta`) -/
  delta : ℚ≥0

/-- The session's windows, one group each: its parameters'. -/
def Node.windows (n : Node) : ℕ := n.params.windows

/-- `N`, the registered blocks: the registration's. -/
def Node.blocks (n : Node) : ℕ := n.reg.blocks

/-- What one outcome gives: the verifier's inputs and what the auditor saw of each window. -/
structure Record where
  coins : Array ByteArray
  appendCoins : Array ByteArray
  notBeforeNs : ℕ
  events : Array RemoteAudit.Session.Event
  seen : Array Seen

/-- The record's session through the Lean verifier, at the node's parameters and registration. -/
def Record.run (node : Node) (r : Record) :
    Except String (Array RemoteAudit.Session.Result × RemoteAudit.Session.State) :=
  RemoteAudit.Session.run
    (RemoteAudit.Session.State.init node.params node.reg r.coins r.appendCoins r.notBeforeNs) r.events

/-- The Lean verifier accepts the record's session: every event runs and the session ends accepted. -/
def Verified (node : Node) (r : Record) : Prop :=
  ∃ rs s, r.run node = .ok (rs, s) ∧ s.outcome = some "accepted"

/-- The Lean verifier's record of the session (`Session.State.summary`): some summary exactly where it accepts. -/
def Record.summary (node : Node) (r : Record) : Option RemoteAudit.Session.Summary :=
  match r.run node with
  | .ok (_, s) => s.summary
  | .error _ => none

/-- **The trace of record**: the trace `T` with the Lean verifier's verdict on the record, accepting exactly where
`Verified` does. -/
def ofRecord {Ω : Type} (T : Trace Ω) (node : Node) (rec : Ω → Record) : Trace Ω :=
  { T with accepts := fun ω => Verified node (rec ω) }

/-- The record's line on the node: `lineOfRecord` at the Lean verifier's record of the session and the node's
`samples`, or refused where it gives none. -/
def lineOf (node : Node) (r : Record) : Except String Line :=
  match r.summary node with
  | some sm => lineOfRecord sm node.params.samples r.seen node.per node.blockBytes node.chipBytes node.nodeBytes
      node.delta node.rates
  | none => .error "the Lean verifier gave the session no record"

/-- The groups the record's line reads (`RemoteAudit.Line.ofSummary`), none where it reads none. -/
def Record.groups (node : Node) (r : Record) : Array Group :=
  match r.summary node with
  | some sm => match _root_.RemoteAudit.Line.ofSummary sm node.params.samples r.seen with
    | .ok (gs, _) => gs
    | .error _ => #[]
  | none => #[]

/-- `U_t`, each window's receipts in the Lean verifier's record of the session, none where it has none. -/
def Record.receipts (node : Node) (r : Record) : Array ℕ :=
  match r.summary node with
  | some sm => sm.windows.map _root_.RemoteAudit.Line.receipts
  | none => #[]

/-- `s`, the node's squarings a tick. -/
noncomputable def Node.squarings (n : Node) : ℝ := (n.rates.squaringsPerS : ℝ) / (ns : ℝ)

/-- `E`, a block's encode in squarings. -/
noncomputable def Node.encode (n : Node) : ℝ := n.rates.encodeSquarings

/-- `β`, the blocks the node's disks bring back a tick. -/
noncomputable def Node.disk (n : Node) : ℝ := (n.rates.diskBytesPerS : ℝ) / n.blockBytes / (ns : ℝ)

/-- `β_l`, the blocks the node's boundary links bring in a tick. -/
noncomputable def Node.link (n : Node) : ℝ := (n.rates.linkBytesPerS : ℝ) / n.blockBytes / (ns : ℝ)

/-- `B_r`, one maximal rebuild batch in blocks. -/
noncomputable def Node.rebuildBurst (n : Node) : ℝ := (n.rates.rebuildBurstBytes : ℝ) / n.blockBytes

/-- `B_d`, one maximal disk read in blocks. -/
noncomputable def Node.diskBurst (n : Node) : ℝ := (n.rates.diskBurstBytes : ℝ) / n.blockBytes

/-- `B_l`, one link window in blocks. -/
noncomputable def Node.linkBurst (n : Node) : ℝ := (n.rates.linkBurstBytes : ℝ) / n.blockBytes

/-- `S`, the registered blocks the node's chips can hold any byte of: `chipBytes / blockBytes` rounded up, so a block
split between a chip and memory is one of them. -/
def Node.chipBlocks (n : Node) : ℕ := (n.chipBytes + n.blockBytes - 1) / n.blockBytes

/-- `e_g`, group `g`'s end as `stretches` reads it: the later of its last root and group `g + 1`'s coin, the last
group's last root. -/
def ended (gs : Array Group) (g : ℕ) : ℕ :=
  if g + 1 < gs.size then max gs[g]!.lastNs gs[g + 1]!.releasedNs else gs[g]!.lastNs

/-- Group `g`'s first pass: the passes of the groups before it. -/
def firstPass (gs : Array Group) (g : ℕ) : ℕ := ((List.range g).map fun i => gs[i]!.passes).sum

/-- `s_g`, group `g`'s coin among `G` groups, the last group's past them. -/
def released (gs : Array Group) (G g : ℕ) : ℕ := gs[min g (G - 1)]!.releasedNs

/-- The record's groups as a paced session on the record's clock: the node's `windows` groups, the last closing. -/
def recordPace {Ω : Type} (node : Node) (rec : Ω → Record) : Pace Ω where
  credited := node.windows - 1
  first ω g := firstPass ((rec ω).groups node) g
  released ω g := released ((rec ω).groups node) node.windows g
  ended ω g := ended ((rec ω).groups node) g

/-- Stretch `m`'s groups `m·per … (m + 1)·per` of the groups `gs ω` as a paced session: `per` credited groups and the
closing group `(m + 1)·per`. -/
def stretchPace {Ω : Type} (gs : Ω → Array Group) (per m : ℕ) : Pace Ω where
  credited := per
  first ω g := firstPass (gs ω) (m * per + g)
  released ω g := released (gs ω) (gs ω).size (m * per + g)
  ended ω g := ended (gs ω) (m * per + g)

/-- Stretch `m`'s first tick: group 0's end for the first stretch, group `m·per`'s coin for the rest. -/
def spanStart (gs : Array Group) (per m : ℕ) : ℕ := if m = 0 then ended gs 0 else gs[m * per]!.releasedNs

/-- Stretch `m`'s end, group `(m + 1)·per`'s coin. -/
def spanStop (gs : Array Group) (per m : ℕ) : ℕ := gs[(m + 1) * per]!.releasedNs

end Properties.RemoteAudit.Line
