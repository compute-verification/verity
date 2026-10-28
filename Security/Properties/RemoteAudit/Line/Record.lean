import Definitions.Accounting.Remote.Held
import Definitions.Accounting.Remote.Draw
import RemoteAudit.Session
import RemoteAudit.Shared
import RemoteAudit.Line

/-!
# The line of record's inputs, read as the memory model's (`Properties.RemoteAudit.Line`)

What a gate run's memory line (`RemoteAudit.Line.lineOfRecordAt`) reads of one session, and the paced session
(`Accounting.Remote.Held.Pace`) the memory model reads it as:

* `Node`: what is fixed before the coins: the session's parameters, the windows a draw (`groups_per_coin`, g) and the
  registration it runs on (so its windows, one group each, `Node.windows`, its draws, `Node.draws`, and its registered
  blocks, `Node.blocks`), the groups a stretch, a block's bytes, the node's on-chip storage and memory in bytes, its
  rates (`RemoteAudit.Line.Rates`) and the δ the line reads its pooled bound at. `Node.squarings` … `Node.chipBlocks`
  are those rates in the model's units: a tick is a nanosecond on the record's clock and a block is `blockBytes`. Its
  δ is a nonnegative rational (`ℚ≥0`): the line computes at no δ outside `(0, 1)`.
* `Record`: what one outcome gives: the verifier's inputs (its coins, the earliest coin release and the session's
  events in order) and what the auditor saw of each window (`RemoteAudit.Line.Seen`: its coin's release and last root,
  and its passes' reads by kind).
* `Verified`: the Lean verifier accepts the record's session (`Shared.run` at the node's g from `State.init` at its
  parameters and registration, then `outcome`; at g = 1 `Session.run`, `RemoteAudit.Shared.run_one`).
* `Record.summary`: the Lean verifier's record of the session (`Session.State.summary`), which is some summary
  exactly where it accepts. The line reads it (`lineOf`, `lineOfRecordAt`): each window's passes, each draw's receipts
  and draws (`drawReceipts`: draw `c` is windows `c·g … (c + 1)·g − 1`, a window at g = 1), and the registered blocks;
  `Record.groups` and `Record.receipts` are its groups, one a window, and its draws' receipts.
* `Record.reached`: the state the record's session reaches in the Lean verifier, its run's last or the one before the
  first event that throws; `Record.receiptsAt` its windows' receipts and `Record.drawReceiptsAt` its draws'.
* `recordDraw`: the record's opening draw (`Accounting.Remote.Draw.Session`), one draw for each of the session's
  (`Node.draws`), draw `c`'s size at `ω` its receipts in the state the record reaches at `ω`, at most `receiptMax`
  (`2^64 − 1`, the verifier's `RECEIPT_LIMIT` less one). It reads the reached state rather than the record's summary so
  that `Live`, a draw's size fixed by the coins before it, can hold of a session whose acceptance turns on later coins.
* `recordCommitted`: the prover's records read through the coin commitment (`Accounting.Remote.Draw.Committed`):
  holding commitment `c` its record at `ω` is `recs c ω`, and the session it plays is that record's `recordDraw`;
  `played` is the record of outcome `ω`, the prover's holding the deployed commitment `com ω`.
* `ofRecord`: a trace over the registration's blocks (`overBlocks`) with the Lean verifier's verdict on the record as
  its own, the trace the line's premises read.
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
  /-- the session's parameters: its windows, one group each, and the receipts each draw draws -/
  params : RemoteAudit.Session.Params
  /-- the windows a draw, `groups_per_coin`: windows `c·g … (c + 1)·g − 1` are drawn from one openings coin; at most 1,
  one a window -/
  groupsPerCoin : ℕ
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

/-- The session's draws: its windows `g` a draw, `⌈W/g⌉` of them (`RemoteAudit.Line.drawsOf`), one a window at
g ≤ 1. -/
def Node.draws (n : Node) : ℕ :=
  if n.groupsPerCoin ≤ 1 then n.windows else (n.windows + n.groupsPerCoin - 1) / n.groupsPerCoin

/-- `N`, the registered blocks: the registration's. -/
def Node.blocks (n : Node) : ℕ := n.reg.blocks

/-- What one outcome gives: the verifier's inputs and what the auditor saw of each window. -/
structure Record where
  coins : Array ByteArray
  appendCoins : Array ByteArray
  notBeforeNs : ℕ
  events : Array RemoteAudit.Session.Event
  seen : Array Seen

/-- The record's session through the Lean verifier, at the node's parameters, windows a draw and registration. -/
def Record.run (node : Node) (r : Record) :
    Except String (Array RemoteAudit.Session.Result × RemoteAudit.Session.State) :=
  _root_.RemoteAudit.Shared.run node.groupsPerCoin
    (RemoteAudit.Session.State.init node.params node.reg r.coins r.appendCoins r.notBeforeNs) r.events

/-- The Lean verifier accepts the record's session: every event runs and the session ends accepted. -/
def Verified (node : Node) (r : Record) : Prop :=
  ∃ rs s, r.run node = .ok (rs, s) ∧ s.outcome = some "accepted"

/-- The Lean verifier's record of the session (`Session.State.summary`): some summary exactly where it accepts. -/
def Record.summary (node : Node) (r : Record) : Option RemoteAudit.Session.Summary :=
  match r.run node with
  | .ok (_, s) => s.summary
  | .error _ => none

/-- **The trace over `N` registered blocks**: `T` with each of its sets of registered blocks read as indices below `N`,
the rest unchanged. A trace's registered blocks are the model's name for the store; at `T.blocks = N` this is `T`. -/
def overBlocks {Ω : Type} (T : Trace Ω) (N : ℕ) : Trace Ω :=
  let re : Finset (Fin T.blocks) → Finset (Fin N) := fun s => Finset.univ.filter fun j => (j : ℕ) ∈ s.image Fin.val
  { T with
    blocks := N
    plan := fun ω p => re (T.plan ω p)
    correct := fun ω p => re (T.correct ω p)
    mem := fun ω t => re (T.mem ω t)
    chip := fun ω t => re (T.chip ω t) }

/-- **The trace of record**: the trace `T` over the registration's blocks (`overBlocks` at `Node.blocks`, the `N` the
Lean verifier's record of the session gives the line) with the Lean verifier's verdict on the record, accepting exactly
where `Verified` does. -/
def ofRecord {Ω : Type} (T : Trace Ω) (node : Node) (rec : Ω → Record) : Trace Ω :=
  { overBlocks T node.blocks with accepts := fun ω => Verified node (rec ω) }

/-- The record's line on the node: `lineOfRecordAt` at the node's windows a draw, the Lean verifier's record of the
session and the node's `samples`, or refused where it gives none. -/
def lineOf (node : Node) (r : Record) : Except String Line :=
  match r.summary node with
  | some sm => lineOfRecordAt node.groupsPerCoin sm node.params.samples r.seen node.per node.blockBytes node.chipBytes
      node.nodeBytes node.delta node.rates
  | none => .error "the Lean verifier gave the session no record"

/-- The groups the record's line reads (`RemoteAudit.Line.ofSummaryAt`), none where it reads none. -/
def Record.groups (node : Node) (r : Record) : Array Group :=
  match r.summary node with
  | some sm => match _root_.RemoteAudit.Line.ofSummaryAt node.groupsPerCoin sm node.params.samples r.seen with
    | .ok (gs, _) => gs
    | .error _ => #[]
  | none => #[]

/-- Each draw's receipts among windows `ws` drawn `g` windows a coin: each window's `Σ lengths_t` at g ≤ 1, and above
it draw `c`'s windows' summed (`RemoteAudit.Line.drawsOf`). -/
def drawReceipts (g : ℕ) (ws : Array RemoteAudit.Session.WindowSummary) : Array ℕ :=
  if g ≤ 1 then ws.map _root_.RemoteAudit.Line.receipts else (_root_.RemoteAudit.Line.drawsOf g ws).map (·.1)

/-- `U_c`, each draw's receipts in the Lean verifier's record of the session (a window's at one window a coin), none
where it has none. -/
def Record.receipts (node : Node) (r : Record) : Array ℕ :=
  match r.summary node with
  | some sm => drawReceipts node.groupsPerCoin sm.windows
  | none => #[]

/-- One event through the Lean verifier at `g` windows a draw: the state it leaves, or the state it was given where it
throws. -/
def reachStep (g : ℕ) (s : RemoteAudit.Session.State) (e : RemoteAudit.Session.Event) :
    Except RemoteAudit.Session.State RemoteAudit.Session.State :=
  match _root_.RemoteAudit.Shared.step g s e with
  | .ok (_, s') => .ok s'
  | .error _ => .error s

/-- The state the record's session reaches in the Lean verifier: its run's last where every event runs, else the
state before the first event that throws. -/
def Record.reached (node : Node) (r : Record) : RemoteAudit.Session.State :=
  match r.events.foldlM (reachStep node.groupsPerCoin)
      (RemoteAudit.Session.State.init node.params node.reg r.coins r.appendCoins r.notBeforeNs) with
  | .ok s => s
  | .error s => s

/-- Window `t`'s receipts in the state the record's session reaches, its passes' lengths summed (`Session.Win.receipts`),
`0` for a window it never opened. -/
def Record.receiptsAt (node : Node) (r : Record) (t : ℕ) : ℕ :=
  ((r.reached node).windows[t]?.map RemoteAudit.Session.Win.receipts).getD 0

/-- Draw `c`'s receipts in the state the record's session reaches: window `c`'s at g ≤ 1 (`receiptsAt`), and above it
those of its windows `c·g … (c + 1)·g − 1` the state has, summed. -/
def Record.drawReceiptsAt (node : Node) (r : Record) (c : ℕ) : ℕ :=
  if node.groupsPerCoin ≤ 1 then r.receiptsAt node c
  else ((r.reached node).windows.extract (c * node.groupsPerCoin) (c * node.groupsPerCoin + node.groupsPerCoin)).foldl
    (fun a w => a + w.receipts) 0

/-- `m`, the most receipts a draw the line reads has: `receive` rejects a pass that would take its window's receipts to
`RECEIPT_LIMIT` (`Properties.RemoteAudit.ReceiptsBelowLimit`), and the line refuses a draw of `g` > 1 windows with
`RECEIPT_LIMIT` or more (`RemoteAudit.Line.ofSummaryAt`). -/
def receiptMax : ℕ := RemoteAudit.Session.RECEIPT_LIMIT - 1

/-- **The record's opening draw**: the session's draws, draw `c` of outcome `ω` with the receipts its windows have in
the state the record's session reaches at `ω` (at most `receiptMax`), drawing `min(s, U_c)` at the session's
`samples`, with the correct receipts `correct`. -/
def recordDraw {κ : ℕ} {X : Type} (node : Node) (rec : Accounting.Remote.Draw.DrawCoins node.draws κ X → Record)
    (correct : Fin node.draws → Accounting.Remote.Draw.DrawCoins node.draws κ X → Finset ℕ) :
    Accounting.Remote.Draw.Session node.draws receiptMax (fun _ => node.params.samples) κ X where
  size c ω := ⟨min ((rec ω).drawReceiptsAt node c) receiptMax, Nat.lt_succ_of_le (min_le_right _ _)⟩
  correct := correct

/-- **The records' opening draw, read through the coin commitment**: holding commitment `c`, the prover's record at
`ω` is `recs c ω`, and the session it plays is that record's opening draw (`recordDraw`), with the correct receipts
`correct c`. -/
def recordCommitted {κ : ℕ} {X Cm : Type} (node : Node)
    (recs : Cm → Accounting.Remote.Draw.DrawCoins node.draws κ X → Record)
    (correct : Cm → Fin node.draws → Accounting.Remote.Draw.DrawCoins node.draws κ X → Finset ℕ) :
    Accounting.Remote.Draw.Committed node.draws receiptMax (fun _ => node.params.samples) κ X Cm where
  view c := recordDraw node (recs c) (correct c)

/-- The record of outcome `ω`: the prover's, holding the deployed commitment `com ω`. -/
def played {Ω Cm : Type} (recs : Cm → Ω → Record) (com : Ω → Cm) (ω : Ω) : Record := recs (com ω) ω

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
