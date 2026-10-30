import RemoteAudit.P2Protocol
import Properties.RemoteAudit.Line

/-!
# PoUB-P2's line of record, over what the P2 subprotocol's check accepts (`RemoteAudit.P2Protocol`)

The P2 subprotocol (`RemoteAudit.P2Protocol.protocol`, onsite's `.poub`) replays the verifier's log of a session on the
store it holds and credits what `P2Session.credit` reads off the session's record: `q_bytes`, `UNIT_BYTES` bytes for
each unit its passes read less the wrong receipts' bound its pooled draw publishes. This file reads that session the way
`Properties.RemoteAudit.Line` reads v0's, so that v0's LineHeld argument carries over:

* the node (`Node`): the session's parameters and the store the verifier holds, `N` the store's units (`Node.blocks`);
* the record of an outcome (`Record`): the verifier's clock floor and its log, run through the subprotocol's step from
  the session's first state (`Record.run`, `replay`'s loop), accepted where it ends accepted (`Verified`), and the state
  it reaches where a request throws (`Record.reached`);
* the record's opening draw (`recordDraw`, `recordCommitted`): draw `t` is window `t`, with the receipts the state the
  record reaches has there, drawing `min(s, U_t)` at the session's `samples`;
* the trace of record (`ofRecord`) and the record's windows as a paced session (`recordPace`);
* what the record credits read (`ReadAt`): `8·q_bytes` of its credit is at most the bits its `P` passes (`passes`)
  read on the node, `UNIT_BYTES` bytes a unit.
-/

namespace Properties.RemoteAudit.P2Line

open Accounting.Remote.Held

/-- What is fixed before the coins: the session's parameters and the store the verifier holds. -/
structure Node where
  params : RemoteAudit.Session.Params
  held : RemoteAudit.P2Session.Held

/-- The session's windows: its parameters'. -/
def Node.windows (n : Node) : ℕ := n.params.windows

/-- The session's draws: one a window. -/
def Node.draws (n : Node) : ℕ := n.params.windows

/-- `N`, the registered blocks: the store's units. -/
def Node.blocks (n : Node) : ℕ := n.held.reg.units

/-- What one outcome gives: the verifier's clock floor and its log. -/
structure Record where
  notBeforeNs : ℕ
  log : Array RemoteAudit.P2Protocol.Op

/-- The session's first state on the node, before its first request: `P2Session.init` at the node's parameters and
store, with its 2W coins not yet released. -/
def Record.init (node : Node) (r : Record) : RemoteAudit.Session.State :=
  RemoteAudit.P2Session.init node.held node.params
    (Array.replicate (2 * node.params.windows) ByteArray.empty) r.notBeforeNs

/-- The record's log through the P2 subprotocol's step, from the session's first state on the node. -/
def Record.run (node : Node) (r : Record) : Except String RemoteAudit.Session.State :=
  r.log.foldlM (RemoteAudit.P2Protocol.step node.held) (r.init node)

/-- The verifier accepts the record's session: every request runs and the session ends accepted. -/
def Verified (node : Node) (r : Record) : Prop :=
  ∃ s, r.run node = .ok s ∧ s.outcome = some "accepted"

/-- The verifier's record of the session (`Session.State.summary`): some summary exactly where it accepts. -/
def Record.summary (node : Node) (r : Record) : Option RemoteAudit.Session.Summary :=
  match r.run node with
  | .ok s => s.summary
  | .error _ => none

/-- One request through the P2 subprotocol's step: the state it leaves, or the state it was given where it throws. -/
def reachStep (held : RemoteAudit.P2Session.Held) (s : RemoteAudit.Session.State) (op : RemoteAudit.P2Protocol.Op) :
    Except RemoteAudit.Session.State RemoteAudit.Session.State :=
  match RemoteAudit.P2Protocol.step held s op with
  | .ok s' => .ok s'
  | .error _ => .error s

/-- The state the record's session reaches: its run's last where every request runs, else the state before the first
request that throws. -/
def Record.reached (node : Node) (r : Record) : RemoteAudit.Session.State :=
  match r.log.foldlM (reachStep node.held) (r.init node) with
  | .ok s => s
  | .error s => s

/-- Window `t`'s receipts in the state the record's session reaches, `0` for a window it never opened. -/
def Record.receiptsAt (node : Node) (r : Record) (t : ℕ) : ℕ :=
  ((r.reached node).windows[t]?.map RemoteAudit.Session.Win.receipts).getD 0

/-- **The record's opening draw**: draw `t` is window `t`, with the receipts it has in the state the record's session
reaches at `ω` (at most `receiptMax`), drawing `min(s, U_t)` at the session's `samples`. -/
def recordDraw {κ : ℕ} {X : Type} (node : Node) (rec : Accounting.Remote.Draw.Coins node.draws κ X → Record)
    (correct : Fin node.draws → Accounting.Remote.Draw.Coins node.draws κ X → Finset ℕ) :
    Accounting.Remote.Draw.Session node.draws Line.receiptMax (fun _ => node.params.samples) κ X where
  size c ω := ⟨min ((rec ω).receiptsAt node c) Line.receiptMax, Nat.lt_succ_of_le (min_le_right _ _)⟩
  correct := correct

/-- **The records' opening draw, read through the coin commitment.** -/
def recordCommitted {κ : ℕ} {X Cm : Type} (node : Node)
    (recs : Cm → Accounting.Remote.Draw.Coins node.draws κ X → Record)
    (correct : Cm → Fin node.draws → Accounting.Remote.Draw.Coins node.draws κ X → Finset ℕ) :
    Accounting.Remote.Draw.Committed node.draws Line.receiptMax (fun _ => node.params.samples) κ X Cm where
  view c := recordDraw node (recs c) (correct c)

/-- The record of outcome `ω`: the prover's, holding the deployed commitment `com ω`. -/
def played {Ω Cm : Type} (recs : Cm → Ω → Record) (com : Ω → Cm) (ω : Ω) : Record := recs (com ω) ω

/-- **The trace of record**: `T` over the store's units with the verifier's verdict on the record as its own. -/
def ofRecord {Ω : Type} (T : Trace Ω) (node : Node) (rec : Ω → Record) : Trace Ω :=
  { Line.overBlocks T node.blocks with accepts := fun ω => Verified node (rec ω) }

/-- The windows of the verifier's record of the session, none where it has none. -/
def Record.windows (node : Node) (r : Record) : Array RemoteAudit.Session.WindowSummary :=
  match r.summary node with
  | some sm => sm.windows
  | none => #[]

/-- Window `g`'s first pass among windows `ws`: the passes of the windows before it. -/
def passesBefore (ws : Array RemoteAudit.Session.WindowSummary) (g : ℕ) : ℕ :=
  ((List.range g).map fun i => ws[i]!.lengths.size).sum

/-- A window's last root's arrival, its release where it has none. -/
def lastNs (w : RemoteAudit.Session.WindowSummary) : ℕ := w.arrivedNs.back?.getD w.releasedNs

/-- `e_g`, window `g`'s end: the later of its last root and window `g + 1`'s coin, the last window's last root. -/
def ended (ws : Array RemoteAudit.Session.WindowSummary) (g : ℕ) : ℕ :=
  if g + 1 < ws.size then max (lastNs ws[g]!) ws[g + 1]!.releasedNs else lastNs ws[g]!

/-- The record's windows as a paced session on the verifier's clock: the node's `windows` groups, one a window, the
last closing; past the last window its coin stands. -/
def recordPace {Ω : Type} (node : Node) (rec : Ω → Record) : Pace Ω where
  credited := node.windows - 1
  first ω g := passesBefore ((rec ω).windows node) g
  released ω g := ((rec ω).windows node)[min g (node.windows - 1)]!.releasedNs
  ended ω g := ended ((rec ω).windows node) g

/-- `P`, the passes a record credits: all its windows'. -/
def passes (sm : RemoteAudit.Session.Summary) : ℕ := passesBefore sm.windows sm.windows.size

/-- What the record credits read at `ω`, in bits: if the verifier accepts the session with record `sm`, `8·q_bytes`
of its credit (`P2Session.credit`) is at most the bits of the blocks its `P` passes read on the node, `UNIT_BYTES` bytes
a block. -/
def ReadAt {Ω : Type} (T : Trace Ω) (ω : Ω) (node : Node) (r : Record) : Prop :=
  ∀ sm, r.summary node = some sm →
    (8 * (RemoteAudit.P2Session.credit sm).qBytes : ℝ) ≤ (8 * RemoteAudit.P2.UNIT_BYTES : ℝ) * T.reads ω (passes sm)

end Properties.RemoteAudit.P2Line
