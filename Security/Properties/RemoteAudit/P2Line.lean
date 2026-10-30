import RemoteAudit.P2Protocol
import Properties.RemoteAudit.Line

/-!
# PoUB-P2's line of record, over what the P2 subprotocol's check accepts (`RemoteAudit.P2Protocol`)

The P2 subprotocol (`RemoteAudit.P2Protocol.protocol`, onsite's `.poub`) replays the verifier's log of a session on the
store it holds and credits what `P2Session.credit` reads off the session's record: `q_bytes`, `UNIT_BYTES` bytes for
each unit its passes read less the wrong receipts' bound its pooled draw publishes. This file reads that session the way
`Properties.RemoteAudit.Line` reads v0's, so that v0's LineHeld argument carries over:

* the node (`Audited`): the session's parameters and the store the verifier holds, `N` the store's units
  (`Audited.blocks`);
* the record of an outcome (`Replay`): the verifier's clock floor, the prover's request and the verifier's log, run
  through the subprotocol's step from the session's first state (`Replay.run`, `replay`'s loop), accepted where the
  request reads as the node's parameters and the run ends accepted (`Verified`), and the state it reaches where a
  request throws (`Replay.reached`);
* the record's opening draw (`recordDraw`, `recordCommitted`): draw `t` is window `t`, with the receipts the state the
  record reaches has there, drawing `min(s, U_t)` at the session's `samples`;
* the trace of record (`ofRecord`) and the record's windows as a paced session (`recordPace`);
* what the record credits read (`ReadAt`): `8·q_bytes` of its credit is at most the bits its `P` passes (`passes`)
  read on the node, `UNIT_BYTES` bytes a unit;
* **LineHeld**: `Properties.RemoteAudit.Line.LineHeld` for the P2 subprotocol, its node, trace, records, premises and
  bound verbatim at the credit's δ (`P2Session.DELTA`, 2^-40): the probability that the verifier accepts the record's
  session, the network certifiers sign nothing but the verifier's messages entering the node, and the record's credit
  claims more bits read than the trace's passes read (`ReadAt`) is at most `δs + δnet + (δ + Σ_c s / 2^64 + Σ_c adv_c +
  Σ_c hide_c + bind)`. The credit's wrong receipts, `⌈wrong⌉` of the pooled bound of the windows it pools
  (`pooledWindows`), cover the session's pooled bound because an accepted record's windows are the record's draw's: each
  window's units read are its receipts, and the session draws `min(s, U_t)` of them.
-/

namespace Properties.RemoteAudit.P2Line

open Accounting.Remote.Held

/-- What is fixed before the coins: the session's parameters and the store the verifier holds. -/
structure Audited where
  params : RemoteAudit.Session.Params
  held : RemoteAudit.P2Session.Held

/-- The session's windows: its parameters'. -/
def Audited.windows (n : Audited) : ℕ := n.params.windows

/-- The session's draws: one a window. -/
def Audited.draws (n : Audited) : ℕ := n.params.windows

/-- `N`, the registered blocks: the store's units. -/
def Audited.blocks (n : Audited) : ℕ := n.held.reg.units

/-- What one outcome gives: the verifier's clock floor, the prover's request and the verifier's log. -/
structure Replay where
  notBeforeNs : ℕ
  request : Lean.Json
  log : Array RemoteAudit.P2Protocol.Op

/-- The session's first state on the node, before its first request: `P2Session.init` at the node's parameters and
store, with its 2W coins not yet released. -/
def Replay.init (node : Audited) (r : Replay) : RemoteAudit.Session.State :=
  RemoteAudit.P2Session.init node.held node.params
    (Array.replicate (2 * node.params.windows) ByteArray.empty) r.notBeforeNs

/-- The record's log through the P2 subprotocol's step, from the session's first state on the node. -/
def Replay.run (node : Audited) (r : Replay) : Except String RemoteAudit.Session.State :=
  r.log.foldlM (RemoteAudit.P2Protocol.step node.held) (r.init node)

/-- The verifier accepts the record's session on the node, as the P2 subprotocol's check does (`replay`): the request
reads as the node's session parameters against the store it holds (`P2Session.requestOf`), and every request of the
log runs and the session ends accepted. -/
def Verified (node : Audited) (r : Replay) : Prop :=
  RemoteAudit.P2Session.requestOf node.held r.request = .ok node.params ∧
    ∃ s, r.run node = .ok s ∧ s.outcome = some "accepted"

/-- The verifier's record of the session (`Session.State.summary`): some summary exactly where it accepts. -/
def Replay.summary (node : Audited) (r : Replay) : Option RemoteAudit.Session.Summary :=
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
def Replay.reached (node : Audited) (r : Replay) : RemoteAudit.Session.State :=
  match r.log.foldlM (reachStep node.held) (r.init node) with
  | .ok s => s
  | .error s => s

/-- Window `t`'s receipts in the state the record's session reaches, `0` for a window it never opened. -/
def Replay.receiptsAt (node : Audited) (r : Replay) (t : ℕ) : ℕ :=
  ((r.reached node).windows[t]?.map RemoteAudit.Session.Win.receipts).getD 0

/-- **The record's opening draw**: draw `t` is window `t`, with the receipts it has in the state the record's session
reaches at `ω` (at most `receiptMax`), drawing `min(s, U_t)` at the session's `samples`. -/
def recordDraw {κ : ℕ} {X : Type} (node : Audited) (rec : Accounting.Remote.Draw.Coins node.draws κ X → Replay)
    (correct : Fin node.draws → Accounting.Remote.Draw.Coins node.draws κ X → Finset ℕ) :
    Accounting.Remote.Draw.Session node.draws Line.receiptMax (fun _ => node.params.samples) κ X where
  size c ω := ⟨min ((rec ω).receiptsAt node c) Line.receiptMax, Nat.lt_succ_of_le (min_le_right _ _)⟩
  correct := correct

/-- **The records' opening draw, read through the coin commitment.** -/
def recordCommitted {κ : ℕ} {X Cm : Type} (node : Audited)
    (recs : Cm → Accounting.Remote.Draw.Coins node.draws κ X → Replay)
    (correct : Cm → Fin node.draws → Accounting.Remote.Draw.Coins node.draws κ X → Finset ℕ) :
    Accounting.Remote.Draw.Committed node.draws Line.receiptMax (fun _ => node.params.samples) κ X Cm where
  view c := recordDraw node (recs c) (correct c)

/-- The record of outcome `ω`: the prover's, holding the deployed commitment `com ω`. -/
def played {Ω Cm : Type} (recs : Cm → Ω → Replay) (com : Ω → Cm) (ω : Ω) : Replay := recs (com ω) ω

/-- **The trace of record**: `T` over the store's units with the verifier's verdict on the record as its own. -/
def ofRecord {Ω : Type} (T : Trace Ω) (node : Audited) (rec : Ω → Replay) : Trace Ω :=
  { Line.overBlocks T node.blocks with accepts := fun ω => Verified node (rec ω) }

/-- The windows of the verifier's record of the session, none where it has none. -/
def Replay.windows (node : Audited) (r : Replay) : Array RemoteAudit.Session.WindowSummary :=
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
def recordPace {Ω : Type} (node : Audited) (rec : Ω → Replay) : Pace Ω where
  credited := node.windows - 1
  first ω g := passesBefore ((rec ω).windows node) g
  released ω g := ((rec ω).windows node)[min g (node.windows - 1)]!.releasedNs
  ended ω g := ended ((rec ω).windows node) g

/-- `P`, the passes a record credits: all its windows'. -/
def passes (sm : RemoteAudit.Session.Summary) : ℕ := passesBefore sm.windows sm.windows.size

/-- What the record credits read at `ω`, in bits: if the verifier accepts the session with record `sm`, `8·q_bytes`
of its credit (`P2Session.credit`) is at most the bits of the blocks its `P` passes read on the node, `UNIT_BYTES` bytes
a block. -/
def ReadAt {Ω : Type} (T : Trace Ω) (ω : Ω) (node : Audited) (r : Replay) : Prop :=
  ∀ sm, r.summary node = some sm →
    (8 * (RemoteAudit.P2Session.credit sm).qBytes : ℝ) ≤ (8 * RemoteAudit.P2.UNIT_BYTES : ℝ) * T.reads ω (passes sm)

section LineHeld

open Accounting.Remote.Held.Assumptions Accounting.Remote.Draw Accounting.Remote.Draw.Assumptions

/-- The windows the P2 credit pools: each window's units read over its passes and its draws. -/
def pooledWindows (sm : RemoteAudit.Session.Summary) : Array (ℕ × ℕ) :=
  sm.windows.map fun w => (w.lengths.toList.sum, w.drawn)

/-- **PoUB-P2's line of record, in bits read.** On any node, on a trace over its session's coins, one for each window
(`Audited.draws`), the prover's records read through the coin commitment (`recordCommitted`, each record's opening draw
at its `samples` s) and the record of each outcome the one played holding the deployed commitment (`played`), whose
trace of record (`ofRecord`: over the store's units, its verdict the P2 subprotocol's on the record) meets the premises
of v0's line (`Properties.RemoteAudit.Line.LineHeld`) at the record's pace: the probability that the verifier accepts
the record's session, the certifiers sign nothing but the verifier's messages entering the node over the credited
passes, and the record's credit claims more bits read than the passes read (`ReadAt`) is at most
`δs + δnet + (δ + Σ_c s / 2^64 + Σ_c adv_c + Σ_c hide_c + bind)`, `δ` the P2 credit's (`P2Session.DELTA`, 2^-40),
`adv_c` the advantage on `prf c` of draw `c`'s distinguisher at it with leaf `c` of the coin commitment committing
zeros, `hide_c` draw `c`'s hiding term and `bind` the receipts' binding term at `R`. -/
def LineHeld : Prop :=
  ∀ (node : Audited) (κ : ℕ) (X Cm : Type) [Fintype X] (prf : Fin node.draws → Pous.Bits κ → Blocks Line.receiptMax)
    (rest : Coins node.draws κ X → Fin node.draws → ℕ → ℕ) (recs : Cm → Coins node.draws κ X → Replay)
    (correct : Cm → Fin node.draws → Coins node.draws κ X → Finset ℕ) (com : Coins node.draws κ X → Cm)
    (T : Trace (Coins node.draws κ X)) (R : Receipts node.draws κ X) (δs δnet : ℝ),
    let r := played recs com
    let F := recordCommitted node recs correct
    ∀ (O : Outside (ofRecord T node r)) (I : Ingress (Coins node.draws κ X)),
    Paced (ofRecord T node r) (recordPace node r) → Covers (ofRecord T node r) →
    SketchBoundAt (ofRecord T node r) O δs → Covered (ofRecord T node r) O I →
    IngressCounted (ofRecord T node r) I δnet → (∀ c, Live (F.view c)) →
    Refused (F.play com) prf rest (ofRecord T node r) R → Counted (F.play com) (ofRecord T node r) →
    Pouw.pr (fun ω => Verified node (r ω) ∧
        ¬ (Accounting.Remote.Held.Quiet (ofRecord T node r) I.counted ω → ReadAt (ofRecord T node r) ω node (r ω))) ≤
      δs + δnet + ((RemoteAudit.P2Session.DELTA : ℝ) +
        ((∑ _t : Fin node.draws, node.params.samples : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t)
          ((F.play (Committed.junk com t)).distinguisher prf (RemoteAudit.P2Session.DELTA : ℝ) t)).toReal +
        (∑ t, F.hide com prf (RemoteAudit.P2Session.DELTA : ℝ) t).toReal + (R.binding (F.play com) prf rest).toReal)

end LineHeld

end Properties.RemoteAudit.P2Line
