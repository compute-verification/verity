module

public import Definitions.Poub.Remote.Held
public import Definitions.Poub.Remote.Sample
public import Poub.Line
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-! The line of record's inputs (`Node`, `Record`), read as the memory model's paced session. -/

@[expose] public section

namespace Properties.Poub.Line

open ResourceAccounting.Remote.Held _root_.Poub.Line

/-- The node and the session's structure the line reads, fixed before the coins. -/
structure Node where
  params : Poub.Session.Params
  /-- the windows a sample, `groups_per_coin`: windows `c·g … (c + 1)·g − 1` are sampled from one openings coin; at most 1,
  one a window -/
  groupsPerCoin : ℕ
  reg : Poub.Registration
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

/-- The session's samples: its windows `g` a sample, `⌈W/g⌉` of them (`Poub.Line.samplesOf`), one a window at
g ≤ 1. -/
def Node.samples (n : Node) : ℕ :=
  if n.groupsPerCoin ≤ 1 then n.windows else (n.windows + n.groupsPerCoin - 1) / n.groupsPerCoin

/-- `N`, the registered blocks: the registration's. -/
def Node.blocks (n : Node) : ℕ := n.reg.blocks

/-- What one outcome gives: the verifier's inputs and what the auditor saw of each window. -/
structure Record where
  coins : Array ByteArray
  appendCoins : Array ByteArray
  notBeforeNs : ℕ
  events : Array Poub.Session.Event
  seen : Array Seen

/-- The record's session through the Lean verifier, at the node's parameters, windows a sample and registration. -/
def Record.run (node : Node) (r : Record) :
    Except String (Array Poub.Session.Result × Poub.Session.State) :=
  _root_.Poub.Shared.run node.groupsPerCoin
    (Poub.Session.State.init node.params node.reg r.coins r.appendCoins r.notBeforeNs) r.events

/-- The Lean verifier accepts the record's session: every event runs and the session ends accepted. -/
def Verified (node : Node) (r : Record) : Prop :=
  ∃ rs s, r.run node = .ok (rs, s) ∧ s.outcome = some "accepted"

/-- The Lean verifier's record of the session (`Session.State.summary`): some summary exactly where it accepts. -/
def Record.summary (node : Node) (r : Record) : Option Poub.Session.Summary :=
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

/-- The record's line on the node: `lineOfRecordAt` at the node's windows a sample, the Lean verifier's record of the
session and the node's `samples`, or refused where it gives none. -/
def lineOf (node : Node) (r : Record) : Except String Line :=
  match r.summary node with
  | some sm => lineOfRecordAt node.groupsPerCoin sm node.params.samples r.seen node.per node.blockBytes node.chipBytes
      node.nodeBytes node.delta node.rates
  | none => .error "the Lean verifier gave the session no record"

/-- The groups the record's line reads (`Poub.Line.ofSummaryAt`), none where it reads none. -/
def Record.groups (node : Node) (r : Record) : Array Group :=
  match r.summary node with
  | some sm => match _root_.Poub.Line.ofSummaryAt node.groupsPerCoin sm node.params.samples r.seen with
    | .ok (gs, _) => gs
    | .error _ => #[]
  | none => #[]

/-- Each sample's receipts among windows `ws` sampled `g` windows a coin: each window's `Σ lengths_t` at g ≤ 1, and
above it sample `c`'s windows' summed (`Poub.Line.samplesOf`). -/
def sampleReceipts (g : ℕ) (ws : Array Poub.Session.WindowSummary) : Array ℕ :=
  if g ≤ 1 then ws.map _root_.Poub.Line.receipts else (_root_.Poub.Line.samplesOf g ws).map (·.1)

/-- `U_c`, each sample's receipts in the Lean verifier's record of the session (a window's at one window a coin), none
where it has none. -/
def Record.receipts (node : Node) (r : Record) : Array ℕ :=
  match r.summary node with
  | some sm => sampleReceipts node.groupsPerCoin sm.windows
  | none => #[]

/-- One event through the Lean verifier at `g` windows a sample: the state it leaves, or the state it was given where it
throws. -/
def reachStep (g : ℕ) (s : Poub.Session.State) (e : Poub.Session.Event) :
    Except Poub.Session.State Poub.Session.State :=
  match _root_.Poub.Shared.step g s e with
  | .ok (_, s') => .ok s'
  | .error _ => .error s

/-- The state the record's session reaches in the Lean verifier: its run's last where every event runs, else the
state before the first event that throws. -/
def Record.reached (node : Node) (r : Record) : Poub.Session.State :=
  match r.events.foldlM (reachStep node.groupsPerCoin)
      (Poub.Session.State.init node.params node.reg r.coins r.appendCoins r.notBeforeNs) with
  | .ok s => s
  | .error s => s

/-- Window `t`'s receipts in the state the record's session reaches, its passes' lengths summed (`Session.Win.receipts`),
`0` for a window it never opened. -/
def Record.receiptsAt (node : Node) (r : Record) (t : ℕ) : ℕ :=
  ((r.reached node).windows[t]?.map Poub.Session.Win.receipts).getD 0

/-- Sample `c`'s receipts in the state the record's session reaches: window `c`'s at g ≤ 1 (`receiptsAt`), and above it
those of its windows `c·g … (c + 1)·g − 1` the state has, summed. -/
def Record.sampleReceiptsAt (node : Node) (r : Record) (c : ℕ) : ℕ :=
  if node.groupsPerCoin ≤ 1 then r.receiptsAt node c
  else ((r.reached node).windows.extract (c * node.groupsPerCoin) (c * node.groupsPerCoin + node.groupsPerCoin)).foldl
    (fun a w => a + w.receipts) 0

/-- `m`, the most receipts a sample the line reads has: `receive` rejects a pass that would take its window's receipts
to `RECEIPT_LIMIT` (`Properties.Poub.ReceiptsBelowLimit`), and the line refuses a sample of `g` > 1 windows with
`RECEIPT_LIMIT` or more (`Poub.Line.ofSummaryAt`). -/
def receiptMax : ℕ := Poub.Session.RECEIPT_LIMIT - 1

/-- **The record's opening sample**: the session's samples, sample `c` of outcome `ω` with the receipts its windows have
in the state the record's session reaches at `ω` (at most `receiptMax`), sampling `min(s, U_c)` at the session's
`samples`, with the correct receipts `correct`. -/
def recordSample {κ : ℕ} {X : Type} (node : Node) (rec : ResourceAccounting.Remote.Sample.SampleCoins node.samples κ X → Record)
    (correct : Fin node.samples → ResourceAccounting.Remote.Sample.SampleCoins node.samples κ X → Finset ℕ) :
    ResourceAccounting.Remote.Sample.Run node.samples receiptMax (fun _ => node.params.samples) κ X where
  size c ω := ⟨min ((rec ω).sampleReceiptsAt node c) receiptMax, Nat.lt_succ_of_le (min_le_right _ _)⟩
  correct := correct

/-- **The records' opening sample, read through the coin commitment**: holding commitment `c`, the prover's record at
`ω` is `recs c ω`, and the session it plays is that record's opening sample (`recordSample`), with the correct receipts
`correct c`. -/
def recordCommitted {κ : ℕ} {X Cm : Type} (node : Node)
    (recs : Cm → ResourceAccounting.Remote.Sample.SampleCoins node.samples κ X → Record)
    (correct : Cm → Fin node.samples → ResourceAccounting.Remote.Sample.SampleCoins node.samples κ X → Finset ℕ) :
    ResourceAccounting.Remote.Sample.Committed node.samples receiptMax (fun _ => node.params.samples) κ X Cm where
  view c := recordSample node (recs c) (correct c)

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

/-- `B_l`, one link burst in blocks. -/
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

end Properties.Poub.Line
