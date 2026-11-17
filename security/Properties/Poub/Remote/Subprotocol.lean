module

public import Poub.Protocol
public import Properties.Poub.Remote.Line

/-! The properties of PoUB's bandwidth component `Poub.Protocol.protocol` (v0, which no composite lists), each a form
over the verifier. -/

@[expose] public section

namespace Properties.Poub.Subprotocol

open ResourceAccounting.Remote.Held ResourceAccounting.Remote.Held.Assumptions ResourceAccounting.Remote.Sample
  ResourceAccounting.Remote.Sample.Assumptions Properties.Poub.Line

/-- **The component accepts bytes that play the record on the node.** The parameters `a` (canonical JSON), the
commitment `b` and the evidence `e` decode under the component's decoders, its check accepts them
(`Properties.Composition.DecodedAccepted` at `Poub.Protocol.protocol`), the commitment is the node's
registration, and the session the parameters and the log decode to (`Poub.Protocol.sessionOf`) is the node's
session parameters and windows a sample with the record's coins, append coins, clock floor and events. -/
def Plays (node : Node) (rec : Record) (a b e : ByteArray) : Prop :=
  ∃ (p : Poub.Protocol.Params) (ev : Poub.Protocol.Evidence),
    (Core.canonicalJson a >>= Poub.Protocol.decodeParams) = .ok p ∧
    Poub.Protocol.decodeCommitment b = .ok node.reg ∧ Poub.Protocol.decodeEvidence e = .ok ev ∧
    Poub.Protocol.sessionOf p ev =
      .ok (node.params, node.groupsPerCoin, rec.coins, rec.appendCoins, rec.notBeforeNs, rec.events) ∧
    ∃ pub prof, Poub.Protocol.check p node.reg ev = .accept pub prof

/-- **The subprotocol accepts bytes that play the record on the node, publishing `pub`**: `Plays` with the public
outputs named, so `Plays` holds exactly where `Publishes` does at some `pub`. -/
def Publishes (node : Node) (rec : Record) (a b e : ByteArray) (pub : Poub.Protocol.Public) : Prop :=
  ∃ (p : Poub.Protocol.Params) (ev : Poub.Protocol.Evidence),
    (Core.canonicalJson a >>= Poub.Protocol.decodeParams) = .ok p ∧
    Poub.Protocol.decodeCommitment b = .ok node.reg ∧ Poub.Protocol.decodeEvidence e = .ok ev ∧
    Poub.Protocol.sessionOf p ev =
      .ok (node.params, node.groupsPerCoin, rec.coins, rec.appendCoins, rec.notBeforeNs, rec.events) ∧
    ∃ prof, Poub.Protocol.check p node.reg ev = .accept pub prof

/-- **`AcceptIsVerified` at a check `chk`**: whenever `chk` accepts parameters `p`, a registration `reg` and a log `e`
with public outputs `pub`, they decode to a session that `Shared.run g` runs to an accepted state with a record `sm`,
and `pub` is what `certify` publishes of `sm` at the session's parameters and g. `AcceptIsVerified`
(`Properties.Poub.Remote.SubprotocolDeployed`) is it at `Poub.Protocol.check`. -/
def AcceptIsVerifiedFor (chk : Poub.Protocol.Params → Poub.Registration → Poub.Protocol.Evidence →
    Core.Outcome Poub.Protocol.Public Lean.Json) : Prop :=
  ∀ (p : Poub.Protocol.Params) (reg : Poub.Registration) (e : Poub.Protocol.Evidence)
    (pub : Poub.Protocol.Public) (prof : Lean.Json),
    chk p reg e = .accept pub prof →
    ∃ (params : Poub.Session.Params) (g : ℕ) (coins appendCoins : Array ByteArray) (notBeforeNs : ℕ)
      (events : Array Poub.Session.Event) (rs : Array Poub.Session.Result)
      (s : Poub.Session.State) (sm : Poub.Session.Summary),
      Poub.Protocol.sessionOf p e = .ok (params, g, coins, appendCoins, notBeforeNs, events) ∧
      Poub.Shared.run g (Poub.Session.State.init params reg coins appendCoins notBeforeNs) events =
        .ok (rs, s) ∧
      s.outcome = some "accepted" ∧ s.summary = some sm ∧ pub = (Poub.Protocol.certify params g sm).1

/-- **`PublishedEps` at a check `chk`**: whenever `chk` accepts parameters `p`, a registration `reg` and a log `e` with
public outputs `pub`, they decode to a session, and either `pub` certifies nothing or its `eps` is
`δ + Σ_c s/2^64` at every node of the session's parameters and g whose δ is `Poub.Protocol.DELTA`.
`PublishedEps` (`Properties.Poub.Remote.SubprotocolDeployed`) is it at `Poub.Protocol.check`. -/
def PublishedEpsFor (chk : Poub.Protocol.Params → Poub.Registration → Poub.Protocol.Evidence →
    Core.Outcome Poub.Protocol.Public Lean.Json) : Prop :=
  ∀ (p : Poub.Protocol.Params) (reg : Poub.Registration) (e : Poub.Protocol.Evidence)
    (pub : Poub.Protocol.Public) (prof : Lean.Json),
    chk p reg e = .accept pub prof →
    ∃ (params : Poub.Session.Params) (g : ℕ) (coins appendCoins : Array ByteArray) (notBeforeNs : ℕ)
      (events : Array Poub.Session.Event),
      Poub.Protocol.sessionOf p e = .ok (params, g, coins, appendCoins, notBeforeNs, events) ∧
      ((pub.eps = ⟨0, 1⟩ ∧ pub.qBytes = 0) ∨
        ∀ node : Node, node.params = params → node.groupsPerCoin = g →
          (node.delta : ℚ) = Poub.Protocol.DELTA →
          (pub.eps.num : ℚ) / pub.eps.den =
            (node.delta : ℚ) + ((∑ _c : Fin node.samples, node.params.samples : ℕ) : ℚ) / 2 ^ 64)

/-- The groups the line reads of record `r` on the node, when a verifier records it as `sum node r`: `Record.groups`
at `sum = Record.summary`. -/
def groupsBy (sum : Node → Record → Option Poub.Session.Summary) (node : Node) (r : Record) :
    Array _root_.Poub.Line.Group :=
  match sum node r with
  | some sm => match _root_.Poub.Line.ofSummaryAt node.groupsPerCoin sm node.params.samples r.seen with
    | .ok (gs, _) => gs
    | .error _ => #[]
  | none => #[]

/-- The line of record `r` on the node, when a verifier records it as `sum node r`: `lineOf` at
`sum = Record.summary`. -/
def lineBy (sum : Node → Record → Option Poub.Session.Summary) (node : Node) (r : Record) :
    Except String _root_.Poub.Line.Line :=
  match sum node r with
  | some sm => _root_.Poub.Line.lineOfRecordAt node.groupsPerCoin sm node.params.samples r.seen node.per
      node.blockBytes node.chipBytes node.nodeBytes node.delta node.rates
  | none => .error "the Lean verifier gave the session no record"

/-- `ReadAt` of the line a verifier's record `sum` gives: if the line computes, `8·b·(P·N − W)` at its groups' passes
`P` is at most the bits of the blocks they read on the node at `ω`. `ReadAt` at `sum = Record.summary`. -/
def ReadBy (sum : Node → Record → Option Poub.Session.Summary) {Ω : Type} (T : Trace Ω) (ω : Ω) (node : Node)
    (r : Record) : Prop :=
  ∀ l, lineBy sum node r = .ok l →
    (8 * node.blockBytes : ℝ) *
        (((firstPass (groupsBy sum node r) (groupsBy sum node r).size * node.blocks : ℕ) : ℝ) - l.wrong) ≤
      (8 * node.blockBytes : ℝ) * T.reads ω (firstPass (groupsBy sum node r) (groupsBy sum node r).size)

/-- `recordPace` of the groups a verifier's record `sum` gives. -/
def paceBy {Ω : Type} (sum : Node → Record → Option Poub.Session.Summary) (node : Node) (rec : Ω → Record) :
    Pace Ω where
  credited := node.windows - 1
  first ω g := firstPass (groupsBy sum node (rec ω)) g
  released ω g := released (groupsBy sum node (rec ω)) node.windows g
  ended ω g := ended (groupsBy sum node (rec ω)) g

/-- `ofRecord` at a verifier's verdict `ver`: the trace over the registration's blocks, accepting where `ver` does. -/
def traceBy {Ω : Type} (T : Trace Ω) (node : Node) (ver : Node → Record → Prop) (rec : Ω → Record) : Trace Ω :=
  { overBlocks T node.blocks with accepts := fun ω => ver node (rec ω) }

/-- **`LineHeld` at a verifier**: its node, trace, records, premises and bound verbatim, with `ver` for the verdict in
the event and the trace (`Verified`) and `sum` for the record the line and its paced session read
(`Record.summary`). `LineHeld` is it at `Verified` and `Record.summary`. -/
def LineHeldBy (ver : Node → Record → Prop) (sum : Node → Record → Option Poub.Session.Summary) : Prop :=
  ∀ (node : Node) (κ : ℕ) (X Cm : Type) [Fintype X] (prf : Fin node.samples → Pous.Bits κ → Blocks receiptMax)
    (rest : SampleCoins node.samples κ X → Fin node.samples → ℕ → ℕ) (recs : Cm → SampleCoins node.samples κ X → Record)
    (correct : Cm → Fin node.samples → SampleCoins node.samples κ X → Finset ℕ)
    (com : SampleCoins node.samples κ X → Cm)
    (T : Trace (SampleCoins node.samples κ X)) (R : Receipts node.samples κ X) (δs δnet : ℝ),
    let r := played recs com
    let F := recordCommitted node recs correct
    ∀ (O : Outside (traceBy T node ver r)) (I : Ingress (SampleCoins node.samples κ X)),
    Paced (traceBy T node ver r) (paceBy sum node r) → Covers (traceBy T node ver r) →
    SketchBoundAt (traceBy T node ver r) O δs → Covered (traceBy T node ver r) O I →
    IngressCounted (traceBy T node ver r) I δnet → (∀ c, Live (F.view c)) →
    Refused (F.play com) prf rest (traceBy T node ver r) R → Counted (F.play com) (traceBy T node ver r) →
    Pouw.pr (fun ω => ver node (r ω) ∧
        ¬ (Quiet (traceBy T node ver r) I.counted ω → ReadBy sum (traceBy T node ver r) ω node (r ω))) ≤
      δs + δnet + ((node.delta : ℝ) + ((∑ _t : Fin node.samples, node.params.samples : ℕ) : ℝ) / 2 ^ 256 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t)
          ((F.play (Committed.junk com t)).distinguisher prf (node.delta : ℝ) t)).toReal +
        (∑ t, F.hide com prf (node.delta : ℝ) t).toReal + (R.binding (F.play com) prf rest).toReal)

/-- **`LineHeldAtSubprotocol` at a verifier**: its node, trace, records, premises and bound verbatim, with `acc` for the
component's acceptance in the event (`Plays`), the trace's verdict `ver` (`Verified`) and the record the line and its
paced session read `sum` (`Record.summary`). `LineHeldAtSubprotocol`
(`Properties.Poub.Remote.SubprotocolDeployed`) is it at `Plays`, `Verified` and `Record.summary`. -/
def LineHeldAtSubprotocolBy (acc : Node → Record → ByteArray → ByteArray → ByteArray → Prop)
    (ver : Node → Record → Prop) (sum : Node → Record → Option Poub.Session.Summary) : Prop :=
  ∀ (node : Node) (κ : ℕ) (X Cm : Type) [Fintype X] (prf : Fin node.samples → Pous.Bits κ → Blocks receiptMax)
    (rest : SampleCoins node.samples κ X → Fin node.samples → ℕ → ℕ) (recs : Cm → SampleCoins node.samples κ X → Record)
    (correct : Cm → Fin node.samples → SampleCoins node.samples κ X → Finset ℕ)
    (com : SampleCoins node.samples κ X → Cm)
    (T : Trace (SampleCoins node.samples κ X)) (R : Receipts node.samples κ X) (δs δnet : ℝ)
    (a b e : SampleCoins node.samples κ X → ByteArray),
    let r := played recs com
    let F := recordCommitted node recs correct
    ∀ (O : Outside (traceBy T node ver r)) (I : Ingress (SampleCoins node.samples κ X)),
    Paced (traceBy T node ver r) (paceBy sum node r) → Covers (traceBy T node ver r) →
    SketchBoundAt (traceBy T node ver r) O δs → Covered (traceBy T node ver r) O I →
    IngressCounted (traceBy T node ver r) I δnet → (∀ c, Live (F.view c)) →
    Refused (F.play com) prf rest (traceBy T node ver r) R → Counted (F.play com) (traceBy T node ver r) →
    Pouw.pr (fun ω => acc node (r ω) (a ω) (b ω) (e ω) ∧
        ¬ (Quiet (traceBy T node ver r) I.counted ω → ReadBy sum (traceBy T node ver r) ω node (r ω))) ≤
      δs + δnet + ((node.delta : ℝ) + ((∑ _t : Fin node.samples, node.params.samples : ℕ) : ℝ) / 2 ^ 256 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t)
          ((F.play (Committed.junk com t)).distinguisher prf (node.delta : ℝ) t)).toReal +
        (∑ t, F.hide com prf (node.delta : ℝ) t).toReal + (R.binding (F.play com) prf rest).toReal)

/-- Each sample's receipts `U_c` in a verifier's record `sum node r` of record `r` on the node, none where it has none:
`Record.receipts` at `sum = Record.summary`. -/
def receiptsBy (sum : Node → Record → Option Poub.Session.Summary) (node : Node) (r : Record) : Array ℕ :=
  match sum node r with
  | some sm => sampleReceipts node.groupsPerCoin sm.windows
  | none => #[]

/-- What a published `q_bytes` says of the trace at `ω`, at a verifier's record `sum`: the figure and the credited
passes' wrong receipts of registered blocks (`Trace.incorrect`), a `BLOCK_BYTES` block each, are together at most the
receipts the record holds (`receiptsBy`), a block each. `QBytesAt` at `sum = Record.summary`. -/
def QBytesBy (sum : Node → Record → Option Poub.Session.Summary) {Ω : Type} (T : Trace Ω) (ω : Ω) (node : Node)
    (r : Record) (pub : Poub.Protocol.Public) : Prop :=
  pub.qBytes + Poub.Codec.BLOCK_BYTES * T.incorrect ω ≤
    Poub.Codec.BLOCK_BYTES * (receiptsBy sum node r).toList.sum

/-- **`QBytesAtSubprotocol` at a verifier**: its node, coins, records, premises and bound verbatim, with `acc` for the
subprotocol's acceptance with its public outputs named (`Publishes`), `ver` for the trace's verdict (`Verified`) and
`sum` for the record the receipts are read off (`Record.summary`). Its event is `∃ pub, acc … pub ∧ ¬ QBytesBy … pub`,
written as the acceptance and then what fails. `QBytesAtSubprotocol`
(`Properties.Poub.Remote.SubprotocolDeployed`) is it at `Publishes`, `Verified` and `Record.summary`. -/
def QBytesAtSubprotocolBy
    (acc : Node → Record → ByteArray → ByteArray → ByteArray → Poub.Protocol.Public → Prop)
    (ver : Node → Record → Prop) (sum : Node → Record → Option Poub.Session.Summary) : Prop :=
  ∀ (node : Node) (κ : ℕ) (X Cm : Type) [Fintype X] (prf : Fin node.samples → Pous.Bits κ → Blocks receiptMax)
    (rest : SampleCoins node.samples κ X → Fin node.samples → ℕ → ℕ) (recs : Cm → SampleCoins node.samples κ X → Record)
    (correct : Cm → Fin node.samples → SampleCoins node.samples κ X → Finset ℕ)
    (com : SampleCoins node.samples κ X → Cm)
    (T : Trace (SampleCoins node.samples κ X)) (R : Receipts node.samples κ X)
    (a b e : SampleCoins node.samples κ X → ByteArray),
    let r := played recs com
    let F := recordCommitted node recs correct
    (∀ c, Live (F.view c)) →
    Refused (F.play com) prf rest (traceBy T node ver r) R → Counted (F.play com) (traceBy T node ver r) →
    Pouw.pr (fun ω => (∃ pub, acc node (r ω) (a ω) (b ω) (e ω) pub) ∧
        ¬ ∀ pub, acc node (r ω) (a ω) (b ω) (e ω) pub → QBytesBy sum (traceBy T node ver r) ω node (r ω) pub) ≤
      (Poub.Protocol.DELTA : ℝ) + ((∑ _t : Fin node.samples, node.params.samples : ℕ) : ℝ) / 2 ^ 256 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t)
          ((F.play (Committed.junk com t)).distinguisher prf (Poub.Protocol.DELTA : ℝ) t)).toReal +
        (∑ t, F.hide com prf (Poub.Protocol.DELTA : ℝ) t).toReal + (R.binding (F.play com) prf rest).toReal

/-- The deployed period's parameters (`period.run`'s defaults): two windows of 100 ms, one pass each, the audit's
33 273 samples a sample, a 500 ms deadline, the lifetime cap of 300 windows, unpaced and not deferred, one window a
sample, from clock 0. -/
def honestParams : Poub.Protocol.Params where
  notBeforeNs := ((0 : ℕ) : Lean.Json)
  params := Lean.Json.mkObj [("deadline_ns", (500000000 : ℕ)), ("deferred", .bool false), ("lifetime", (300 : ℕ)),
    ("min_passes", (1 : ℕ)), ("pace", .null), ("samples", (33273 : ℕ)), ("window_ns", (100000000 : ℕ)),
    ("windows", (2 : ℕ))]

/-- The log of `honestParams`' session before its first event: its two challenges and two openings coins, 32 bytes
each, no append coin and no event. -/
def honestEvidence : Poub.Protocol.Evidence where
  appendCoins := .arr #[]
  coins := .arr (Array.replicate 4 (.str "0000000000000000000000000000000000000000000000000000000000000000"))
  events := #[]

/-- **An honest node's trace**: every registered block in node memory at every tick, none on a chip, nothing rebuilt
or read in from its disks or links, every pass planning and receipting every block correctly, at `passes` credited
passes on the clock `tick`. -/
def honestTrace {Ω : Type} (node : Node) (passes : ℕ) (tick : Ω → ℕ → ℕ) : Trace Ω where
  blocks := node.blocks
  passes := passes
  accepts _ := True
  tick := tick
  plan _ _ := Finset.univ
  correct _ _ := Finset.univ
  mem _ _ := Finset.univ
  chip _ _ := ∅
  rebuilt _ _ := 0
  disk _ _ := 0
  link _ _ := 0

/-- No device outside the node computes for it. -/
def noOutside {Ω : Type} (T : Trace Ω) : Outside T := ⟨fun _ _ => ∅, fun _ _ => ∅⟩

/-- Nothing but the verifier's messages enters the node, and the certifiers count nothing else. -/
def quietIngress (Ω : Type) : Ingress Ω := ⟨fun _ _ => 0, fun _ _ => 0⟩

/-- **An honest node's schedule**: the session has a window; its passes take a tick or more each; each sample's
receipts are the store's, the same at every commitment and every coin; and wherever the Lean verifier accepts, the
auditor saw every window, so the line's groups are the session's windows, each of one pass or more with its last root
no sooner than its coin, the coins in order, `passes` at least the session's passes, and each pass run inside its
group's span `[s_g, e_g]` on the record's clock. -/
def HonestSchedule {Ω Cm : Type} (node : Node) (passes : ℕ) (tick : Ω → ℕ → ℕ) (recs : Cm → Ω → Record)
    (com : Ω → Cm) : Prop :=
  1 ≤ node.windows ∧ (∀ ω p, tick ω p < tick ω (p + 1)) ∧
    (∀ c c' ω ω' t, (recs c ω).sampleReceiptsAt node t = (recs c' ω').sampleReceiptsAt node t) ∧
    ∀ ω, Verified node (played recs com ω) → ∀ gs, (played recs com ω).groups node = gs →
      gs.size = node.windows ∧
      (∀ g < node.windows, 1 ≤ gs[g]!.passes ∧ gs[g]!.releasedNs ≤ gs[g]!.lastNs) ∧
      (∀ g h, g ≤ h → h < node.windows → gs[g]!.releasedNs ≤ gs[h]!.releasedNs) ∧
      firstPass gs node.windows ≤ passes ∧
      ∀ g < node.windows, ∀ p, firstPass gs g ≤ p → p < firstPass gs (g + 1) →
        released gs node.windows g ≤ tick ω p ∧ tick ω (p + 1) ≤ ended gs g

namespace Toy

/-- A window of one receipt over one block, which it sampled. -/
def win : Poub.Session.WindowSummary where
  lengths := #[1]
  steps := #[]
  post := #[]
  heads := #[]
  stale := #[]
  arrivedNs := #[]
  releasedNs := 0
  deadlineNs := 0
  openedNs := 0
  drawn := 1

/-- The record the nontrivial witnesses' verifier that accepts everything reads of every record: four windows of `win`
over one block, nothing appended or released, a lifetime of one window, unpaced. -/
def sm : Poub.Session.Summary where
  windows := #[win, win, win, win]
  blocks := 1
  appended := 0
  released := 0
  lifetime := 1
  pace := none
  appends := none

end Toy

end Properties.Poub.Subprotocol
