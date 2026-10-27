import RemoteAudit.Protocol
import Properties.RemoteAudit.Line

/-!
# The bandwidth component accepts only what the memory line is stated on (`RemoteAudit.Protocol`)

Onsite's bandwidth component is `RemoteAudit.Protocol.protocol` (`verity/remote-bandwidth/v0`): its check replays the
verifier's log of a session through `Shared.run g` at the session's windows a draw g (`groups_per_coin`; `Session.run`
at g = 1) and accepts once the session does, publishing `certify` of the run's record at g. The composition reads of
it that the check accepts three byte strings (`Properties.Composition.DecodedAccepted` at the component, which
`Verity.ComponentAccepts` reads); the remote audit's guarantees are stated on `Shared.run` at the node's g accepting a
record's session (`Properties.RemoteAudit.Line.Verified`). The two properties join them:

* **AcceptIsVerified**, the refinement: what the component's check accepts is a session the parameters and the log
  decode to (`RemoteAudit.Protocol.sessionOf`: its parameters, its g, coins, append coins, clock floor and events),
  which `Shared.run g` runs from `State.init` to an accepted state with a record `sm`, and the public outputs are
  `certify sm g`'s, the reads that record's openings certify and their error. The check refuses what `sessionOf`
  refuses, so what it accepts is `Shared.run g`'s, never `Flight.run`'s at one window in flight, which no property
  covers.
* **LineHeldAtComponent**: `Properties.RemoteAudit.Line.LineHeld` with the component's acceptance in the event in place
  of `Verified`. On the same node, trace, records and premises, and the bytes `a ω`, `b ω` and `e ω` the component is
  given at each outcome (the parameters, the commitment and the evidence), the probability that the component accepts
  bytes that play the record of `ω` on the node (`Plays`) while the record's link-charged line credits a stretch more
  than the trace held over its span is at most `LineHeld`'s bound, term for term.

`Plays` is the component's acceptance (`DecodedAccepted` unfolded at `RemoteAudit.Protocol.protocol`, whose decoders
are `decodeParams`, `decodeCommitment` and `decodeEvidence`) with what picks out the record the line reads: the
commitment decodes to the node's registration, and the parameters and the log to the node's session parameters and
windows a draw and the record's coins, append coins, clock floor and events. It is in the event, not a hypothesis: a log
that is not the record's is an outcome the bound counts nothing for, not one the statement assumes away.

Each property is its verifier's instance of a form over the verifier, so that its witnesses can say what the verifier
contributes. `AcceptIsVerifiedFor chk` is `AcceptIsVerified` at a check `chk`, and `LineHeldAtComponentBy acc ver sum`
is `LineHeldAtComponent` at an acceptance `acc` in the event, a verdict `ver` the trace reads and a record `sum` the
line reads: at `RemoteAudit.Protocol.check`, and at `Plays`, `Verified` and `Record.summary`, they are the two
properties. The honest instance the witnesses read is the deployed period's parameters with no event yet
(`honestParams`, `honestEvidence`) and a node that holds its whole store in memory throughout (`honestTrace`), with no
outside device (`noOutside`), on a schedule that runs each pass inside its window (`HonestSchedule`).
-/

namespace Properties.RemoteAudit.Component

open Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Draw
  Accounting.Remote.Draw.Assumptions Properties.RemoteAudit.Line

/-- **The component's check accepts only what `Shared.run g` accepts.** Whenever `RemoteAudit.Protocol.check` accepts
parameters `p`, a registration `reg` and a log `e` with public outputs `pub`: `p` and `e` decode to a session
(`sessionOf`: its parameters, its windows a draw g, coins, append coins, clock floor and events), `Shared.run g` from
`State.init` at those and `reg` runs every event to a state that ended accepted with a record `sm`, and `pub` is what
`certify` publishes of `sm` at g. -/
def AcceptIsVerified : Prop :=
  ∀ (p : RemoteAudit.Protocol.Params) (reg : RemoteAudit.Registration) (e : RemoteAudit.Protocol.Evidence)
    (pub : RemoteAudit.Protocol.Public) (prof : Lean.Json),
    RemoteAudit.Protocol.check p reg e = .accept pub prof →
    ∃ (params : RemoteAudit.Session.Params) (g : ℕ) (coins appendCoins : Array ByteArray) (notBeforeNs : ℕ)
      (events : Array RemoteAudit.Session.Event) (rs : Array RemoteAudit.Session.Result)
      (s : RemoteAudit.Session.State) (sm : RemoteAudit.Session.Summary),
      RemoteAudit.Protocol.sessionOf p e = .ok (params, g, coins, appendCoins, notBeforeNs, events) ∧
      RemoteAudit.Shared.run g (RemoteAudit.Session.State.init params reg coins appendCoins notBeforeNs) events =
        .ok (rs, s) ∧
      s.outcome = some "accepted" ∧ s.summary = some sm ∧ pub = (RemoteAudit.Protocol.certify sm g).1

/-- **The component accepts bytes that play the record on the node.** The parameters `a` (canonical JSON), the
commitment `b` and the evidence `e` decode under the component's decoders, its check accepts them
(`Properties.Composition.DecodedAccepted` at `RemoteAudit.Protocol.protocol`), the commitment is the node's
registration, and the session the parameters and the log decode to (`RemoteAudit.Protocol.sessionOf`) is the node's
session parameters and windows a draw with the record's coins, append coins, clock floor and events. -/
def Plays (node : Node) (rec : Record) (a b e : ByteArray) : Prop :=
  ∃ (p : RemoteAudit.Protocol.Params) (ev : RemoteAudit.Protocol.Evidence),
    (Core.canonicalJson a >>= RemoteAudit.Protocol.decodeParams) = .ok p ∧
    RemoteAudit.Protocol.decodeCommitment b = .ok node.reg ∧ RemoteAudit.Protocol.decodeEvidence e = .ok ev ∧
    RemoteAudit.Protocol.sessionOf p ev =
      .ok (node.params, node.groupsPerCoin, rec.coins, rec.appendCoins, rec.notBeforeNs, rec.events) ∧
    ∃ pub prof, RemoteAudit.Protocol.check p node.reg ev = .accept pub prof

/-- **The memory line of record, link-charged, at the bandwidth component.** `LineHeld`'s node, trace, records and
premises verbatim, one coin for each of the session's draws (`Node.draws`), and the bytes the component is given at
each outcome (`a`, `b`, `e`: the parameters, the commitment and the evidence): the probability that the component
accepts bytes that play the record of `ω` on the node (`Plays`) while a stretch of the record's link-charged line
credits more than the trace held over its span is at most `δs + (δ + Σ_c s / 2^64 + Σ_c adv_c + Σ_c hide_c + bind)`,
`LineHeld`'s bound. -/
def LineHeldAtComponent : Prop :=
  ∀ (node : Node) (κ : ℕ) (X Cm : Type) [Fintype X] (prf : Fin node.draws → Pous.Bits κ → Blocks receiptMax)
    (rest : Coins node.draws κ X → Fin node.draws → ℕ → ℕ) (recs : Cm → Coins node.draws κ X → Record)
    (correct : Cm → Fin node.draws → Coins node.draws κ X → Finset ℕ) (com : Coins node.draws κ X → Cm)
    (T : Trace (Coins node.draws κ X)) (R : Receipts node.draws κ X) (δs : ℝ)
    (a b e : Coins node.draws κ X → ByteArray),
    let r := played recs com
    let F := recordCommitted node recs correct
    ∀ O : Outside (ofRecord T node r),
    Links (ofRecord T node r) node.link node.linkBurst →
    SquaringCeiling (ofRecord T node r) node.squarings node.encode node.rebuildBurst →
    Disks (ofRecord T node r) node.disk node.diskBurst → ChipCeiling (ofRecord T node r) node.chipBlocks →
    Routed (ofRecord T node r) → Paced (ofRecord T node r) (recordPace node r) → Covers (ofRecord T node r) →
    SketchBoundAt (ofRecord T node r) O δs → NoOutsideCompute (ofRecord T node r) O → (∀ c, Live (F.view c)) →
    Refused (F.play com) prf rest (ofRecord T node r) R → Counted (F.play com) (ofRecord T node r) →
    Pouw.pr (fun ω => Plays node (r ω) (a ω) (b ω) (e ω) ∧ ¬ HeldAt (ofRecord T node r) ω node (r ω) 1) ≤
      δs + ((node.delta : ℝ) + ((∑ _t : Fin node.draws, node.params.samples : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t)
          ((F.play (Committed.junk com t)).distinguisher prf (node.delta : ℝ) t)).toReal +
        (∑ t, F.hide com prf (node.delta : ℝ) t).toReal + (R.binding (F.play com) prf rest).toReal)

/-- **`AcceptIsVerified` at a check `chk`**: whenever `chk` accepts parameters `p`, a registration `reg` and a log `e`
with public outputs `pub`, they decode to a session that `Shared.run g` runs to an accepted state with a record `sm`,
and `pub` is what `certify` publishes of `sm` at g. `AcceptIsVerified` is it at `RemoteAudit.Protocol.check`. -/
def AcceptIsVerifiedFor (chk : RemoteAudit.Protocol.Params → RemoteAudit.Registration → RemoteAudit.Protocol.Evidence →
    Core.Outcome RemoteAudit.Protocol.Public Lean.Json) : Prop :=
  ∀ (p : RemoteAudit.Protocol.Params) (reg : RemoteAudit.Registration) (e : RemoteAudit.Protocol.Evidence)
    (pub : RemoteAudit.Protocol.Public) (prof : Lean.Json),
    chk p reg e = .accept pub prof →
    ∃ (params : RemoteAudit.Session.Params) (g : ℕ) (coins appendCoins : Array ByteArray) (notBeforeNs : ℕ)
      (events : Array RemoteAudit.Session.Event) (rs : Array RemoteAudit.Session.Result)
      (s : RemoteAudit.Session.State) (sm : RemoteAudit.Session.Summary),
      RemoteAudit.Protocol.sessionOf p e = .ok (params, g, coins, appendCoins, notBeforeNs, events) ∧
      RemoteAudit.Shared.run g (RemoteAudit.Session.State.init params reg coins appendCoins notBeforeNs) events =
        .ok (rs, s) ∧
      s.outcome = some "accepted" ∧ s.summary = some sm ∧ pub = (RemoteAudit.Protocol.certify sm g).1

/-- The groups the line reads of record `r` on the node, when a verifier records it as `sum node r`: `Record.groups`
at `sum = Record.summary`. -/
def groupsBy (sum : Node → Record → Option RemoteAudit.Session.Summary) (node : Node) (r : Record) :
    Array _root_.RemoteAudit.Line.Group :=
  match sum node r with
  | some sm => match _root_.RemoteAudit.Line.ofSummaryAt node.groupsPerCoin sm node.params.samples r.seen with
    | .ok (gs, _) => gs
    | .error _ => #[]
  | none => #[]

/-- The line of record `r` on the node, when a verifier records it as `sum node r`: `lineOf` at
`sum = Record.summary`. -/
def lineBy (sum : Node → Record → Option RemoteAudit.Session.Summary) (node : Node) (r : Record) :
    Except String _root_.RemoteAudit.Line.Line :=
  match sum node r with
  | some sm => _root_.RemoteAudit.Line.lineOfRecordAt node.groupsPerCoin sm node.params.samples r.seen node.per
      node.blockBytes node.chipBytes node.nodeBytes node.delta node.rates
  | none => .error "the Lean verifier gave the session no record"

/-- `HeldAt` of the line a verifier's record `sum` gives: every stretch of form `form`, if the line computes one,
credits what the trace held at `ω`. `HeldAt` at `sum = Record.summary`. -/
def HeldBy (sum : Node → Record → Option RemoteAudit.Session.Summary) {Ω : Type} (T : Trace Ω) (ω : Ω) (node : Node)
    (r : Record) (form : ℕ) : Prop :=
  ∀ l, lineBy sum node r = .ok l → ∀ f, l.forms[form]? = some f → ∀ x ∈ f.stretches,
    Credits T ω (groupsBy sum node r) node.per node.blockBytes x

/-- `recordPace` of the groups a verifier's record `sum` gives. -/
def paceBy {Ω : Type} (sum : Node → Record → Option RemoteAudit.Session.Summary) (node : Node) (rec : Ω → Record) :
    Pace Ω where
  credited := node.windows - 1
  first ω g := firstPass (groupsBy sum node (rec ω)) g
  released ω g := released (groupsBy sum node (rec ω)) node.windows g
  ended ω g := ended (groupsBy sum node (rec ω)) g

/-- `ofRecord` at a verifier's verdict `ver`. -/
def traceBy {Ω : Type} (T : Trace Ω) (node : Node) (ver : Node → Record → Prop) (rec : Ω → Record) : Trace Ω :=
  { overBlocks T node.blocks with accepts := fun ω => ver node (rec ω) }

/-- **`LineHeldAtComponent` at a verifier**: its node, trace, records, premises and bound verbatim, with `acc` for the
component's acceptance in the event (`Plays`), the trace's verdict `ver` (`Verified`) and the record the line and its
paced session read `sum` (`Record.summary`). `LineHeldAtComponent` is it at `Plays`, `Verified` and
`Record.summary`. -/
def LineHeldAtComponentBy (acc : Node → Record → ByteArray → ByteArray → ByteArray → Prop)
    (ver : Node → Record → Prop) (sum : Node → Record → Option RemoteAudit.Session.Summary) : Prop :=
  ∀ (node : Node) (κ : ℕ) (X Cm : Type) [Fintype X] (prf : Fin node.draws → Pous.Bits κ → Blocks receiptMax)
    (rest : Coins node.draws κ X → Fin node.draws → ℕ → ℕ) (recs : Cm → Coins node.draws κ X → Record)
    (correct : Cm → Fin node.draws → Coins node.draws κ X → Finset ℕ) (com : Coins node.draws κ X → Cm)
    (T : Trace (Coins node.draws κ X)) (R : Receipts node.draws κ X) (δs : ℝ)
    (a b e : Coins node.draws κ X → ByteArray),
    let r := played recs com
    let F := recordCommitted node recs correct
    ∀ O : Outside (traceBy T node ver r),
    Links (traceBy T node ver r) node.link node.linkBurst →
    SquaringCeiling (traceBy T node ver r) node.squarings node.encode node.rebuildBurst →
    Disks (traceBy T node ver r) node.disk node.diskBurst → ChipCeiling (traceBy T node ver r) node.chipBlocks →
    Routed (traceBy T node ver r) → Paced (traceBy T node ver r) (paceBy sum node r) →
    Covers (traceBy T node ver r) → SketchBoundAt (traceBy T node ver r) O δs →
    NoOutsideCompute (traceBy T node ver r) O → (∀ c, Live (F.view c)) →
    Refused (F.play com) prf rest (traceBy T node ver r) R → Counted (F.play com) (traceBy T node ver r) →
    Pouw.pr (fun ω => acc node (r ω) (a ω) (b ω) (e ω) ∧ ¬ HeldBy sum (traceBy T node ver r) ω node (r ω) 1) ≤
      δs + ((node.delta : ℝ) + ((∑ _t : Fin node.draws, node.params.samples : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t)
          ((F.play (Committed.junk com t)).distinguisher prf (node.delta : ℝ) t)).toReal +
        (∑ t, F.hide com prf (node.delta : ℝ) t).toReal + (R.binding (F.play com) prf rest).toReal)

/-- The deployed period's parameters (`period.run`'s defaults): two windows of 100 ms, one pass each, the audit's
33 273 samples a draw, a 500 ms deadline, the lifetime cap of 300 windows, unpaced and not deferred, one window a
draw, from clock 0. -/
def honestParams : RemoteAudit.Protocol.Params where
  notBeforeNs := ((0 : ℕ) : Lean.Json)
  params := Lean.Json.mkObj [("deadline_ns", (500000000 : ℕ)), ("deferred", .bool false), ("lifetime", (300 : ℕ)),
    ("min_passes", (1 : ℕ)), ("pace", .null), ("samples", (33273 : ℕ)), ("window_ns", (100000000 : ℕ)),
    ("windows", (2 : ℕ))]

/-- The log of `honestParams`' session before its first event: its two challenges and two openings coins, 32 bytes
each, no append coin and no event. -/
def honestEvidence : RemoteAudit.Protocol.Evidence where
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

/-- No device outside the node computes on anything. -/
def noOutside {Ω : Type} (T : Trace Ω) : Outside T := ⟨fun _ _ => ∅, fun _ _ => ∅⟩

/-- **An honest node's schedule**: the session has a window; its passes take a tick or more each; each draw's
receipts are the store's, the same at every commitment and every coin; and wherever the Lean verifier accepts, the
auditor saw every window, so the line's groups are the session's windows, each of one pass or more with its last root
no sooner than its coin, the coins in order, `passes` at least the session's passes, and each pass run inside its
group's span `[s_g, e_g]` on the record's clock. -/
def HonestSchedule {Ω Cm : Type} (node : Node) (passes : ℕ) (tick : Ω → ℕ → ℕ) (recs : Cm → Ω → Record)
    (com : Ω → Cm) : Prop :=
  1 ≤ node.windows ∧ (∀ ω p, tick ω p < tick ω (p + 1)) ∧
    (∀ c c' ω ω' t, (recs c ω).drawReceiptsAt node t = (recs c' ω').drawReceiptsAt node t) ∧
    ∀ ω, Verified node (played recs com ω) → ∀ gs, (played recs com ω).groups node = gs →
      gs.size = node.windows ∧
      (∀ g < node.windows, 1 ≤ gs[g]!.passes ∧ gs[g]!.releasedNs ≤ gs[g]!.lastNs) ∧
      (∀ g h, g ≤ h → h < node.windows → gs[g]!.releasedNs ≤ gs[h]!.releasedNs) ∧
      firstPass gs node.windows ≤ passes ∧
      ∀ g < node.windows, ∀ p, firstPass gs g ≤ p → p < firstPass gs (g + 1) →
        released gs node.windows g ≤ tick ω p ∧ tick ω (p + 1) ≤ ended gs g

end Properties.RemoteAudit.Component
