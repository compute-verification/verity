import PoubCheck.Protocol
import Properties.PoubCheck.Line

/-!
# The bandwidth component accepts only what the memory line is stated on (`PoubCheck.Protocol`)

Onsite's bandwidth component is `PoubCheck.Protocol.protocol` (`verity/poub/v0`): its check replays the
verifier's log of a session through `Shared.run g` at the session's windows a draw g (`groups_per_coin`; `Session.run`
at g = 1) and accepts once the session does, publishing `certify` of the run's record at g. The composition reads of
it that the check accepts three byte strings (`Properties.Composition.DecodedAccepted` at the component, which
`Verity.ComponentAccepts` reads); PoUB's guarantees are stated on `Shared.run` at the node's g accepting a
record's session (`Properties.PoubCheck.Line.Verified`). The two properties join them:

* **AcceptIsVerified**, the refinement: what the component's check accepts is a session the parameters and the log
  decode to (`PoubCheck.Protocol.sessionOf`: its parameters, its g, coins, append coins, clock floor and events),
  which `Shared.run g` runs from `State.init` to an accepted state with a record `sm`, and the public outputs are
  `certify sm g`'s, the reads that record's openings certify and their error. The check refuses what `sessionOf`
  refuses, so what it accepts is `Shared.run g`'s, never `Flight.run`'s at one window in flight, which no property
  covers.
* **LineHeldAtComponent**: `Properties.PoubCheck.Line.LineHeld` with the component's acceptance in the event in place
  of `Verified`. On the same node, trace, records and premises, and the bytes `a ω`, `b ω` and `e ω` the component is
  given at each outcome (the parameters, the commitment and the evidence), the probability that the component accepts
  bytes that play the record of `ω` on the node (`Plays`) while the record's link-charged line credits a stretch more
  than the trace held over its span is at most `LineHeld`'s bound, term for term.

`Plays` is the component's acceptance (`DecodedAccepted` unfolded at `PoubCheck.Protocol.protocol`, whose decoders
are `decodeParams`, `decodeCommitment` and `decodeEvidence`) with what picks out the record the line reads: the
commitment decodes to the node's registration, and the parameters and the log to the node's session parameters and
windows a draw and the record's coins, append coins, clock floor and events. It is in the event, not a hypothesis: a log
that is not the record's is an outcome the bound counts nothing for, not one the statement assumes away.
-/

namespace Properties.PoubCheck.Component

open Accounting.Poub.Held Accounting.Poub.Held.Assumptions Accounting.Poub.Draw
  Accounting.Poub.Draw.Assumptions Properties.PoubCheck.Line

/-- **The component's check accepts only what `Shared.run g` accepts.** Whenever `PoubCheck.Protocol.check` accepts
parameters `p`, a registration `reg` and a log `e` with public outputs `pub`: `p` and `e` decode to a session
(`sessionOf`: its parameters, its windows a draw g, coins, append coins, clock floor and events), `Shared.run g` from
`State.init` at those and `reg` runs every event to a state that ended accepted with a record `sm`, and `pub` is what
`certify` publishes of `sm` at g. -/
def AcceptIsVerified : Prop :=
  ∀ (p : PoubCheck.Protocol.Params) (reg : PoubCheck.Registration) (e : PoubCheck.Protocol.Evidence)
    (pub : PoubCheck.Protocol.Public) (prof : Lean.Json),
    PoubCheck.Protocol.check p reg e = .accept pub prof →
    ∃ (params : PoubCheck.Session.Params) (g : ℕ) (coins appendCoins : Array ByteArray) (notBeforeNs : ℕ)
      (events : Array PoubCheck.Session.Event) (rs : Array PoubCheck.Session.Result)
      (s : PoubCheck.Session.State) (sm : PoubCheck.Session.Summary),
      PoubCheck.Protocol.sessionOf p e = .ok (params, g, coins, appendCoins, notBeforeNs, events) ∧
      PoubCheck.Shared.run g (PoubCheck.Session.State.init params reg coins appendCoins notBeforeNs) events =
        .ok (rs, s) ∧
      s.outcome = some "accepted" ∧ s.summary = some sm ∧ pub = (PoubCheck.Protocol.certify sm g).1

/-- **The component accepts bytes that play the record on the node.** The parameters `a` (canonical JSON), the
commitment `b` and the evidence `e` decode under the component's decoders, its check accepts them
(`Properties.Composition.DecodedAccepted` at `PoubCheck.Protocol.protocol`), the commitment is the node's
registration, and the session the parameters and the log decode to (`PoubCheck.Protocol.sessionOf`) is the node's
session parameters and windows a draw with the record's coins, append coins, clock floor and events. -/
def Plays (node : Node) (rec : Record) (a b e : ByteArray) : Prop :=
  ∃ (p : PoubCheck.Protocol.Params) (ev : PoubCheck.Protocol.Evidence),
    (Core.canonicalJson a >>= PoubCheck.Protocol.decodeParams) = .ok p ∧
    PoubCheck.Protocol.decodeCommitment b = .ok node.reg ∧ PoubCheck.Protocol.decodeEvidence e = .ok ev ∧
    PoubCheck.Protocol.sessionOf p ev =
      .ok (node.params, node.groupsPerCoin, rec.coins, rec.appendCoins, rec.notBeforeNs, rec.events) ∧
    ∃ pub prof, PoubCheck.Protocol.check p node.reg ev = .accept pub prof

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

end Properties.PoubCheck.Component
