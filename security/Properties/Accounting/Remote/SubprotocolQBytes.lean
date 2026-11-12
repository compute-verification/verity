module

public import Properties.Accounting.Remote.Subprotocol

/-!
# The reads the bandwidth component publishes, at its acceptance (`RemoteAudit.Protocol.certify`)

On acceptance the bandwidth component (`RemoteAudit.Protocol.protocol`) publishes `q_bytes`
(`RemoteAudit.Protocol.Public.qBytes`, `certify`): the receipts `U` the Lean verifier's record of the session holds,
less the wrong ones its openings leave, `W` rounded up (`RemoteAudit.Pooled.pooled` of each draw's receipts and draws
at `RemoteAudit.Protocol.DELTA`), a `RemoteAudit.Codec.BLOCK_BYTES` block each, and 0 where the pooled bound refuses
the draws. `QBytesAtSubprotocol` states what that figure says of the trace that produced the record: its credited
passes' wrong receipts are no more than the figure took off.

* **QBytesAtSubprotocol**: `LineHeldAtSubprotocol`'s node, coins, records read through the coin commitment, trace of
  record and bytes verbatim, and of its premises the draw's three alone (`Live` at every fixed coin commitment,
  `Refused` at the receipts `R` and `Counted`, of the session the prover plays holding `com`). The probability that
  the component accepts bytes that play the record of `ω` on the node with public outputs `pub` (`Publishes`) while
  `pub.qBytes` and the trace's wrong receipts, a block each, together exceed the record's receipts (`QBytesAt`:
  `q_bytes + BLOCK_BYTES · wrong ≤ BLOCK_BYTES · U` fails) is at most
  `δ + Σ_c s / 2^64 + Σ_c adv_c + Σ_c hide_c + bind` at `δ = RemoteAudit.Protocol.DELTA`, the δ `certify` reads its
  pooled bound at. That is the draw's part of `LineHeldAtSubprotocol`'s bound, term for term:
  `Accounting.Guarantees.Remote.PooledSample`'s, with the hiding terms of a session read through the coin commitment
  (`Committed.hide`, 0 where the records and their correct receipts read no commitment). Its first two terms are the
  `eps` the component publishes (`PublishedEps`); the PRF advantages, the hiding terms and the receipts' binding term
  are in no published figure.

It takes none of the memory line's other premises (`Paced`, `Covers`, `SketchBoundAt`, `Covered`, `IngressCounted`)
and says less than the line. It counts the trace's wrong receipts against the figure. It does not say that a correct
receipt's block was in node memory or on a chip while its pass ran (`Trace.reads`, which `LineHeld` has under
`SketchBoundAt` and the certifiers' premises), nor that the record's `U` receipts are the registered blocks the trace's
credited passes planned (`Trace.planned`, which the line has from `Paced` and `Covers`): `U` is the Lean verifier's own
count, read off its record (`Record.receipts`).

`Publishes` is `Plays` with the public outputs named, so `Plays` is `Publishes` at some `pub`. As the component's
other properties, the statement is its verifier's instance of a form over the verifier: `QBytesAtSubprotocolBy acc ver
sum` is it at an acceptance `acc` that names the public outputs, a verdict `ver` the trace reads and a record `sum` the
receipts are read off, and `QBytesAtSubprotocol` is it at `Publishes`, `Verified` and `Record.summary`.

Both are statements only: `security/Proofs` holds no proof of either, and no lock lists them.
-/

@[expose] public section

namespace Properties.RemoteAudit.Subprotocol

open Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Draw
  Accounting.Remote.Draw.Assumptions Properties.RemoteAudit.Line

/-- **The component accepts bytes that play the record on the node, publishing `pub`.** `Plays` with its public
outputs named: the parameters `a` (canonical JSON), the commitment `b` and the evidence `e` decode under the
component's decoders, the commitment is the node's registration, the session the parameters and the log decode to
(`RemoteAudit.Protocol.sessionOf`) is the node's session parameters and windows a draw with the record's coins, append
coins, clock floor and events, and the component's check accepts them with public outputs `pub`. -/
def Publishes (node : Node) (rec : Record) (a b e : ByteArray) (pub : RemoteAudit.Protocol.Public) : Prop :=
  ∃ (p : RemoteAudit.Protocol.Params) (ev : RemoteAudit.Protocol.Evidence),
    (Core.canonicalJson a >>= RemoteAudit.Protocol.decodeParams) = .ok p ∧
    RemoteAudit.Protocol.decodeCommitment b = .ok node.reg ∧ RemoteAudit.Protocol.decodeEvidence e = .ok ev ∧
    RemoteAudit.Protocol.sessionOf p ev =
      .ok (node.params, node.groupsPerCoin, rec.coins, rec.appendCoins, rec.notBeforeNs, rec.events) ∧
    ∃ prof, RemoteAudit.Protocol.check p node.reg ev = .accept pub prof

/-- What a published `q_bytes` says of the trace at `ω`: the figure and the credited passes' wrong receipts of
registered blocks (`Trace.incorrect`), a `BLOCK_BYTES` block each, are together at most the receipts the Lean
verifier's record of the session holds (`Record.receipts`, its draws' `U_c` summed), a block each. Of the record's `U`
receipts, the passes got at most `U − q_bytes / BLOCK_BYTES` wrong. -/
def QBytesAt {Ω : Type} (T : Trace Ω) (ω : Ω) (node : Node) (r : Record) (pub : RemoteAudit.Protocol.Public) : Prop :=
  pub.qBytes + RemoteAudit.Codec.BLOCK_BYTES * T.incorrect ω ≤
    RemoteAudit.Codec.BLOCK_BYTES * (r.receipts node).toList.sum

/-- **The reads the bandwidth component publishes, at its acceptance.** `LineHeldAtSubprotocol`'s node, coins, records,
trace of record and bytes verbatim (`a`, `b`, `e`: the parameters, the commitment and the evidence the component is
given at each outcome), under the draw's three premises alone: the probability that the component accepts bytes that
play the record of `ω` on the node with public outputs `pub` (`Publishes`) while `pub.qBytes` and the trace's wrong
receipts, a block each, together exceed the record's receipts (`QBytesAt`) is at most
`δ + Σ_c s / 2^64 + Σ_c adv_c + Σ_c hide_c + bind` at `δ = RemoteAudit.Protocol.DELTA`, the draw's part of
`LineHeldAtSubprotocol`'s bound at the δ `certify` reads its pooled bound at. -/
def QBytesAtSubprotocol : Prop :=
  ∀ (node : Node) (κ : ℕ) (X Cm : Type) [Fintype X] (prf : Fin node.samples → Pous.Bits κ → Blocks receiptMax)
    (rest : DrawCoins node.samples κ X → Fin node.samples → ℕ → ℕ) (recs : Cm → DrawCoins node.samples κ X → Record)
    (correct : Cm → Fin node.samples → DrawCoins node.samples κ X → Finset ℕ) (com : DrawCoins node.samples κ X → Cm)
    (T : Trace (DrawCoins node.samples κ X)) (R : Receipts node.samples κ X)
    (a b e : DrawCoins node.samples κ X → ByteArray),
    let r := played recs com
    let F := recordCommitted node recs correct
    (∀ c, Live (F.view c)) →
    Refused (F.play com) prf rest (ofRecord T node r) R → Counted (F.play com) (ofRecord T node r) →
    Pouw.pr (fun ω => ∃ pub, Publishes node (r ω) (a ω) (b ω) (e ω) pub ∧
        ¬ QBytesAt (ofRecord T node r) ω node (r ω) pub) ≤
      (RemoteAudit.Protocol.DELTA : ℝ) + ((∑ _t : Fin node.samples, node.params.samples : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t)
          ((F.play (Committed.junk com t)).distinguisher prf (RemoteAudit.Protocol.DELTA : ℝ) t)).toReal +
        (∑ t, F.hide com prf (RemoteAudit.Protocol.DELTA : ℝ) t).toReal + (R.binding (F.play com) prf rest).toReal

/-- Each draw's receipts in a verifier's record `sum node r` of record `r` on the node, none where it has none:
`Record.receipts` at `sum = Record.summary`. -/
def receiptsBy (sum : Node → Record → Option RemoteAudit.Session.Summary) (node : Node) (r : Record) : Array ℕ :=
  match sum node r with
  | some sm => sampleReceipts node.groupsPerCoin sm.windows
  | none => #[]

/-- `QBytesAt` of the receipts a verifier's record `sum` holds: the figure and the trace's wrong receipts, a block
each, are together at most them. `QBytesAt` at `sum = Record.summary`. -/
def QBytesBy (sum : Node → Record → Option RemoteAudit.Session.Summary) {Ω : Type} (T : Trace Ω) (ω : Ω) (node : Node)
    (r : Record) (pub : RemoteAudit.Protocol.Public) : Prop :=
  pub.qBytes + RemoteAudit.Codec.BLOCK_BYTES * T.incorrect ω ≤
    RemoteAudit.Codec.BLOCK_BYTES * (receiptsBy sum node r).toList.sum

/-- **`QBytesAtSubprotocol` at a verifier**: its node, coins, records, premises and bound verbatim, with `acc` for the
component's acceptance and the public outputs it names (`Publishes`), `ver` for the trace's verdict (`Verified`) and
`sum` for the record the receipts are read off (`Record.summary`). `QBytesAtSubprotocol` is it at `Publishes`,
`Verified` and `Record.summary`. -/
def QBytesAtSubprotocolBy
    (acc : Node → Record → ByteArray → ByteArray → ByteArray → RemoteAudit.Protocol.Public → Prop)
    (ver : Node → Record → Prop) (sum : Node → Record → Option RemoteAudit.Session.Summary) : Prop :=
  ∀ (node : Node) (κ : ℕ) (X Cm : Type) [Fintype X] (prf : Fin node.samples → Pous.Bits κ → Blocks receiptMax)
    (rest : DrawCoins node.samples κ X → Fin node.samples → ℕ → ℕ) (recs : Cm → DrawCoins node.samples κ X → Record)
    (correct : Cm → Fin node.samples → DrawCoins node.samples κ X → Finset ℕ) (com : DrawCoins node.samples κ X → Cm)
    (T : Trace (DrawCoins node.samples κ X)) (R : Receipts node.samples κ X)
    (a b e : DrawCoins node.samples κ X → ByteArray),
    let r := played recs com
    let F := recordCommitted node recs correct
    (∀ c, Live (F.view c)) →
    Refused (F.play com) prf rest (traceBy T node ver r) R → Counted (F.play com) (traceBy T node ver r) →
    Pouw.pr (fun ω => ∃ pub, acc node (r ω) (a ω) (b ω) (e ω) pub ∧
        ¬ QBytesBy sum (traceBy T node ver r) ω node (r ω) pub) ≤
      (RemoteAudit.Protocol.DELTA : ℝ) + ((∑ _t : Fin node.samples, node.params.samples : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t)
          ((F.play (Committed.junk com t)).distinguisher prf (RemoteAudit.Protocol.DELTA : ℝ) t)).toReal +
        (∑ t, F.hide com prf (RemoteAudit.Protocol.DELTA : ℝ) t).toReal + (R.binding (F.play com) prf rest).toReal

end Properties.RemoteAudit.Subprotocol
