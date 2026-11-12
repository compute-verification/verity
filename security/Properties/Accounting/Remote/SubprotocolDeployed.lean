module

public import Properties.Accounting.Remote.Subprotocol

/-!
# The bandwidth component's properties at the deployed verifier (`RemoteAudit.Protocol`)

The properties `Properties.Accounting.Remote.Subprotocol` describes, each a closed statement: its form over the
verifier there at the deployed one, `AcceptIsVerifiedFor` and `PublishedEpsFor` at `RemoteAudit.Protocol.check`,
`LineHeldAtSubprotocolBy` at `Plays`, `Verified` and `Record.summary`, and `QBytesAtSubprotocolBy` at `Publishes`,
`Verified` and `Record.summary`; `PublishedQBytes` is stated at the deployed check alone.

* **AcceptIsVerified**: what the component's check accepts is a session the parameters and the log decode to, which
  `Shared.run g` runs to an accepted state with a record `sm`, and the public outputs are `certify params g sm`'s.
* **PublishedEps**: what the component publishes as `eps` on acceptance is the sample's term of
  `LineHeldAtSubprotocol`'s bound, `δ + Σ_c s/2^64`, or nothing certified at all.
* **LineHeldAtSubprotocol**: `Properties.RemoteAudit.Line.LineHeld` with the component's acceptance in the event in
  place of `Verified`, at `LineHeld`'s bound.
* **PublishedQBytes**: what the check itself gives of a record it accepts, with no premise: `q_bytes` and `BLOCK_BYTES`
  times the pooled bound of the record's samples at `DELTA` are together at most `BLOCK_BYTES · U`.
* **QBytesAtSubprotocol**: under the samples' three premises of `LineHeldAtSubprotocol`, the subprotocol accepts a
  `q_bytes` that the trace's wrong receipts, a block each, would take past `BLOCK_BYTES · U` with probability at most the
  samples' part of the line's bound at `DELTA`: `PublishedQBytes` read through the samples' bound
  (`Accounting.Guarantees.Remote.PooledSample`).

Each reads the deployed verifier, so it is no claim of `Subprotocol`'s, whose forms over the verifier carry the
witnesses. `LineHeld` stands the same way: closed in `Properties.Accounting.Remote.Line`, its form `LineHeldBy` in
`Subprotocol`.
-/

@[expose] public section

namespace Properties.RemoteAudit.Subprotocol

open Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Draw
  Accounting.Remote.Draw.Assumptions Properties.RemoteAudit.Line

/-- **The component's check accepts only what `Shared.run g` accepts.** Whenever `RemoteAudit.Protocol.check` accepts
parameters `p`, a registration `reg` and a log `e` with public outputs `pub`: `p` and `e` decode to a session
(`sessionOf`: its parameters, its windows a draw g, coins, append coins, clock floor and events), `Shared.run g` from
`State.init` at those and `reg` runs every event to a state that ended accepted with a record `sm`, and `pub` is what
`certify` publishes of `sm` at the session's parameters and g. -/
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
      s.outcome = some "accepted" ∧ s.summary = some sm ∧ pub = (RemoteAudit.Protocol.certify params g sm).1

/-- **The component publishes the draw term of `LineHeldAtSubprotocol`'s bound.** Whenever `RemoteAudit.Protocol.check`
accepts parameters `p`, a registration `reg` and a log `e` with public outputs `pub`: `p` and `e` decode to a session
(`sessionOf`: its parameters, its windows a draw g, coins, append coins, clock floor and events), and either the public
outputs certify nothing (`eps` and `q_bytes` 0, the pooled bound refusing the windows), or at every node of those
parameters and g whose δ is the component's `RemoteAudit.Protocol.DELTA` the published `eps` is `δ + Σ_c s/2^64` over
the node's `⌈W/g⌉` draws (`Node.samples`) at its `samples` s, the draw term of `LineHeldAtSubprotocol`'s bound. It is a
constant of the session's parameters, not of the draws the session made. -/
def PublishedEps : Prop :=
  ∀ (p : RemoteAudit.Protocol.Params) (reg : RemoteAudit.Registration) (e : RemoteAudit.Protocol.Evidence)
    (pub : RemoteAudit.Protocol.Public) (prof : Lean.Json),
    RemoteAudit.Protocol.check p reg e = .accept pub prof →
    ∃ (params : RemoteAudit.Session.Params) (g : ℕ) (coins appendCoins : Array ByteArray) (notBeforeNs : ℕ)
      (events : Array RemoteAudit.Session.Event),
      RemoteAudit.Protocol.sessionOf p e = .ok (params, g, coins, appendCoins, notBeforeNs, events) ∧
      ((pub.eps = ⟨0, 1⟩ ∧ pub.qBytes = 0) ∨
        ∀ node : Node, node.params = params → node.groupsPerCoin = g →
          (node.delta : ℚ) = RemoteAudit.Protocol.DELTA →
          (pub.eps.num : ℚ) / pub.eps.den =
            (node.delta : ℚ) + ((∑ _c : Fin node.samples, node.params.samples : ℕ) : ℚ) / 2 ^ 64)

/-- **What the subprotocol's check gives of the `q_bytes` it publishes.** Whenever bytes that play record `rec` on the
node are accepted publishing `pub` (`Publishes`), `q_bytes` and `BLOCK_BYTES` times the pooled bound at
`RemoteAudit.Protocol.DELTA` of the record's samples (`Accounting.Remote.Draw.pooled`) are together at most
`BLOCK_BYTES · U`, over the reals. The samples' sizes and counts are `recordSample`'s: sample `c` holds the receipts its
windows have in the state the record's run reaches (`Record.sampleReceiptsAt`), capped at `receiptMax`, and opens
`min(s, U_c)` of them at the run's `samples` s; `U` is the receipts of the Lean verifier's record of the run
(`Record.receipts`), the count `certify` subtracts from. It takes no premise: it is a property of the compiled check,
and it holds what the code gives and the bound reads, that the samples the code pooled are the record's, each opening
`min(s, U_c)`, and that the cap at `receiptMax` raises no bound. -/
def PublishedQBytes : Prop :=
  ∀ (node : Node) (rec : Record) (a b e : ByteArray) (pub : RemoteAudit.Protocol.Public),
    Publishes node rec a b e pub →
    (pub.qBytes : ℝ) + (RemoteAudit.Codec.BLOCK_BYTES : ℝ) *
        Accounting.Remote.Draw.pooled node.samples (fun c => min (rec.sampleReceiptsAt node c) receiptMax)
          (fun c => min node.params.samples (min (rec.sampleReceiptsAt node c) receiptMax))
          (RemoteAudit.Protocol.DELTA : ℝ) ≤
      (RemoteAudit.Codec.BLOCK_BYTES : ℝ) * ((rec.receipts node).toList.sum : ℕ)

/-- What a published `q_bytes` says of the trace at `ω`: the figure and the credited passes' wrong receipts of
registered blocks (`Trace.incorrect`), a `BLOCK_BYTES` block each, are together at most the receipts the Lean
verifier's record of the run holds (`Record.receipts`, its samples' `U_c` summed), a block each. -/
def QBytesAt {Ω : Type} (T : Trace Ω) (ω : Ω) (node : Node) (r : Record) (pub : RemoteAudit.Protocol.Public) : Prop :=
  pub.qBytes + RemoteAudit.Codec.BLOCK_BYTES * T.incorrect ω ≤
    RemoteAudit.Codec.BLOCK_BYTES * (r.receipts node).toList.sum

/-- **The `q_bytes` the subprotocol publishes, at its acceptance.** `LineHeldAtSubprotocol`'s node, coins, records, trace
of record and bytes verbatim, and of its premises the samples' three alone (`Live` at every fixed coin commitment,
`Refused` at the receipts `R` and `Counted`): the probability that the subprotocol accepts bytes that play the record of
`ω` on the node with public outputs `pub` while `pub.qBytes` and the trace's wrong receipts, a block each, together
exceed the record's receipts (the event `∃ pub, Publishes … pub ∧ ¬ QBytesAt … pub`, written as the acceptance and then
what fails) is at most `δ + Σ_c s / 2^64 + Σ_c adv_c + Σ_c hide_c + bind` over the run's samples at
`δ = RemoteAudit.Protocol.DELTA`, the δ `certify` reads its pooled bound at: the samples' part of
`LineHeldAtSubprotocol`'s bound. Its first two terms are the `eps` the subprotocol publishes (`PublishedEps`); the PRF
advantages, the hiding terms and the receipts' binding term are in no published figure. A hiding term is 0 where the
records and their correct receipts read no commitment.

**What it says.** On acceptance, except with the bound, `q_bytes ≤ BLOCK_BYTES × (U − the trace's wrong receipts)`
(`BLOCK_BYTES` is 4096): `U` is the verifier's own count of the receipts in its record, and the wrong receipts are those
of registered blocks in the trace's credited passes. It bounds wrong receipts among the verifier's count. It does not
say that `q_bytes` bytes were read on the node, nor that the `U` receipts are of blocks the trace planned: bits read on
the node are `LineHeldAtSubprotocol`'s, under `Paced`, `Covers`, `SketchBoundAt`, `Covered` and `IngressCounted`. -/
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
    Pouw.pr (fun ω => (∃ pub, Publishes node (r ω) (a ω) (b ω) (e ω) pub) ∧
        ¬ ∀ pub, Publishes node (r ω) (a ω) (b ω) (e ω) pub → QBytesAt (ofRecord T node r) ω node (r ω) pub) ≤
      (RemoteAudit.Protocol.DELTA : ℝ) + ((∑ _t : Fin node.samples, node.params.samples : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t)
          ((F.play (Committed.junk com t)).distinguisher prf (RemoteAudit.Protocol.DELTA : ℝ) t)).toReal +
        (∑ t, F.hide com prf (RemoteAudit.Protocol.DELTA : ℝ) t).toReal + (R.binding (F.play com) prf rest).toReal

/-- **The memory line of record, in bits read, at the bandwidth component.** `LineHeld`'s node, trace, records and
premises verbatim, one coin for each of the session's draws (`Node.samples`), and the bytes the component is given at
each outcome (`a`, `b`, `e`: the parameters, the commitment and the evidence): the probability that the component
accepts bytes that play the record of `ω` on the node (`Plays`) while the network certifiers sign nothing but the
verifier's messages entering the node over the credited passes and the record's line credits more bits read than the
passes read is at most `δs + δnet + (δ + Σ_c s / 2^64 + Σ_c adv_c + Σ_c hide_c + bind)`, `LineHeld`'s bound. -/
def LineHeldAtSubprotocol : Prop :=
  ∀ (node : Node) (κ : ℕ) (X Cm : Type) [Fintype X] (prf : Fin node.samples → Pous.Bits κ → Blocks receiptMax)
    (rest : DrawCoins node.samples κ X → Fin node.samples → ℕ → ℕ) (recs : Cm → DrawCoins node.samples κ X → Record)
    (correct : Cm → Fin node.samples → DrawCoins node.samples κ X → Finset ℕ) (com : DrawCoins node.samples κ X → Cm)
    (T : Trace (DrawCoins node.samples κ X)) (R : Receipts node.samples κ X) (δs δnet : ℝ)
    (a b e : DrawCoins node.samples κ X → ByteArray),
    let r := played recs com
    let F := recordCommitted node recs correct
    ∀ (O : Outside (ofRecord T node r)) (I : Ingress (DrawCoins node.samples κ X)),
    Paced (ofRecord T node r) (recordPace node r) → Covers (ofRecord T node r) →
    SketchBoundAt (ofRecord T node r) O δs → Covered (ofRecord T node r) O I →
    IngressCounted (ofRecord T node r) I δnet → (∀ c, Live (F.view c)) →
    Refused (F.play com) prf rest (ofRecord T node r) R → Counted (F.play com) (ofRecord T node r) →
    Pouw.pr (fun ω => Plays node (r ω) (a ω) (b ω) (e ω) ∧
        ¬ (Quiet (ofRecord T node r) I.counted ω → ReadAt (ofRecord T node r) ω node (r ω))) ≤
      δs + δnet + ((node.delta : ℝ) + ((∑ _t : Fin node.samples, node.params.samples : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t)
          ((F.play (Committed.junk com t)).distinguisher prf (node.delta : ℝ) t)).toReal +
        (∑ t, F.hide com prf (node.delta : ℝ) t).toReal + (R.binding (F.play com) prf rest).toReal)

end Properties.RemoteAudit.Subprotocol
