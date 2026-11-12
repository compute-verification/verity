module

public import Properties.Accounting.Remote.Subprotocol

/-!
# The bandwidth component's three properties at the deployed verifier (`RemoteAudit.Protocol`)

The three properties `Properties.Accounting.Remote.Subprotocol` describes, each a closed statement: its form over the
verifier there at the deployed one, `AcceptIsVerifiedFor` and `PublishedEpsFor` at `RemoteAudit.Protocol.check`, and
`LineHeldAtSubprotocolBy` at `Plays`, `Verified` and `Record.summary`.

* **AcceptIsVerified**: what the component's check accepts is a session the parameters and the log decode to, which
  `Shared.run g` runs to an accepted state with a record `sm`, and the public outputs are `certify params g sm`'s.
* **PublishedEps**: what the component publishes as `eps` on acceptance is the sample's term of
  `LineHeldAtSubprotocol`'s bound, `δ + Σ_c s/2^64`, or nothing certified at all.
* **LineHeldAtSubprotocol**: `Properties.RemoteAudit.Line.LineHeld` with the component's acceptance in the event in
  place of `Verified`, at `LineHeld`'s bound.

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
