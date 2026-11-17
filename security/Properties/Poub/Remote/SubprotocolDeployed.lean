module

public import Properties.Poub.Remote.Subprotocol

/-! The bandwidth component's properties as closed statements at the deployed verifier (`Poub.Protocol.check`). -/

@[expose] public section

namespace Properties.Poub.Subprotocol

open ResourceAccounting.Remote.Held ResourceAccounting.Remote.Held.Assumptions ResourceAccounting.Remote.Sample
  ResourceAccounting.Remote.Sample.Assumptions Properties.Poub.Line

/-- **The component's check accepts only what `Shared.run g` accepts.** Whenever `Poub.Protocol.check` accepts
parameters `p`, a registration `reg` and a log `e` with public outputs `pub`: `p` and `e` decode to a session
(`sessionOf`: its parameters, its windows a sample g, coins, append coins, clock floor and events), `Shared.run g` from
`State.init` at those and `reg` runs every event to a state that ended accepted with a record `sm`, and `pub` is what
`certify` publishes of `sm` at the session's parameters and g. -/
def AcceptIsVerified : Prop :=
  ∀ (p : Poub.Protocol.Params) (reg : Poub.Registration) (e : Poub.Protocol.Evidence)
    (pub : Poub.Protocol.Public) (prof : Lean.Json),
    Poub.Protocol.check p reg e = .accept pub prof →
    ∃ (params : Poub.Session.Params) (g : ℕ) (coins appendCoins : Array ByteArray) (notBeforeNs : ℕ)
      (events : Array Poub.Session.Event) (rs : Array Poub.Session.Result)
      (s : Poub.Session.State) (sm : Poub.Session.Summary),
      Poub.Protocol.sessionOf p e = .ok (params, g, coins, appendCoins, notBeforeNs, events) ∧
      Poub.Shared.run g (Poub.Session.State.init params reg coins appendCoins notBeforeNs) events =
        .ok (rs, s) ∧
      s.outcome = some "accepted" ∧ s.summary = some sm ∧ pub = (Poub.Protocol.certify params g sm).1

/-- **The component publishes at least the sample term of `LineHeldAtSubprotocol`'s bound.** Whenever
`Poub.Protocol.check` accepts parameters `p`, a registration `reg` and a log `e` with public outputs `pub`: `p`
and `e` decode to a session (`sessionOf`: its parameters, its windows a sample g, coins, append coins, clock floor and
events), and either the public outputs certify nothing (`eps` and `q_bytes` 0, the pooled bound refusing the windows),
or at every node of those parameters and g whose δ is the component's `Poub.Protocol.DELTA` the published `eps`
is `δ + Σ_c s/2^64` over the node's `⌈W/g⌉` samples (`Node.samples`) at its `samples` s, at least
`LineHeldAtSubprotocol`'s sample term `δ + Σ_c s/2^256`. It is a constant of the session's parameters, not of the
samples the session made. -/
def PublishedEps : Prop :=
  ∀ (p : Poub.Protocol.Params) (reg : Poub.Registration) (e : Poub.Protocol.Evidence)
    (pub : Poub.Protocol.Public) (prof : Lean.Json),
    Poub.Protocol.check p reg e = .accept pub prof →
    ∃ (params : Poub.Session.Params) (g : ℕ) (coins appendCoins : Array ByteArray) (notBeforeNs : ℕ)
      (events : Array Poub.Session.Event),
      Poub.Protocol.sessionOf p e = .ok (params, g, coins, appendCoins, notBeforeNs, events) ∧
      ((pub.eps = ⟨0, 1⟩ ∧ pub.qBytes = 0) ∨
        ∀ node : Node, node.params = params → node.groupsPerCoin = g →
          (node.delta : ℚ) = Poub.Protocol.DELTA →
          (pub.eps.num : ℚ) / pub.eps.den =
            (node.delta : ℚ) + ((∑ _c : Fin node.samples, node.params.samples : ℕ) : ℚ) / 2 ^ 64)

/-- **What the subprotocol's check gives of the `q_bytes` it publishes**, with no premise: on bytes that play record
`rec` accepted with `pub`, `q_bytes` and `BLOCK_BYTES` times the pooled bound at `DELTA` of the record's samples
are together at most `BLOCK_BYTES · U`, `U` the receipts of the Lean verifier's record. -/
def PublishedQBytes : Prop :=
  ∀ (node : Node) (rec : Record) (a b e : ByteArray) (pub : Poub.Protocol.Public)
    (p : Poub.Protocol.Params) (ev : Poub.Protocol.Evidence) (prof : Lean.Json),
    (Core.canonicalJson a >>= Poub.Protocol.decodeParams) = .ok p →
    Poub.Protocol.decodeCommitment b = .ok node.reg → Poub.Protocol.decodeEvidence e = .ok ev →
    Poub.Protocol.sessionOf p ev =
      .ok (node.params, node.groupsPerCoin, rec.coins, rec.appendCoins, rec.notBeforeNs, rec.events) →
    Poub.Protocol.check p node.reg ev = .accept pub prof →
    (pub.qBytes : ℝ) + (Poub.Codec.BLOCK_BYTES : ℝ) *
        ResourceAccounting.Remote.Sample.pooled node.samples (fun c => min (rec.sampleReceiptsAt node c) receiptMax)
          (fun c => min node.params.samples (min (rec.sampleReceiptsAt node c) receiptMax))
          (Poub.Protocol.DELTA : ℝ) ≤
      (Poub.Codec.BLOCK_BYTES : ℝ) * ((rec.receipts node).toList.sum : ℕ)

/-- What a published `q_bytes` says of the trace at `ω`: the figure and the credited passes' wrong receipts of
registered blocks (`Trace.incorrect`), a `BLOCK_BYTES` block each, are together at most the receipts the Lean
verifier's record of the run holds (`Record.receipts`, its samples' `U_c` summed), a block each. -/
def QBytesAt {Ω : Type} (T : Trace Ω) (ω : Ω) (node : Node) (r : Record) (pub : Poub.Protocol.Public) : Prop :=
  pub.qBytes + Poub.Codec.BLOCK_BYTES * T.incorrect ω ≤
    Poub.Codec.BLOCK_BYTES * (r.receipts node).toList.sum

/-- **The `q_bytes` the subprotocol publishes**: under `LineHeldAtSubprotocol`'s sample premises, it accepts
`pub.qBytes` and the trace's wrong receipts, a block each, past the record's receipts with at most that bound's
samples' part at `DELTA`. It does not say that `q_bytes` bytes were read on the node. -/
def QBytesAtSubprotocol : Prop :=
  ∀ (node : Node) (κ : ℕ) (X Cm : Type) [Fintype X] (prf : Fin node.samples → Pous.Bits κ → Blocks receiptMax)
    (rest : SampleCoins node.samples κ X → Fin node.samples → ℕ → ℕ) (recs : Cm → SampleCoins node.samples κ X → Record)
    (correct : Cm → Fin node.samples → SampleCoins node.samples κ X → Finset ℕ)
    (com : SampleCoins node.samples κ X → Cm)
    (T : Trace (SampleCoins node.samples κ X)) (R : Receipts node.samples κ X)
    (a b e : SampleCoins node.samples κ X → ByteArray),
    let r := played recs com
    let F := recordCommitted node recs correct
    (∀ c, Live (F.view c)) →
    Refused (F.play com) prf rest (ofRecord T node r) R → Counted (F.play com) (ofRecord T node r) →
    Pouw.pr (fun ω => (∃ pub, Publishes node (r ω) (a ω) (b ω) (e ω) pub) ∧
        ¬ ∀ pub, Publishes node (r ω) (a ω) (b ω) (e ω) pub → QBytesAt (ofRecord T node r) ω node (r ω) pub) ≤
      (Poub.Protocol.DELTA : ℝ) + ((∑ _t : Fin node.samples, node.params.samples : ℕ) : ℝ) / 2 ^ 256 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t)
          ((F.play (Committed.junk com t)).distinguisher prf (Poub.Protocol.DELTA : ℝ) t)).toReal +
        (∑ t, F.hide com prf (Poub.Protocol.DELTA : ℝ) t).toReal + (R.binding (F.play com) prf rest).toReal

/-- **The memory line of record, in bits read, at the bandwidth component.** `LineHeld`'s node, trace, records and
premises verbatim, one coin for each of the session's samples (`Node.samples`), and the bytes the component is given at
each outcome (`a`, `b`, `e`: the parameters, the commitment and the evidence): the probability that the component
accepts bytes that play the record of `ω` on the node (`Plays`) while the network attesters sign nothing but the
verifier's messages entering the node over the credited passes and the record's line credits more bits read than the
passes read is at most `δs + δnet + (δ + Σ_c s / 2^256 + Σ_c adv_c + Σ_c hide_c + bind)`, `LineHeld`'s bound. -/
def LineHeldAtSubprotocol : Prop :=
  ∀ (node : Node) (κ : ℕ) (X Cm : Type) [Fintype X] (prf : Fin node.samples → Pous.Bits κ → Blocks receiptMax)
    (rest : SampleCoins node.samples κ X → Fin node.samples → ℕ → ℕ) (recs : Cm → SampleCoins node.samples κ X → Record)
    (correct : Cm → Fin node.samples → SampleCoins node.samples κ X → Finset ℕ)
    (com : SampleCoins node.samples κ X → Cm)
    (T : Trace (SampleCoins node.samples κ X)) (R : Receipts node.samples κ X) (δs δnet : ℝ)
    (a b e : SampleCoins node.samples κ X → ByteArray),
    let r := played recs com
    let F := recordCommitted node recs correct
    ∀ (O : Outside (ofRecord T node r)) (I : Ingress (SampleCoins node.samples κ X)),
    Paced (ofRecord T node r) (recordPace node r) → Covers (ofRecord T node r) →
    SketchBoundAt (ofRecord T node r) O δs → Covered (ofRecord T node r) O I →
    IngressCounted (ofRecord T node r) I δnet → (∀ c, Live (F.view c)) →
    Refused (F.play com) prf rest (ofRecord T node r) R → Counted (F.play com) (ofRecord T node r) →
    Pouw.pr (fun ω => Plays node (r ω) (a ω) (b ω) (e ω) ∧
        ¬ (Quiet (ofRecord T node r) I.counted ω → ReadAt (ofRecord T node r) ω node (r ω))) ≤
      δs + δnet + ((node.delta : ℝ) + ((∑ _t : Fin node.samples, node.params.samples : ℕ) : ℝ) / 2 ^ 256 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t)
          ((F.play (Committed.junk com t)).distinguisher prf (node.delta : ℝ) t)).toReal +
        (∑ t, F.hide com prf (node.delta : ℝ) t).toReal + (R.binding (F.play com) prf rest).toReal)

end Properties.Poub.Subprotocol
