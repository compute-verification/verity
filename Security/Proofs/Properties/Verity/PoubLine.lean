import Proofs.Properties.Verity.Subprotocols
import Proofs.Accounting.RemoteHeld.P2Subprotocol
import Properties.RemoteAudit.P2Line.Premises

/-!
# PoUB-P2 at the code, by its line of record

`Subprotocol.poub_atCode` gives PoUB-P2's per-period guarantee at the code from the bandwidth audit's game, through a
realization of the device model's oracle. This file gives it from PoUB-P2's line of record instead
(`Properties.RemoteAudit.P2Subprotocol.LineHeldAtSubprotocol`, `SecurityProofs.lineHeldAtSubprotocol`), with no random
oracle: the verdict at the code is the P2 subprotocol's acceptance, and the line's bound is the period's `δ`.

* **The line at the period's coins** (`LineAt`): at each value of the other audits' coins, a line instance on the node:
  its records, coin commitment, trace, receipts, the bytes the subprotocol is given at each outcome, and the model's
  premises at the record's trace and pace. Its event is at most its bound (`LineAt.pr_le`).
* **The refinement** (`Subprotocol.LineRefines`): where PoUB's verdict holds, PoUB's own coin, carried to the line's
  coins (`coin`), is an outcome where the subprotocol accepts bytes that play the record (`Plays`).
* **The claim** (`Subprotocol.poub_atCode_of_line`): where the verdict holds, the certifiers counted nothing entering
  the node over the credited passes (`P2Line.QuietOverPasses`), the certificate claims no more than the record's credit
  (`P2Line.ClaimWithinCredit`) and the period's schedule reads at least what the record's passes read
  (`P2Line.PassesWithinSchedule`), so a failed claim `Q ≤ read` is a credit that claims more than the passes read, the
  line's event (`plays_summary`: the record exists where the subprotocol accepts).
* **The target** (`Vocab.bandwidthAuditSound_onsite_of_line`): onsite's PoUB couplings, each read by a line of record
  under the three premises at a `δ` within PoUB's published `δ` (`Vocab.LineCoupled`), with the error beyond `δ`
  answered for where the finder meets `FinderCR` (`Vocab.Coupling.BreakBound`), give `Vocab.BandwidthAuditSound` at
  onsite (`Vocab.poubSound_of_coupling`).
-/

namespace Properties.RemoteAudit.P2Subprotocol.SecurityProofs

open Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Draw
  Accounting.Remote.Draw.Assumptions Properties.RemoteAudit.P2Line

/-- **A line of record on a node**: what `LineHeldAtSubprotocol` reads beside the node and its coins: the records read
through the coin commitment, the commitment, the trace, the receipts, the bytes the subprotocol is given at each outcome
(the parameters, the commitment and the evidence), and the model's premises at the trace of record at the verifier's
verdict (`traceBy` at `Verified`) and the record's pace (`paceBy` at `Replay.summary`). -/
structure LineAt (node : Audited) (κ : ℕ) (X : Type) [Fintype X] where
  Cm : Type
  prf : Fin node.draws → Pous.Bits κ → Blocks Line.receiptMax
  rest : Coins node.draws κ X → Fin node.draws → ℕ → ℕ
  recs : Cm → Coins node.draws κ X → Replay
  correct : Cm → Fin node.draws → Coins node.draws κ X → Finset ℕ
  com : Coins node.draws κ X → Cm
  T : Trace (Coins node.draws κ X)
  R : Receipts node.draws κ X
  δs : ℝ
  δnet : ℝ
  a : Coins node.draws κ X → ByteArray
  b : Coins node.draws κ X → ByteArray
  e : Coins node.draws κ X → ByteArray
  outside : Outside (traceBy T node Verified (played recs com))
  ingress : Ingress (Coins node.draws κ X)
  paced : Paced (traceBy T node Verified (played recs com)) (paceBy Replay.summary node (played recs com))
  covers : Covers (traceBy T node Verified (played recs com))
  sketch : SketchBoundAt (traceBy T node Verified (played recs com)) outside δs
  covered : Covered (traceBy T node Verified (played recs com)) outside ingress
  counted_ingress : IngressCounted (traceBy T node Verified (played recs com)) ingress δnet
  live : ∀ c, Live ((recordCommitted node recs correct).view c)
  refused : Refused ((recordCommitted node recs correct).play com) prf rest (traceBy T node Verified (played recs com)) R
  counted : Counted ((recordCommitted node recs correct).play com) (traceBy T node Verified (played recs com))

namespace LineAt

variable {node : Audited} {κ : ℕ} {X : Type} [Fintype X] (L : LineAt node κ X)

/-- The record of each outcome: the one played holding the deployed commitment. -/
def record : Coins node.draws κ X → Replay := played L.recs L.com

/-- The trace of record at the verifier's verdict. -/
def trace : Trace (Coins node.draws κ X) := traceBy L.T node Verified L.record

/-- The line's bound: `δs + δnet + (δ + Σ_c s / 2^64 + Σ_c adv_c + Σ_c hide_c + bind)` at the credit's δ. -/
noncomputable def bound : ℝ :=
  L.δs + L.δnet + ((RemoteAudit.P2Session.DELTA : ℝ) +
    ((∑ _t : Fin node.draws, node.params.samples : ℕ) : ℝ) / 2 ^ 64 +
    (∑ t, Pous.StreamedWipe.prfAdvantage (L.prf t)
      (((recordCommitted node L.recs L.correct).play (Committed.junk L.com t)).distinguisher L.prf
        (RemoteAudit.P2Session.DELTA : ℝ) t)).toReal +
    (∑ t, (recordCommitted node L.recs L.correct).hide L.com L.prf (RemoteAudit.P2Session.DELTA : ℝ) t).toReal +
    (L.R.binding ((recordCommitted node L.recs L.correct).play L.com) L.prf L.rest).toReal)

/-- **The line's event is at most its bound** (`lineHeldAtSubprotocol`): the subprotocol accepts bytes that play the
record while the certifiers counted nothing entering the node over the credited passes and the record's credit claims
more bits than the passes read. -/
theorem pr_le : Pouw.pr (fun ω => Plays node (L.record ω) (L.a ω) (L.b ω) (L.e ω) ∧
    ¬ (Accounting.Remote.Held.Quiet L.trace L.ingress.counted ω →
      ReadBy Replay.summary L.trace ω node (L.record ω))) ≤ L.bound :=
  lineHeldAtSubprotocol node κ X L.Cm L.prf L.rest L.recs L.correct L.com L.T L.R L.δs L.δnet L.a L.b L.e L.outside
    L.ingress L.paced L.covers L.sketch L.covered L.counted_ingress L.live L.refused L.counted

end LineAt

end Properties.RemoteAudit.P2Subprotocol.SecurityProofs

namespace Verity.Subprotocol

open Accounting Properties.RemoteAudit.P2Line Properties.RemoteAudit.P2Subprotocol
  Properties.RemoteAudit.P2Subprotocol.SecurityProofs

section

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}
  [DecidableEq I]

/-- **PoUB-P2's refinement by its line**: at every value `ξ` of the other audits' coins and every value `x` of PoUB's
own where PoUB's verdict `A .poub` holds (at the code, `Vocab.Coupling.verdict`: onsite's PoUB-P2 component,
`RemoteAudit.P2Protocol.protocol` on PoUS's store, accepts its case, its releases are logged, and the coupling's break is
absent), at the line's coins `coin x` the subprotocol accepts bytes that play the record of the line at `ξ`
(`Plays`). -/
def LineRefines {r : PeriodRun d α C Ω} {node : Audited} {κ : ℕ} {X : Type} [Fintype X]
    (coin : Ω (r.own .poub) ≃ Accounting.Remote.Draw.Coins node.draws κ X) (L : ((i : I) → Ω i) → LineAt node κ X)
    (A : Audit → ((i : I) → Ω i) → Prop) : Prop :=
  ∀ ξ x, A .poub (Function.update ξ (r.own .poub) x) →
    Plays node ((L ξ).record (coin x)) ((L ξ).a (coin x)) ((L ξ).b (coin x)) ((L ξ).e (coin x))

variable [∀ i, Fintype (Ω i)]

/-- **PoUB-P2 at the code, by its line of record**: at the verdicts `A` that `LineRefines` reads as the P2
subprotocol's acceptance, where at each accepted outcome the certifiers counted nothing entering the node over the
credited passes (`QuietOverPasses`), the certificate claims no more than the record's credit (`ClaimWithinCredit`) and
the period's schedule reads at least what the record's passes read (`PassesWithinSchedule`), PoUB's per-period
guarantee holds at the line's bound: a failed claim at an accepted outcome is the line's event. -/
theorem poub_atCode_of_line {r : PeriodRun d α C Ω} {c : Certificates} {p : Price} {node : Audited} {κ : ℕ}
    {X : Type} [Fintype X] (coin : Ω (r.own .poub) ≃ Accounting.Remote.Draw.Coins node.draws κ X)
    (L : ((i : I) → Ω i) → LineAt node κ X) {A : Audit → ((i : I) → Ω i) → Prop} (hR : LineRefines coin L A)
    (hq : ∀ ξ x, A .poub (Function.update ξ (r.own .poub) x) →
      QuietOverPasses (L ξ).trace (L ξ).ingress (coin x))
    (hQ : ∀ ξ x, A .poub (Function.update ξ (r.own .poub) x) →
      ClaimWithinCredit c.Q node ((L ξ).record (coin x)))
    (hS : ∀ ξ x, A .poub (Function.update ξ (r.own .poub) x) →
      PassesWithinSchedule (r.read (Function.update ξ (r.own .poub) x)) (L ξ).trace (coin x) node
        ((L ξ).record (coin x)))
    {δ : ℝ} (hδ : ∀ ξ, (L ξ).bound ≤ δ) : (atCode r A).Sound c p .poub δ := by
  intro ξ
  let E : Accounting.Remote.Draw.Coins node.draws κ X → Prop := fun ω =>
    Plays node ((L ξ).record ω) ((L ξ).a ω) ((L ξ).b ω) ((L ξ).e ω) ∧
      ¬ (Accounting.Remote.Held.Quiet (L ξ).trace (L ξ).ingress.counted ω →
        ReadBy Replay.summary (L ξ).trace ω node ((L ξ).record ω))
  have key : ∀ x : Ω (r.own .poub),
      (atCode r A).accepts .poub (Function.update ξ ((atCode r A).own .poub) x) ∧
        ¬ (atCode r A).Holds c p .poub (Function.update ξ ((atCode r A).own .poub) x) → E (coin x) := by
    intro x ⟨hA, hH⟩
    have hP := hR ξ x hA
    refine ⟨hP, fun h => hH ((holds_atCode r A c p .poub _).2 ?_)⟩
    obtain ⟨sm, hsm⟩ := plays_summary hP
    exact (hQ ξ x hA sm hsm).trans ((h (hq ξ x hA) sm hsm).trans (hS ξ x hA sm hsm))
  exact (Pouw.pr_mono key).trans ((Pouw.pr_equiv coin E).le.trans (((L ξ).pr_le).trans (hδ ξ)))

end

end Verity.Subprotocol

namespace Verity

open Accounting Core Properties.RemoteAudit.P2Line Properties.RemoteAudit.P2Subprotocol.SecurityProofs

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- **Onsite's PoUB coupling, read by a line of record**: a line on a node at each value `ξ` of the other audits' coins
(`L`), PoUB's own coin carried to the line's coins (`coin`), the coupling's verdict at the code refining the P2
subprotocol's acceptance (`Subprotocol.LineRefines`), the three premises at each outcome where the verdict holds
(`QuietOverPasses`, `ClaimWithinCredit` at the certificate's `Q`, `PassesWithinSchedule` at the coupled run's
`read`), and a `δ` at least every line's bound and within PoUB's published `δ` at the strategy's calls. -/
structure LineCoupled {θ c : ByteArray} {F : Array ByteArray} {σ : Game.Strategy (V.game θ F c)}
    (q : V.Coupling .onsite .poub θ F c σ) where
  node : Audited
  κ : ℕ
  X : Type
  [fintypeX : Fintype X]
  coin : q.Ω (q.run.own .poub) ≃ Accounting.Remote.Draw.Coins node.draws κ X
  L : ((i : q.I) → q.Ω i) → LineAt node κ X
  refines : Subprotocol.LineRefines (r := q.run) coin L q.verdict
  quiet : ∀ ξ x, q.verdict .poub (Function.update ξ (q.run.own .poub) x) →
    QuietOverPasses (L ξ).trace (L ξ).ingress (coin x)
  claim : ∀ ξ x, q.verdict .poub (Function.update ξ (q.run.own .poub) x) →
    ClaimWithinCredit (V.certOf θ).Q node ((L ξ).record (coin x))
  schedule : ∀ ξ x, q.verdict .poub (Function.update ξ (q.run.own .poub) x) →
    PassesWithinSchedule (q.run.read (Function.update ξ (q.run.own .poub) x)) (L ξ).trace (coin x) node
      ((L ξ).record (coin x))
  δ : ℝ
  bound_le : ∀ ξ, (L ξ).bound ≤ δ
  within : ENNReal.ofReal δ ≤ V.delta θ .poub (V.cost θ F c σ)

attribute [instance] LineCoupled.fintypeX

/-- **Onsite's bandwidth audit by PoUB-P2's line of record** (`poubSound_of_coupling`, `Subprotocol.poub_atCode_of_line`):
where every onsite PoUB coupling is read by a line of record (`LineCoupled`), and its error beyond `δ` answers for the
rest of the event where the finder meets `FinderCR` (`Coupling.BreakBound`), `BandwidthAuditSound` holds at onsite. No
random oracle: the line bounds the P2 subprotocol's acceptance directly. -/
theorem bandwidthAuditSound_onsite_of_line (q : ∀ θ F c σ, V.Coupling .onsite .poub θ F c σ)
    (h : ∀ θ F c σ, V.LineCoupled (q θ F c σ))
    (hb : ∀ θ F c σ, V.FinderCR .poub θ F c σ → (q θ F c σ).BreakBound) : V.BandwidthAuditSound .onsite :=
  V.poubSound_of_coupling q (fun θ F c σ =>
    ⟨(h θ F c σ).δ, Subprotocol.poub_atCode_of_line (h θ F c σ).coin (h θ F c σ).L (h θ F c σ).refines
      (h θ F c σ).quiet (h θ F c σ).claim (h θ F c σ).schedule (h θ F c σ).bound_le, (h θ F c σ).within⟩) hb

end Vocab

end Verity
