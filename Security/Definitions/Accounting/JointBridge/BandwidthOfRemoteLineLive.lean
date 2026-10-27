import Definitions.Accounting.JointPeriod

/-!
# A period whose PoUB audit is the remote bandwidth audit's session (trusted)

`Accounting.PeriodRun.Sound` at `.poub` is the joint period's bandwidth guarantee: over PoUB's own coordinate of the
coins, the audit accepts while the bandwidth account read fewer than `Q` bits over the period's window with
probability at most its error. The remote bandwidth audit's per-session statement is
`Properties.RemoteAudit.Line.LineHeld` (and `LineHeldAtComponent`, at the bandwidth component's acceptance, #4786),
over the session's openings coins, drawn live, with no random oracle. Its bits-read form (MEM-97 item 4) is to bound
the probability that the Lean verifier accepts the session while the bits its line credits as read exceed the bits the
node read from its registered store.

`RealizesBandwidthOfRemoteLinePeriod` is a period whose PoUB audit is that session. It is stated over the source's three
functions, at each value of the other audits' coins: its acceptance (`acc`), the bits its line credits as read
(`credit`) and the bits the node read from its registered store over the session (`readBits`). The bridge that
instantiates them at `LineHeld`'s bits-read form is to sit in `security_properties`, where `LineHeld` is, once that
form is stated; this module can't name it.

Its last field is the one modelling link: the session's reads of the registered store are reads from device memory
that the bandwidth account is charged for over the period's window.
-/

namespace Accounting

/-- **A period whose PoUB audit is the remote bandwidth audit's session**, at certificates `c`, over the session's
coins `X` with acceptance `acc`, credited bits read `credit` and bits read `readBits` at each value `ξ` of the period's
coins (which are read only off PoUB's coordinate: the other audits' coins fix the prover the node is in the session).
Each field is a property of the deployed run. -/
structure RealizesBandwidthOfRemoteLinePeriod {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit}
    {I : Type} [DecidableEq I] {Ω : I → Type} (r : PeriodRun d α C Ω) (c : Certificates) (X : Type)
    (acc : ((i : I) → Ω i) → X → Prop) (credit readBits : ((i : I) → Ω i) → X → ℝ) where
  /-- PoUB's coordinate of the period's coins is the session's coins: its openings coins, drawn live -/
  coin : Ω (r.own .poub) ≃ X
  /-- PoUB's verdict on the period is the session's acceptance at those coins -/
  accepts_iff : ∀ ξ y, r.accepts .poub (Function.update ξ (r.own .poub) y) ↔ acc ξ (coin y)
  /-- the bandwidth certificate is within the line's credit on every accepted period: what the deployed audit certifies
  as `Q` is at most the bits its line credits as read (`q_bytes`, eight bits a byte) -/
  cert_le : ∀ ξ y, acc ξ (coin y) → c.Q ≤ credit ξ (coin y)
  /-- **the modelling link**: the bits the node read from its registered store over the session are within the
  bandwidth account's reads from device memory over the period's window -/
  read_le : ∀ ξ y, readBits ξ (coin y) ≤ (r.read (Function.update ξ (r.own .poub) y) : ℝ)

end Accounting
