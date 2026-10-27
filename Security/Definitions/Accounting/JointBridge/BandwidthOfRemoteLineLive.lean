import Definitions.Accounting.JointLive

/-!
# A period whose PoUB is the remote bandwidth audit's session, over the causal game (trusted)

PoUB's per-period guarantee over the causal game is `Accounting.PeriodLive.Sound` at `.poub`: at every strategy, the
probability over the coins the game draws that PoUB accepts while the bandwidth account read fewer than `Q` bits over
the period's window is at most its error at that strategy. The remote bandwidth audit's per-session statement is
`Properties.RemoteAudit.Line.LineHeld` (and `LineHeldAtComponent`, at the bandwidth component's acceptance, #4786),
over the session's openings coins, drawn live, with no random oracle. Its bits-read form (MEM-97 item 4) is to bound, at
every strategy of the game the session is played in, the probability that the Lean verifier accepts the session while
the bits its line credits as read exceed the bits the node read from its registered store.

`RealizesBandwidthOfRemoteLineLive` is a period whose PoUB is that session, stated over the source's three functions:
its acceptance on the run (`acc`), the bits its line credits as read on the run (`credit`), and the bits the node read
from its registered store over the session on the run of a strategy (`readBits`). The session's coins are among the
game's, drawn where the device's order puts them. The bridge that instantiates the three at `LineHeld`'s bits-read form
is to sit in `security_properties`, where `LineHeld` is, once that form is stated; this module can't name it.

Its last field is the one modelling link: the session's reads of the registered store are reads from device memory
that the bandwidth account is charged for over the period's window.
-/

namespace Accounting

/-- **A period whose PoUB is the remote bandwidth audit's session**, at certificates `c`, with the session's acceptance
`acc` and the bits its line credits as read `credit` on each run, and the bits `readBits` the node read from its
registered store on each run of each strategy. Each field is a property of the deployed run. -/
structure RealizesBandwidthOfRemoteLineLive {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit}
    {O : Type} {G : Verity.Game O} (r : PeriodLive d α C G) (c : Certificates) (acc : O → Prop) (credit : O → ℝ)
    (readBits : Verity.Game.Strategy G → O → ℝ) : Prop where
  /-- PoUB's verdict on the run is the session's acceptance -/
  accepts_iff : ∀ o, r.accepts .poub o ↔ acc o
  /-- the bandwidth certificate is within the line's credit on every accepted run: what the deployed audit certifies
  as `Q` is at most the bits its line credits as read (`q_bytes`, eight bits a byte) -/
  cert_le : ∀ o, acc o → c.Q ≤ credit o
  /-- **the modelling link**: the bits the node read from its registered store over the session are within the
  bandwidth account's reads from device memory over the period's window, on every run of every strategy -/
  read_le : ∀ s o, readBits s o ≤
    ((r.world s o).schedule.read (r.world s o).win ((r.world s o).acct .bandwidth) : ℝ)

end Accounting
