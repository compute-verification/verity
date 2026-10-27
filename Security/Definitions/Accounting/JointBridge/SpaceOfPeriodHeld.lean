import Definitions.Accounting.JointPeriod

/-!
# A period whose PoUS audit is a joint run's capacity audit (trusted)

In the coins model (`Accounting.PeriodRun`), PoUS's per-period guarantee is `PeriodRun.Sound c p .pous δ`.
PoUS proves its per-period guarantee (`Pous.Period.PeriodHeld`) over a joint run (`Accounting.Run`) whose capacity
coins are the ring game's. `RealizesSpaceOfPeriodHeld` is what ties the two: a period whose PoUS coordinate of the coins
is that run's capacity coins, whose capacity bit-ticks over its window are that run's over its capacity window, and
whose PoUS verdict is the source's verdict (`verdict`, the verifier's whole decision outside a break in the period's
own data, at `Accounting.SecurityProofs.Bridge.SpaceOfPeriodHeld`).

PoUS's audit grades the capacity account at the challenge times: `Pous.Period.RingDevice` reads the account's memory at
each tick of the capacity window, and `PeriodHeld` bounds the peak by the bit-ticks (`Pous.Period`'s header), so the
certificate it gives a window of `len` ticks is `S · len ≤ S₀ − S_chip − c` bit-ticks. A certificate graded at every
round boundary over a longer window is `Pous.Guarantees.ContinuousAudit`'s, which is not yet tied to the deployed
check; until it is, this bridge's guarantee holds at the challenge times only.
-/

namespace Accounting

/-- **A period realizes its PoUS audit as the capacity audit of the joint run `run`, under the source's verdict
`verdict`.** Each field is a property of the deployed period: PoUS draws its own coordinate of the period's coins, and
that coordinate is the capacity audit's coins; the other audits' coins, as the joint run reads them, don't depend on
it; the period's window has the capacity window's length; the capacity account's bit-ticks over the period's window,
at the challenge times PoUS grades, are the joint run's over its capacity window; and PoUS's verdict on the period is
the source's verdict on the joint run's coins. -/
structure RealizesSpaceOfPeriodHeld {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit}
    {I : Type} [DecidableEq I] {Ω : I → Type} (r : PeriodRun d α C Ω) {ΩW ΩQ ΩS : Type}
    (run : Run d α ΩW ΩQ ΩS) (verdict : ΩW × ΩQ × ΩS → Prop) where
  /-- PoUS's coordinate of the period's coins is the joint run's capacity coins -/
  coins : Ω (r.own .pous) ≃ ΩS
  /-- the compute and bandwidth audits' coins of the joint run, at the period's coins -/
  other : ((i : I) → Ω i) → ΩW × ΩQ
  /-- the joint run's other coins don't read PoUS's coordinate -/
  other_update : ∀ ξ x, other (Function.update ξ (r.own .pous) x) = other ξ
  /-- the period's window is as long as the joint run's capacity window -/
  len : r.win.len = (run.win .capacity).len
  /-- the capacity account's bit-ticks over the period are the joint run's over its capacity window -/
  held : ∀ ξ, r.held ξ = run.held ((other ξ).1, (other ξ).2, coins (ξ (r.own .pous)))
  /-- PoUS's verdict on the period is the source's verdict on the joint run's coins -/
  accepts : ∀ ξ, r.accepts .pous ξ → verdict ((other ξ).1, (other ξ).2, coins (ξ (r.own .pous)))

end Accounting
