module

public import Definitions.Compose.JointPeriod

/-! A period whose PoUS audit is a joint run's capacity audit; its guarantee holds at the challenge times only, since
`Pous.Guarantees.ContinuousAudit` is not tied to the deployed check. -/

@[expose] public section

namespace ResourceAccounting

/-- **A period realizes its PoUS audit as the capacity audit of the joint run `run`, under the source's verdict
`verdict`.** Each field is a property of the deployed period: PoUS samples its own coordinate of the period's coins, and
that coordinate is the capacity audit's coins; the other audits' coins, as the joint run reads them, don't depend on
it; the period's window has the capacity window's length; the capacity account's bit-ticks over the period's window,
at the challenge times PoUS grades, are the joint run's over its capacity window; and PoUS's verdict on the period is
the source's verdict on the joint run's coins. -/
structure RealizesSpaceOfPeriodHeld {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit}
    {I : Type} [DecidableEq I] {Ω : I → Type} (r : PeriodRun d α C Ω) {ΩW ΩQ ΩS : Type}
    (run : Run d α ΩW ΩQ ΩS) (verdict : ΩW × ΩQ × ΩS → Prop) where
  coins : Ω (r.own .pous) ≃ ΩS
  other : ((i : I) → Ω i) → ΩW × ΩQ
  other_update : ∀ ξ x, other (Function.update ξ (r.own .pous) x) = other ξ
  len : r.win.len = (run.win .capacity).len
  held : ∀ ξ, r.held ξ = run.held ((other ξ).1, (other ξ).2, coins (ξ (r.own .pous)))
  accepts : ∀ ξ, r.accepts .pous ξ → verdict ((other ξ).1, (other ξ).2, coins (ξ (r.own .pous)))

end ResourceAccounting
