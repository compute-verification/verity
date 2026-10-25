import Definitions.Accounting.JointPeriod

/-!
# A period whose PoUW audit is H1's checked credit over the torch loop (trusted)

`Accounting.PeriodRun.Sound` at `.pouw` is the joint period's PoUW guarantee: over PoUW's own coordinate of the coins,
the audit accepts while the compute account did less than `W` of work at price `p` with probability at most its error.
`Properties.Pouw.Period.H1CheckedCreditPeriod` is PoUW's per-period statement at the deployed check for H1's tile unit
on the torch loop: over its own coins (`Paths …`), its play accepts while its fixed words fall below the check's credit
(`creditPrefix` at the accepted prefix's live words) with probability at most `δ_s + n·(δ_bind + δ_ks)`.

`RealizesWorkOfH1CheckedCreditPeriod` is a period whose PoUW audit is that statement's game. It is stated over the
source game's three functions, at each value of the other audits' coins: its acceptance (`acc`), its fixed words
(`fixed`) and its credit (`credit`). `Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod` instantiates them at
the play's acceptance, `n` times its run's fixed words, and the check's credit over the accepted prefix; this module
can't name them, since the check's statement is in `security_properties`.

Its last field is the one modelling gap (RT4-LOGIC-6): no statement turns the period's words fixed right into the
compute account's device-model work at a price (`Accounting.Guarantees.ComposePeriod.CircuitWorkPeriod`'s docstring),
so the field says it of the run.
-/

namespace Accounting

/-- **A period whose PoUW audit is H1's checked credit**, at certificates `c` and price `p`, over the source game on the
coins `X` with acceptance `acc`, fixed words `fixed` and credit `credit` at each value `ξ` of the period's coins (which
are read only off PoUW's coordinate: the other audits' coins fix the adversary the device is in the source game). Each
field is a property of the deployed run. -/
structure RealizesWorkOfH1CheckedCreditPeriod {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit}
    {I : Type} [DecidableEq I] {Ω : I → Type} (r : PeriodRun d α C Ω) (c : Certificates) (p : Price) (X : Type)
    (acc : ((i : I) → Ω i) → X → Prop) (fixed credit : ((i : I) → Ω i) → X → ℝ) where
  /-- PoUW's coordinate of the period's coins is the source game's coins: the period's PoUW coins are the windows'
  round coins and draw coins the check is played on -/
  coin : Ω (r.own .pouw) ≃ X
  /-- PoUW's verdict on the period is the source game's acceptance at those coins: the period's PoUW audit accepts
  exactly when the play the check's statement is about accepts -/
  accepts_iff : ∀ ξ y, r.accepts .pouw (Function.update ξ (r.own .pouw) y) ↔ acc ξ (coin y)
  /-- the work certificate is within the check's credit on every accepted period: what the deployed loop certifies as
  `W` is at most the live words the check credits over its accepted prefix -/
  cert_le : ∀ ξ y, acc ξ (coin y) → c.W ≤ credit ξ (coin y)
  /-- **the modelling gap (RT4-LOGIC-6)**: on an accepted period, the words fixed right are within the compute account's
  device-model work at price `p` over the period's window. No statement derives it: the source credits live words of the
  declared circuit, and nothing yet prices a word fixed right as work on the device. -/
  work_of_fixed : ∀ ξ y, acc ξ (coin y) → fixed ξ (coin y) ≤ (r.work p (Function.update ξ (r.own .pouw) y) : ℝ)

end Accounting
