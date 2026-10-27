import Definitions.Accounting.JointBridge.CreditedWords
import Definitions.Accounting.Assumptions.CreditedWords

/-!
# The joint period's PoUW guarantee from a credit in words, at any price

`Accounting.PeriodRun.Sound` at `.pouw` is the joint period's PoUW guarantee, in priced ops: the probability over PoUW's
own coins that it accepts while the compute account did less than the work certificate `W` at price `p` is at most its
error. A PoUW source statement credits words instead: the probability that its game accepts while its words fixed right
fall below its credit is at most its error (`Properties.Pouw.Period.H1CheckedCreditPeriod`, at
`δ_s + n·(δ_bind + δ_ks)`).

`WorkOfCreditedWords` turns the second into the first at the same error, for every price `p`, when the work certificate
is the credit priced less the units the network certifier admits (`Accounting.RealizesCreditedWords`' `cert_le`:
`W ≤ m·p(tc)·credit − u·p(tc)·K/b_unit`), each right word of a unit the device didn't outsource cost `m` multiply-adds
of priced work (`Accounting.Assumptions.WordCostsMacs`, `m` the fewest any credited word stands for, `u` a unit's), and
the device outsourced at most `K/b_unit` units (`Accounting.Assumptions.HelpWithinCertifier`). It takes the source's
bound as a hypothesis over the source's own event, so it restates neither bound, and needs no time credit (Compose's
time line): the joint period's claim is priced ops.

H1's instance is `Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod`, and its H200 one at INT8
`WorkOfH1CheckedCreditPeriodH200`. Any other source that credits words, a period of v5/v6 tiles among them, takes the
bridge unchanged.
-/

namespace Properties.Accounting.JointBridge

/-- **PoUW's per-period guarantee in the joint period from a credit in words, at any price.** On any device and any
period of a node's run whose PoUW audit is a source game's on PoUW's coordinate of the coins, with the work certificate
within the credit at `m` multiply-adds a word, less the `K / b_unit` units the network certifier's bits can carry at `u`
multiply-adds a unit, all at `p .tc` (`Accounting.RealizesCreditedWords`), each right word of a unit the device didn't
outsource costing `m` multiply-adds of priced work (`Accounting.Assumptions.WordCostsMacs`) and at most `K / b_unit`
units outsourced (`Accounting.Assumptions.HelpWithinCertifier`): if, at each value `ξ` of the coins, the source game
accepts while its words fixed right fall below its credit with probability at most `δ`, then PoUW's per-period
guarantee holds at certificates `c`, price `p` and error `δ`. -/
def WorkOfCreditedWords : Prop :=
  ∀ (ι α I : Type) [Fintype α] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    (d : Accounting.Device ι) (C : Verity.Circuit) (r : Accounting.PeriodRun d α C Ω) (c : Accounting.Certificates)
    (p : Accounting.Price) (m u bUnit : ℕ) (X : Type) [Fintype X] (acc : ((i : I) → Ω i) → X → Prop)
    (fixed credit help : ((i : I) → Ω i) → X → ℝ) (δ : ℝ)
    (real : Accounting.RealizesCreditedWords r c p m u bUnit X acc credit),
    Accounting.Assumptions.WordCostsMacs r p m u real.coin acc fixed help →
    Accounting.Assumptions.HelpWithinCertifier r c bUnit real.coin acc help →
    (∀ ξ, Pouw.pr (fun x : X => acc ξ x ∧ fixed ξ x < credit ξ x) ≤ δ) →
    r.Sound c p .pouw δ

end Properties.Accounting.JointBridge
