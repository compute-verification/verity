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
is the credit priced (`Accounting.RealizesCreditedWords`' `cert_le`: `W ≤ m·p(tc)·credit`, `m` the multiply-adds a word
stands for) and the words fixed right ran on the device's tensor cores (`Accounting.Assumptions.WordsOnTensorCores`, the
one premise it adds, price-free). It takes the source's bound as a hypothesis over the source's own event, so it
restates neither bound, and needs no time credit (RT4-LOGIC-6's Compose line): the joint period's claim is priced ops.

H1's instance (`Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod`'s functions): `X` the windows' paths,
`acc ξ` the play's acceptance, `fixed ξ x = n · ((Pl ξ).run x).fixed`, `credit ξ` the check's credit over the accepted
prefix (`creditPrefix …`), `δ = δ_s + n·(δ_bind + δ_ks)`, and `m = Accounting.receiptWordMacs 4 8 = 512` for
`PouwH1Tile_v1`'s receipt words, its multiply-adds INT8 ones (W1's S8 format, priced as `p .tc` by the price the
instance picks). It replaces that bridge's unpriced `cert_le` and its `work_of_fixed`, which compares words with priced
work directly.
-/

namespace Properties.Accounting.JointBridge

/-- **PoUW's per-period guarantee in the joint period from a credit in words, at any price.** On any device and any
period of a node's run whose PoUW audit is a source game's on PoUW's coordinate of the coins, with the work certificate
within the credit priced at `m` multiply-adds a word and `p .tc` each (`Accounting.RealizesCreditedWords`) and the words
fixed right run on the tensor cores (`Accounting.Assumptions.WordsOnTensorCores`): if, at each value `ξ` of the coins,
the source game accepts while its words fixed right fall below its credit with probability at most `δ`, then PoUW's
per-period guarantee holds at certificates `c`, price `p` and error `δ`. -/
def WorkOfCreditedWords : Prop :=
  ∀ (ι α I : Type) [Fintype α] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    (d : Accounting.Device ι) (C : Verity.Circuit) (r : Accounting.PeriodRun d α C Ω) (c : Accounting.Certificates)
    (p : Accounting.Price) (m : ℕ) (X : Type) [Fintype X] (acc : ((i : I) → Ω i) → X → Prop)
    (fixed credit : ((i : I) → Ω i) → X → ℝ) (δ : ℝ) (real : Accounting.RealizesCreditedWords r c p m X acc credit),
    Accounting.Assumptions.WordsOnTensorCores r m real.coin acc fixed →
    (∀ ξ, Pouw.pr (fun x : X => acc ξ x ∧ fixed ξ x < credit ξ x) ≤ δ) →
    r.Sound c p .pouw δ

end Properties.Accounting.JointBridge
