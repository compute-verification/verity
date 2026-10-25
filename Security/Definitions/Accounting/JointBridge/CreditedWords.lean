import Definitions.Accounting.JointPeriod

/-!
# A credit in words, priced as work on the device model (trusted)

`Accounting.PeriodRun.Sound` at `.pouw` is the joint period's PoUW guarantee: over PoUW's own coordinate of the coins,
the audit accepts while the compute account did less than the work certificate `W` at price `p` (`PeriodRun.Holds`:
`c.W ≤ r.work p ξ`) with probability at most its error. `r.work p ξ` is the device model's priced work: the compute
account's ops of each class (`Accounting.Op`) over the period's window, each at its price, summed (`Schedule.work`). It
counts ops, not time and not words.

PoUW's per-period statements credit words. `Properties.Pouw.Period.H1CheckedCreditPeriod` bounds the probability that
the play accepts while its words fixed right fall below the check's credit (`creditPrefix`: `(1 − ε(j))₊` of the live
words of the windows before the check's first refusal). The two differ in their unit and in what they are about:

* A credited word is a committed word of the declared circuit (`Compose.isWord`), not an op. An H1 + fx1 tile's words
  are its CP-W receipt words before `O′` (`Gates.roleOf` at `Gates.Site.receipt`), `2·(K/32)` a tile. Each checkpoint
  binds one 32-wide `K`-segment of the tile's `R·C` products, `32·R·C` multiply-adds, and is split into two committed
  words (`Gates.Site.parts`), so one word stands for `16·R·C` of them (`receiptWordMacs`). `PouwH1Tile_v1` (`K = 64`,
  `R = 4`, `C = 8`) has 4 words of 512, its `64·32 = 2048` products (`core_check.TILES`' `tile_work`), all INT8
  multiply-adds on the integer tensor cores (`EXACT=GemmAccumulatorS8_v2{DOT=BlackwellS8ImmaDot32_v2}`).
* The words are the play's, read off the run its game is about (`Accounting.TS.Play.run`); the work is the schedule's,
  the device's ticks. Nothing PoUW's check sees ties one to the other. That a word fixed right was computed on the
  device's tensor cores in the window, under the compute account, is a premise
  (`Accounting.Assumptions.WordsOnTensorCores`), and it is price-free.

So the bridge prices the credit, `W ≤ m·p(tc)·credit` with `m` the multiply-adds a word stands for and `p(tc)` the price
of one, at every price `p`. It needs no time credit: Compose's time line (`(1 − ε)·T_cred`, `CreditOfWork`), which
RT4-LOGIC-6 says no period statement gives, would credit device time, and the joint period's PoUW claim is priced ops.

* `PeriodRun.tensorMacs`: the compute account's tensor-core multiply-adds over the period's window.
* `RealizesCreditedWords`: a period whose PoUW audit is a source game's on PoUW's coordinate of the coins, with the work
  certificate within the priced credit on every accepted period.
* `receiptWordMacs`: the multiply-adds an H1 + fx1 tile's committed receipt word stands for.
-/

namespace Accounting

/-- The compute account's tensor-core multiply-adds over the period's window, at the coins `ξ`. -/
def PeriodRun.tensorMacs {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type}
    {Ω : I → Type} (r : PeriodRun d α C Ω) (ξ : (i : I) → Ω i) : ℕ :=
  (r.schedule ξ).ops r.win (r.acct .compute) .tc

/-- **A period whose PoUW audit credits words**, at certificates `c`, price `p` and `m` tensor-core multiply-adds a
word, over a source game on the coins `X` with acceptance `acc` and credit `credit` (in words) at each value `ξ` of the
period's coins (read only off PoUW's coordinate: the other audits' coins fix the adversary the device is in the source
game). Each field is a property of the deployed run. -/
structure RealizesCreditedWords {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type}
    [DecidableEq I] {Ω : I → Type} (r : PeriodRun d α C Ω) (c : Certificates) (p : Price) (m : ℕ) (X : Type)
    (acc : ((i : I) → Ω i) → X → Prop) (credit : ((i : I) → Ω i) → X → ℝ) where
  /-- PoUW's coordinate of the period's coins is the source game's coins -/
  coin : Ω (r.own .pouw) ≃ X
  /-- PoUW's verdict on the period is the source game's acceptance at those coins -/
  accepts_iff : ∀ ξ y, r.accepts .pouw (Function.update ξ (r.own .pouw) y) ↔ acc ξ (coin y)
  /-- the work certificate is within the priced credit on every accepted period: what the deployed loop certifies as
  `W` at `p` is at most the words it credits, `m` multiply-adds each at `p .tc` apiece -/
  cert_le : ∀ ξ y, acc ξ (coin y) → c.W ≤ m * p .tc * credit ξ (coin y)

/-- **The tensor-core multiply-adds one committed receipt word stands for**, on an H1 + fx1 tile of `R` rows and `C`
columns: a CP-W checkpoint binds one 32-wide `K`-segment of the tile's `R·C` products and is two committed words
(`Gates.Site.parts`), so `32·R·C / 2`. `PouwH1Tile_v1`'s is `receiptWordMacs 4 8 = 512`. -/
def receiptWordMacs (R C : ℕ) : ℕ := 16 * R * C

end Accounting
