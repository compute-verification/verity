import Definitions.Accounting.JointPeriod
import Definitions.Accounting.Compose

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
  multiply-adds on the integer tensor cores (`EXACT=GemmAccumulatorS8_v2{DOT=BlackwellS8ImmaDot32_v2}`). A circuit of
  several word kinds is priced at the fewest a word of any kind stands for (`Compose.minWordMacs`).
* The words are the play's, read off the run its game is about (`Accounting.TS.Play.run`); the work is the schedule's,
  the device's ticks. Nothing PoUW's check sees ties one to the other. That each right word the device didn't import
  cost at least `m` multiply-adds of priced work on it is a premise (`Accounting.Assumptions.WordCostsMacs`), and that
  it imported at most `K / b` right words, `K` the network certifier's bits and `b` a word's, is another
  (`Accounting.Assumptions.HelpWithinCertifier`).

So the bridge prices the credit less the words the network certifier's bits could have brought in,
`W ≤ m·p(tc)·(credit − K/b)`, at every price `p`. It needs no time credit: Compose's time line (`(1 − ε)·T_cred`,
`CreditOfWork`), which RT4-LOGIC-6 says no period statement gives, would credit device time, and the joint period's PoUW
claim is priced ops.

* `Certificates.helpWords`: the words the network certifier's `K` bits hold, at `b` bits a word.
* `RealizesCreditedWords`: a period whose PoUW audit is a source game's on PoUW's coordinate of the coins, with the work
  certificate within the priced credit, less the help, on every accepted period.
* `receiptWordMacs`: the multiply-adds an H1 + fx1 tile's committed receipt word stands for.
* `Compose.minWordMacs`: the fewest multiply-adds any credited word of a declared circuit stands for.
-/

namespace Accounting

/-- **The words the network certifier's bits hold**: `K / b`, at `b` bits a committed word (`32` for an H1 + fx1 tile's
receipt word, the XOR of the lanes' 32-bit halves). At `b = 0` it is `0`. -/
noncomputable def Certificates.helpWords (c : Certificates) (b : ℕ) : ℝ := c.K / b

/-- **A period whose PoUW audit credits words**, at certificates `c`, price `p`, `m` multiply-adds a word and `b` bits
a word, over a source game on the coins `X` with acceptance `acc` and credit `credit` (in words) at each value `ξ` of
the period's coins (read only off PoUW's coordinate: the other audits' coins fix the adversary the device is in the
source game). Each field is a property of the deployed run. -/
structure RealizesCreditedWords {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type}
    [DecidableEq I] {Ω : I → Type} (r : PeriodRun d α C Ω) (c : Certificates) (p : Price) (m b : ℕ) (X : Type)
    (acc : ((i : I) → Ω i) → X → Prop) (credit : ((i : I) → Ω i) → X → ℝ) where
  /-- PoUW's coordinate of the period's coins is the source game's coins -/
  coin : Ω (r.own .pouw) ≃ X
  /-- PoUW's verdict on the period is the source game's acceptance at those coins -/
  accepts_iff : ∀ ξ y, r.accepts .pouw (Function.update ξ (r.own .pouw) y) ↔ acc ξ (coin y)
  /-- the work certificate is within the priced credit less the help on every accepted period: what the deployed loop
  certifies as `W` at `p` is at most the words it credits, less the `K / b` the network certifier's bits could have
  brought in, `m` multiply-adds each at `p .tc` apiece -/
  cert_le : ∀ ξ y, acc ξ (coin y) → c.W ≤ m * p .tc * (credit ξ (coin y) - c.helpWords b)

/-- **The tensor-core multiply-adds one committed receipt word stands for**, on an H1 + fx1 tile of `R` rows and `C`
columns, at any `K`: a CP-W checkpoint binds one 32-wide `K`-segment of the tile's `R·C` products and is two committed
words (`Gates.Site.parts`), so `32·R·C / 2`. `PouwH1Tile_v1`'s is `receiptWordMacs 4 8 = 512`; `PouwH1Tile_v5`'s and
`_v6`'s (v1 and v3 under J10-R's noise, with v1's receipt, so the same words credited) at a warp,
`receiptWordMacs 32 32 = 16384`. -/
def receiptWordMacs (R C : ℕ) : ℕ := 16 * R * C

/-- **The fewest multiply-adds any credited word of the declared circuit stands for**, at `macs k` for a word of kind
`k` (`receiptWordMacs R C` for an H1 + fx1 tile's): the minimum over its credited words (`Compose.creditedWords`), and
`0` for a circuit of none. It is exact for a circuit whose words are all of one kind (H1 + fx1 alone). -/
def Compose.minWordMacs {ι K : Type} (Sc : Compose.Scheme K) (Cd : Compose.Declared ι) (macs : Compose.Kind → ℕ) : ℕ :=
  ((Compose.creditedWords Sc Cd).map fun i => macs (Compose.kindAt Cd i)).min?.getD 0

end Accounting
