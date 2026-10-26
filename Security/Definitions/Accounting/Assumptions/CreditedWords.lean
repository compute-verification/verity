import Definitions.Accounting.JointBridge.CreditedWords

/-!
# A credited word's named premises: its priced cost on the device, and the help the network certifier admits (trusted)

The two premises `Properties.Accounting.JointBridge.WorkOfCreditedWords` adds to a PoUW source statement that credits
words (`Properties.Pouw.Period.H1CheckedCreditPeriod`), each a `Prop` taken as a hypothesis, never an `axiom`:

* `WordCostsMacs`: a per-word work lower bound, priced;
* `HelpWithinCertifier`: the right words the device imported are within the network certifier's bits.
-/

namespace Accounting.Assumptions

/-- **Each right word the device didn't import cost `m` multiply-adds of priced work**: on every period PoUW's audit
accepts, the words the source game counts as fixed right (`fixed`, its owned right committed words less the adaptive
ones), at `m` tensor-core multiply-adds each at `p .tc`, are within the compute account's work at price `p` over the
period's window (`Accounting.PeriodRun.work`) plus the same price of the `help` right words the device obtained from
outside, through the network.

It is priced, not counted: work on every op class counts at its price, so a word computed on the CUDA cores counts
too, and its multiply-adds cost more there (at `Accounting.h200S8Price` an INT8 tensor-core multiply-add costs `1` and
a CUDA-core op `32`, so `IDP4A`'s four multiply-adds an op cost `8` each). `m` is the fewest multiply-adds any credited
word stands for (`Accounting.Compose.minWordMacs`), `512` for an H1 + fx1 tile of `PouwH1Tile_v1`'s shape alone.

Why it is a premise: it is a work lower bound for computing a word, Theorem O (the naive-optimal priced floor, which
holds on paper only: `Definitions.Pouw.Guarantees.RelativeSecurity`'s docstring) at the H1 tile's scale, and Theorem O
would discharge it. Nothing proved excludes a shortcut that computes a tile's words in fewer priced ops than its
multiply-adds (a fast matrix multiplication at `K = 64`, `R = 4`, `C = 8`, or a word computed without its products).
The source game is about the run its play reads (`Accounting.TS.Play.run`), and nothing PoUW's check sees ties a word
of it to an op of the schedule. It also fails for a word computed under another account or outside the window. -/
def WordCostsMacs {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type}
    [DecidableEq I] {Ω : I → Type} (r : PeriodRun d α C Ω) (p : Price) (m : ℕ) {X : Type}
    (coin : Ω (r.own .pouw) ≃ X) (acc : ((i : I) → Ω i) → X → Prop) (fixed help : ((i : I) → Ω i) → X → ℝ) : Prop :=
  ∀ ξ y, acc ξ (coin y) →
    (m : ℝ) * p .tc * fixed ξ (coin y) ≤
      (r.work p (Function.update ξ (r.own .pouw) y) : ℝ) + m * p .tc * help ξ (coin y)

/-- **The right words the device imported are within the network certifier's bits**: on every period PoUW's audit
accepts, the `help` right words the device obtained from outside are at most `c.K / b`
(`Accounting.Certificates.helpWords`), `c.K` the network certifier's bits for the period (the joint period's
communication certificate, `PeriodRun.Holds .network`) and `b` the bits of a committed word.

Why it is a premise: the network certifier bounds the bits the node moved (`PeriodRun.commWithin`), not the words they
carry, and a right word may take fewer than `b` bits to send to a device that computes part of it. The joint period
could discharge it from the network leg with a statement that a right word the device didn't compute costs `b` bits of
what the network certifiers see; no such statement exists, and this one is unconditional, so it doesn't read the network
leg's verdict. -/
def HelpWithinCertifier {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type}
    [DecidableEq I] {Ω : I → Type} (r : PeriodRun d α C Ω) (c : Certificates) (b : ℕ) {X : Type}
    (coin : Ω (r.own .pouw) ≃ X) (acc : ((i : I) → Ω i) → X → Prop) (help : ((i : I) → Ω i) → X → ℝ) : Prop :=
  ∀ ξ y, acc ξ (coin y) → help ξ (coin y) ≤ c.helpWords b

end Accounting.Assumptions
