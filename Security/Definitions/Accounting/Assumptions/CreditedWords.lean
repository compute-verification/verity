import Definitions.Accounting.JointBridge.CreditedWords

/-!
# A credited word's named premises: its priced cost on the device, and the units the network certifier admits (trusted)

The two premises `Properties.Accounting.JointBridge.WorkOfCreditedWords` adds to a PoUW source statement that credits
words (`Properties.Pouw.Period.H1CheckedCreditPeriod`), each a `Prop` taken as a hypothesis, never an `axiom`:

* `WordCostsMacs`: a per-word work lower bound, priced, for the words of units the device didn't outsource;
* `HelpWithinCertifier`: the units the device outsourced are within the network certifier's bits, at `b_unit` a unit.
-/

namespace Accounting.Assumptions

/-- **Each right word of a unit the device didn't outsource cost `m` multiply-adds of priced work**: on every period
PoUW's audit accepts, the words the source game counts as fixed right (`fixed`, its owned right committed words less
the adaptive ones), at `m` tensor-core multiply-adds each at `p .tc`, are within the compute account's work at price `p`
over the period's window (`Accounting.PeriodRun.work`) plus `u` multiply-adds at the same price for each of the `help`
units the device outsourced (had computed elsewhere and brought in through the network), `u` at least a unit's credited
words times `m`.

It is priced, not counted: work on every op class counts at its price, so a word computed on the CUDA cores counts
too, and its multiply-adds cost more there (at `Accounting.h200S8Price` an INT8 tensor-core multiply-add costs `1` and
a CUDA-core op `32`, so `IDP4A`'s four multiply-adds an op cost `8` each). `m` is the fewest multiply-adds any credited
word stands for (`Accounting.Compose.minWordMacs`): `16·R·C` for H1 + fx1 tiles of one shape `R × C`
(`Accounting.receiptWordMacs`), `512` at `PouwH1Tile_v1`'s `4 × 8` and `16384` at `_v5`'s and `_v6`'s warp.

What it rests on: a credited word costs at least its multiply-adds at the public best. That is SAK(matmul), the one
computational premise of the trust base (R-ACCEL-6), at the tile's scale, not a new kind of trust; the ACCEL program's
headline states it in general form, and this is a special case. It is Theorem O (the naive-optimal priced floor, which
holds on paper only: `Definitions.Pouw.Guarantees.RelativeSecurity`'s docstring) at that scale, and Theorem O would
discharge it. Nothing proved excludes a shortcut that computes a tile's words in fewer priced ops than its
multiply-adds (a fast matrix multiplication at `K = 64`, `R = 4`, `C = 8`, or a word computed without its products).
The source game is about the run its play reads (`Accounting.TS.Play.run`), and nothing PoUW's check sees ties a word
of it to an op of the schedule. It also fails for a word computed under another account or outside the window. -/
def WordCostsMacs {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type}
    [DecidableEq I] {Ω : I → Type} (r : PeriodRun d α C Ω) (p : Price) (m u : ℕ) {X : Type}
    (coin : Ω (r.own .pouw) ≃ X) (acc : ((i : I) → Ω i) → X → Prop) (fixed help : ((i : I) → Ω i) → X → ℝ) : Prop :=
  ∀ ξ y, acc ξ (coin y) →
    (m : ℝ) * p .tc * fixed ξ (coin y) ≤
      (r.work p (Function.update ξ (r.own .pouw) y) : ℝ) + u * p .tc * help ξ (coin y)

/-- **The units the device outsourced are within the network certifier's bits**: on every period PoUW's audit
accepts, the `help` units the device had computed elsewhere are at most `c.K / b_unit`
(`Accounting.Certificates.helpUnits`), `c.K` the network certifier's bits for the period (the joint period's
communication certificate, `PeriodRun.Holds .network`) and `b_unit` the bits that carry one unit out and back.

It is stated per unit, not per bit of a word: a unit whose inputs are seed-derived (the prover's own workload) goes out
whole in about `1024` bits, its SHA-512 seed out and its SHA-512 root in, however many words it has,
so a word count `K / b` at a word's bits fails.

Why it is a premise: the network certifier bounds the bits the node moved (`PeriodRun.commWithin`), not the units they
carry, and nothing proved says a unit costs `b_unit` of them; a unit may go out in fewer bits than `b_unit` assumes. The
joint period could discharge it from the network leg with a communication lower bound per unit; no such statement
exists, and this one is unconditional, so it doesn't read the network leg's verdict. -/
def HelpWithinCertifier {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type}
    [DecidableEq I] {Ω : I → Type} (r : PeriodRun d α C Ω) (c : Certificates) (bUnit : ℕ) {X : Type}
    (coin : Ω (r.own .pouw) ≃ X) (acc : ((i : I) → Ω i) → X → Prop) (help : ((i : I) → Ω i) → X → ℝ) : Prop :=
  ∀ ξ y, acc ξ (coin y) → help ξ (coin y) ≤ c.helpUnits bUnit

end Accounting.Assumptions
