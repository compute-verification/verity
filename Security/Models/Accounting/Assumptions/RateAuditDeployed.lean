import Definitions.Accounting.RateAuditDeployed

/-!
# The deployed rate audit's measured capacity: the squaring ceiling (trusted)

What the composed guarantee (`Models/Accounting/Guarantees/RateAuditDeployed.lean`) assumes beyond
`Accounting.Assumptions.WorkHardness` (`Models/Accounting/Assumptions/RateAudit.lean`), a named `Prop` taken as a
hypothesis, never an `axiom`:

* `SquaringCeiling cost F A`: a pass's completions cost at most `F`, the node's squaring capacity over the pass (its
  measured ceiling `s` times the pass's duration, in the unit `cost` is priced in). The time is explicit in
  `Rate.PIDevice.TimeBudget`: it splits the work across the node's concurrent GPUs and CPUs, each bounded by its
  measured rate times the pass duration; `PIDevice.TimeBudget.squaringCeiling` derives this assumption at the sum of
  their rates.
-/

namespace Accounting.Assumptions

open Accounting.Rate

/-- **The squaring ceiling: a pass's completions cost at most the node's measured squaring capacity `F`.** From every
on-chip state and in every world, the compute the chip's completions cost, at `cost x` per completion of `x` bits, is
at most `F`:

  `work ≤ F`.

`F = s·t_pass` is the measured capacity: the squarings a second the node can do at the codec's width (all its GPUs and
host CPUs, the chip being everything on the node that computes), times the pass's duration `t_pass`, in the unit
`cost` and `WorkHardness`'s `c_root` are priced in. The ceiling is the adversary's best measured rate, not the honest
one: at the stated codec's 1,024-bit words (masked `p = 2^1024 − 105`, `k = 3`) the stated ceiling is 1.42× the
reference kernel's rate, 1.68× on the conservative reading (lane ra-squaring-ceiling, verity#1887). With
`WorkHardness b rounds c_root cost` it bounds the bits a pass completes by `b·F/(rounds·c_root)`
(`Accounting.Rate.Deployed.rebuildBits`), the term `μ_rebuild`.

**Why every world.** As for the device's `bounded` field (the link's `L` and the `q` queries), the bound is on the
program, whatever its steps are answered with: it is the hardware's capacity, which no answer raises. A device whose
completions would exceed `F` in some world is outside the node.

**Where time and concurrency enter.** A `PIDevice.TimeBudget A cost k duration rate` assigns every unit of completion
work in every world to one of `k` concurrent devices and bounds device `i` by `rate i * duration`. Its proved
`squaringCeiling` lemma gives this `Prop` at `F = (∑ i, rate i) * duration`. Thus adding a GPU or CPU adds its rate;
it does not silently give every processor the node's whole ceiling. -/
def SquaringCeiling {N b h ρ S L q : ℕ} (cost : ℕ → ℕ) (F : ℕ) (A : PIDevice N b h ρ S L q) : Prop :=
  ∀ (st : Pous.Bits S) (w : IWorld N b h ρ), (A.chip st).work cost w [] ≤ F

end Accounting.Assumptions
