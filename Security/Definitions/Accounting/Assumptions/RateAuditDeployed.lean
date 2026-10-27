import Mathlib.LinearAlgebra.Matrix.Rank
import Definitions.Accounting.RateAuditDeployed
import Definitions.Accounting.RateAuditSketchDeployed

/-!
# The deployed rate audit's measured capacity: the squaring ceiling (trusted)

What the composed guarantee (`Definitions/Accounting/Guarantees/RateAuditDeployed.lean`) assumes beyond
`Accounting.Assumptions.WorkHardness` (`Definitions/Accounting/Assumptions/RateAudit.lean`), a named `Prop` taken as a
hypothesis, never an `axiom`:

* `SquaringCeiling cost F A`: a pass's completions cost at most `F`, the node's squaring capacity over the pass (its
  measured ceiling `s` times the pass's duration, in the unit `cost` is priced in). The time is explicit in
  `Rate.PIDevice.TimeBudget`: it splits the work across the node's concurrent GPUs and CPUs, each bounded by its
  measured rate times the pass duration; `PIDevice.TimeBudget.squaringCeiling` derives this assumption at the sum of
  their rates.
* `SketchRank ks m r ε`: the probability that some sector's signs block has rational rank below `r` is at most `ε`.
  Its numeric value is computed outside Lean.
* `SketchHashBudget A qtot`: across a pass's `G` grinding candidates, the oracle queries total at most `qtot`, the
  node's hash budget for the pass.
-/

namespace Accounting.Assumptions

open Accounting.Rate Accounting.RateSketch
open Pous (pr)
open scoped ENNReal

/-- **The measured rank tail of the tensor-sketch signs.** A uniformly drawn `ks × (32·m)` signs matrix has some
sector's `ks × 32` block below rational rank `r` with probability at most `ε`:

  `Pr_σ[∃ u, rank_ℚ(block_u σ) < r] ≤ ε`.

The finite probability is exact; only its numerical evaluation is outside Lean. At the deployed 16 rows the chosen
rank is `r = 12`, where `ε = 128 · Pr[rank_ℚ(16 × 32 block) ≤ 11] ≤ 2^(-96.25)`: evaluated, with the per-compression
reading `2^(-96) + ε = 2^(-95.12)`, by a standalone seeded script (exact rational arithmetic through
the F2 rank law, with exhaustive and seeded self-checks; the script and its output are an evidence artifact, and
`verity/core/protocols/remote_bandwidth/tests/test_sketch_reading.py` holds the in-suite falsification tests). At 64
rows it is `r = 27`. -/
def SketchRank (ks m r : ℕ) (ε : ℝ≥0∞) : Prop :=
  pr (fun σ : SketchSigns ks m =>
    ∃ u, ((Accounting.Tag.signMatrix (σ.block u)).map (Int.cast : ℤ → ℚ)).rank < r) ≤ ε

/-- **The per-pass hash budget: a pass's grinding candidates share at most `qtot` oracle queries.** Across a family
of `G` candidate devices on shared data — a pass's grinding candidates, which differ only in their oracles — the
oracle queries total at most `qtot`, at every signs, oracle family and data:

  `∑_c |queries_c| ≤ qtot`.

`qtot` is the node's hash budget for the pass: the compressions it can compute in the pass's duration, the same
measured cap the deployed parameters state for a single candidate's `q` (at most `2^55` compressions a pass). The
budget is what makes grinding pay for its queries once rather than per candidate: a candidate the prover grinds spends
real compressions, so the `G` candidates' queries add within the one budget instead of each getting it whole. The
per-candidate cap `Device.bounded` alone gives only the trivial budget `qtot = G·q`
(`Accounting.RateSketch.Device.budget_of_bounded` in `Security/Proofs/Accounting/RateAudit/DeployedSketchSum.lean`),
at which the budgeted bound recovers the multiplicative one. As for `SquaringCeiling`, the bound holds in every world:
it is the hardware's capacity, which no oracle answer raises. -/
def SketchHashBudget {N m ρ h S L q w G : ℕ} {X Sig Dg : Type}
    (A : Fin G → RateSketch.Device N m ρ h S L q w X Sig Dg) (qtot : ℕ) : Prop :=
  ∀ (σ : Sig) (Os : Fin G → RateSketch.Oracle N ρ h X Dg) (D : RateSketch.Data N m ρ X),
    ∑ c, (((A c).prog σ D).queries (Os c) D).length ≤ qtot

/-- **The squaring ceiling: a pass's completions cost at most the node's measured squaring capacity `F`.** From every
on-chip state and in every world, the compute the chip's completions cost, at `cost x` per completion of `x` bits, is
at most `F`:

  `work ≤ F`.

`F = s·t_pass` is the measured capacity: the squarings a second the node can do at the codec's width (all its GPUs and
host CPUs, the chip being everything on the node that computes), times the pass's duration `t_pass`, in the unit
`cost` and `WorkHardness`'s `c_root` are priced in. The ceiling is the adversary's best measured rate, not the honest
one: at the stated codec's 1,024-bit words (masked `p = 2^1024 − 105`, `k = 3`) the stated ceiling is about 1.485×
the reference kernel's rate on the GPU, 1.67× with the host's share, against 1.68× on the GPU on the conservative
reading. With
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
