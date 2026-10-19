import Definitions.Accounting.TS

/-!
# FILL-OPT per word filled (trusted)

`Accounting.Assumptions.TS.FillOpt` prices every run's filler at the menu's `χ*`. A run that skips the filler on some
words breaks it, as a run that skips folds breaks `FoldOpt` (RT-COMPOSE's B1). `FillOptFilled` is FILL-OPT per word
filled, the filler's analogue of `FoldOptFolded` (ASK-6 = A): the run's filler is priced at the share of words it
fills. The H200's headline statements (`Models.Accounting.Guarantees.HeadlineDc`) take it at the run's unfolded share,
the share of words whose fold or filler it skips, which the whole-unit luck (`WholeLuckAt`) charges: ffill's receipt
enters LI-CP-W's last checkpoint, so a unit whose filler is skipped passes only on a guess, as one whose fold is.
-/

namespace Accounting.Assumptions.TS

open Accounting.TS

variable {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
  [Fintype X]

/-- `accounting-ts-fill-opt` (assumed; FILL-OPT per word filled, the filler's analogue of ASK-6 = A). **FILL-OPT_μ per
word filled**: at every certificate, the run's filler is priced at least `(1 − μ_x)·filled·χ*(π)`, `filled` the share
of its words it fills. `FillOpt` is its case at `1`. -/
def FillOptFilled (Z : Model K R T L U F X) (A : Anatomy K T L U) (filled : ℝ) : Prop :=
  ∀ lam : R → ℝ, IsCert lam → (1 - Z.muX) * filled * Z.chiStar lam ≤ Z.G.price lam A.nfill

end Accounting.Assumptions.TS
