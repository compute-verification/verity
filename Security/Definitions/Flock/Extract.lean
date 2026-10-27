import Definitions.Flock.Game.Expect
import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# Extraction by filling: the extractor's own table

The extractor reruns the prover `K` times and takes, at each position, the first opening one of its
runs made there, and a fixed row `z₀` where none did (`firstTable`). A fresh run opens a position
off that table only if it conflicts with the run that supplied the row, or opens a position no
extractor run opened (`off_firstTable_le`). Over the `K` independent runs, the first costs
`K·Pr₂[conflict]` (a union over the `K` fresh–extractor pairs, each distributed as two independent
runs) and the second costs `Σ_p q_p (1 − q_p)^K ≤ N/(eK)` (`expect_off_firstTable_le`,
`sum_missing_le`), where `q_p` is the probability that a run opens `p`.

If every table `T` with a failing property costs at least `ε − ε_c` of such off-table mass, one
Markov step bounds the probability that the extractor's table fails (`tail_of_fixed`), and the
product form `ε · Pr[fails] ≤ 2ε_c + 2·E[mass]` needs no condition on `ε` (`joint_of_fixed`).
-/

namespace FlockSoundness

namespace Extract

open Game Rewinding
variable {β P V : Type} [Fintype P] [DecidableEq V]

/-- **The extractor's table** from runs `bs`: at each position, the first opening a run made there;
`z₀` where none did. -/
def firstTable (op : β → P → Option V) (z₀ : P → V) (bs : List β) (p : P) : V :=
  (bs.findSome? fun b => op b p).getD (z₀ p)

end Extract

end FlockSoundness
