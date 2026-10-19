import Definitions.Nci.Info
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# NCI over threads, reads counted by information: the named assumption

`WeightsIncompressible` charges a subcircuit's reads by width. Its support form, `WeightsSupport`, charges them by the
number of distinct values they take, as `ActivationsSupport` does the chain's: wiring cannot make a read carry more than
its values.
-/

namespace Nci.Assumptions

open Verity

/-- **Weights are incompressible, in support form**: every subcircuit `R` of the reference circuit `c` small by
information (at most `Γ` computed gates, at most `S` bits of information read) has novel reads that take at least
`2^(b_w · #(weights R uses) + b_a · Σ_t #(novel values of thread t that R reads))` distinct values over the runs with
the baseline at `θv`. `b_w` and `b_a` are rates in bits of information per weight used and per novel thread value read.

It gives `WeightsIncompressible` at the same rates, since reads carry no more information than their width, and the two
agree on a subcircuit whose novel reads are novel input gates. It is stronger where a subcircuit's novel reads are
redundant: a thread value and a copy of it, two thread values one of which determines the other, or a computed value
that takes fewer than `2^b_a` values over the runs. Each is charged `b_a`, and their information is less. It fails where
`WeightsIncompressible` does (weights committed in `θ`, a low-rank update, updates rebuilt from fewer bits), and where
thread values are redundant at rate above `b_a`. -/
def WeightsSupport (c : Declaration) (θv : c.D.Transcript) (T : Threading c) (Γ : ℕ) (S b_w b_a : ℝ) : Prop :=
  ∀ R, SmallInfo c θv Γ S R →
    (2 : ℝ) ^ (b_w * (T.weightsUsed R).card + b_a * ∑ t, (T.threadReads R t).card) ≤
      support c θv (c.D.into R \ c.θ)

end Nci.Assumptions
