import Definitions.Nci.Info

/-!
# NCI over threads, each thread charged by the work done on it: the vocabulary

`WeightsSupport` charges a subcircuit `b_a` for each novel value of a thread it reads, and `(d, e)`-threading bounds the
counted multiplications of a thread it does by `e` per value read, below `d` values. Charging reads fails where reads
are redundant: gates reading the products `W[i,j] · x[t,j]` of a tile `A × I × J` read `|A||I||J|` thread values that
carry at most `|J|(|A| + |I|)` entries' worth of information. Theorem 1's proof reads a thread's count only through those
two premises, so it needs no count of reads: only, for each thread, the fewest values that threading would ask of the
work done on it.

* `Threading.work d e R t`: `R`'s counted multiplications of thread `t` at `e` per value, capped at `d` values:
  `min(d, #threadMults R t / e)`. At `e > 0` it meets the threading inequality itself, and every count `ρ` of thread
  `t`'s values that meets it (`ρ < d → #threadMults R t ≤ e ρ`) is at least `work`.
-/

namespace Nci.Threading

open Verity

variable {c : RegisteredCircuit} (T : Threading c)

/-- **The work on a thread, in thread values**: `R`'s counted multiplications of thread `t` at `e` per value, capped at
`d` values, `min(d, #threadMults R t / e)`: the fewest of thread `t`'s values that `(d, e)`-threading asks a subcircuit
doing that work to read. -/
noncomputable def work (d e : ℕ) (R : Finset (Fin c.D.N)) (t : Fin T.m) : ℝ :=
  min (d : ℝ) ((T.threadMults R t).card / e)

end Nci.Threading
