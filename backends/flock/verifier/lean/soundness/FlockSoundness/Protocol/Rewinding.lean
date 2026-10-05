import Mathlib.Algebra.Order.Chebyshev
import Mathlib.Combinatorics.Enumerative.DoubleCounting
import Mathlib.Data.Fintype.Prod
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Order.ConditionallyCompleteLattice.Finset

/-!
# The rewinding lemma: openings off the plurality table cost collisions

The compiled theorem (`README.md` §1.2) covers every prover, including one that never
"knows" a table. Its committed table is the **plurality** table: at each position, the column the
prover opens in the most runs (`IsPlurality`). A run is `bad` when it opens some position to a
column off the plurality; a pair of runs `conflict`s when both open one position, to different
columns. By `Merkle.opening_binding`, each conflicting pair (both paths valid against one cap) is
a collision of the hash, found by running the prover twice with independent coins.

`bad_sq_le`: if each run opens at most `Q` of the `N` positions, then

  `#bad ² ≤ N · Q · #conflict`   (counts over runs `Ω` and pairs `Ω × Ω`),

so `Pr[bad] ≤ √(N · Q · Pr[conflict])` (`prBad_le`). With `Pr[conflict] ≤ Adv_CR(B)` for the
two-run collision finder `B`, this is the per-level term `√(N_ℓ Q_ℓ · Adv_CR)` of `DESIGN.md` §1.2.

Proof: `bad` is covered by the per-position sets of off-plurality runs, of sizes `b_p`. A run
that opens `p` to `v` conflicts at `p` with every run opening `p` to anything but `v`, and there
are at least `b_p` of those because no value beats the plurality, so `#conflictAt p ≥ q_p b_p ≥
b_p²`. Each conflicting pair is counted at no more than `Q` positions, and Cauchy–Schwarz gives
`(Σ b_p)² ≤ N Σ b_p²`.
-/

open Finset

namespace FlockSoundness.Rewinding

variable {Ω P V : Type} [Fintype Ω] [DecidableEq Ω] [DecidableEq V]

-- `op ω p = some v`: the run with coins `ω` opens position `p` to `v` (with a path that
-- verifies); `none`: it does not open `p`.
variable (op : Ω → P → Option V)

/-- The run opens `p` to a value other than `vstar p`. -/
abbrev Off (vstar : P → V) (ω : Ω) (p : P) : Prop := (op ω p).isSome ∧ op ω p ≠ some (vstar p)

/-- The two runs open `p` to different values. -/
abbrev ConflictAt (x : Ω × Ω) (p : P) : Prop :=
  (op x.1 p).isSome ∧ (op x.2 p).isSome ∧ op x.1 p ≠ op x.2 p

section counting

variable {op}

end counting

variable [Fintype P]

section rewinding

variable {op}

end rewinding

end FlockSoundness.Rewinding
