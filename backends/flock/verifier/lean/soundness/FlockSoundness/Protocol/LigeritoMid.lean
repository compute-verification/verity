import FlockSoundness.LigeritoDecode

/-!
# From one level to the next (PROTOCOL.md §13.3 steps 5–8)

After a level's folds, the prover commits the next table, the verifier binds it by two
out-of-domain samples, and checks the fold against it at the level's queries. From `PostFold` on
the level, this leaves `PreFold` on the next level except with probability
* `pairs(L')·(μ/|F|)²` (two list elements agree at both OOD points, `value_twoOod_collide_le`),
* `2L'/|F|` (the OOD claims' batching coins),
* the level's `queryError` (the queries miss the unique OOD-consistent message's mismatch),
* `⌈log2 Q⌉/|F|` (the consistency coins cancel it) and `L'/|F|` (its batching coin).
-/

namespace FlockSoundness

open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## Batching coins over `K`-valued differences -/

/-! ## The two OOD samples -/

/-! ## The queries and the consistency claim -/

theorem positions_length (A : Arith F K) (Q d : ℕ) (w : Fin Q → F) :
    (positions A Q d w).length = Q := by
  rw [positions_eq]; simp

/-! ## One level to the next -/

end FlockSoundness
