import Definitions.Pouw.Game.Defs
import Mathlib.Algebra.Order.Archimedean.Real.Basic
import Mathlib.Data.Nat.Choose.Basic

/-!
# Clopper–Pearson's one-sided upper bound (trusted)

`t` work-weighted samples land in a set holding the share `p` of the tickets `X ~ Bin(t, p)` times. Clopper–Pearson's
upper bound at `δ` on seeing `x` of them is the least `p` from which on `Pr_p[X ≤ x] ≤ δ`: a share above it is seen more
than `x` times except with probability `δ`. It is `MissBound`'s table (`Pouw.Proofs.missBound_cpUpper`): whatever the
set, the bound at the samples it drew falls below its share with probability at most `δ`. At `x = 0` it is
`1 − δ^(1/t)`, the ε of the audit that accepts only checked units (`(1 − ε)^t = δ`), and at `x = t` it is `1`.
-/

namespace Pouw

open Finset

/-- `Pr_p[X ≤ x]` for `X ~ Bin(t, p)`. -/
noncomputable def binomCdf (t : ℕ) (p : ℝ) (x : ℕ) : ℝ :=
  ∑ k ∈ range (x + 1), (t.choose k : ℝ) * p ^ k * (1 - p) ^ (t - k)

/-- **Clopper–Pearson's one-sided upper bound at `δ` on `x` of `t` samples**: the least share `p ∈ [0, 1]` from which on
every share `q ≤ 1` has `Pr_q[X ≤ x] ≤ δ`, and `1` when there is none. -/
noncomputable def cpUpper (t : ℕ) (δ : ℝ) (x : ℕ) : ℝ :=
  sInf {p : ℝ | 0 ≤ p ∧ p ≤ 1 ∧ (p = 1 ∨ ∀ q, p ≤ q → q ≤ 1 → binomCdf t q x ≤ δ)}

end Pouw
