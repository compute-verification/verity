import Definitions.Accounting.Physical
import Definitions.Nci.Threads
import Definitions.Nci.Dampener

/-!
# The onsite parameters (trusted)

The end-to-end overhead proof's onsite spine (note:training-20261027T0134Z-draft-e2e-overhead-proof, steps 3 to 7).
`Onsite` is the onsite deployment's parameters: the hardware (its latencies `ℓ`, each location's compute `Φ` and the
links with their ends and capacities `β`), the `n` monitored servers' leaves `X i`, the wipe window `ω`, the small
subcircuit's gates `Γ`, the state `R` a server keeps across a wipe, the link budget `α F` bits a tick, the prices
`b_w, b_a`, and the reference circuit `c` with its threading `T` and `(d, e)`.

`Accounting.Guarantees.Actual.DampenerWindowsCap`'s hypotheses split three ways: those that read the actual circuit
are the onsite class (`Accounting.OnsiteClass`), `Nci.Assumptions.WeightsIncompressible` is a named premise, and the
rest read only the parameters (`Onsite.Valid`). On the class at valid parameters, `DampenerWindowsCap` gives
`b_a M ≤ n ⌈t/ω⌉₊ K Λ(K)` for `M ≤ |T.M|` counted multiplications (`M_r + M_d`), with `K = R + α F ω` (`Onsite.K`) and
`Λ(K) = Nci.Dampener.Lambda` at `T`'s reuse (`Onsite.Λ`), so `M ≥ N` takes at least `Onsite.tMin N` ticks.
-/

namespace Accounting

open Verity Finset

/-- **The onsite parameters**: the hardware `hw` (of which the class reads the latencies `ℓ`, each location's compute
`compute`, the links' ends and capacities `β`), the `n` monitored servers' leaves `X i`, the wipe window `ω` in ticks,
the small subcircuit's computed gates `Γ`, the bits `R` a server's state holds at each window's start, the link budget
`α F` bits a tick, the prices `b_w` (bits per weight used) and `b_a` (bits per activation read), and the reference
circuit `c` with its threading `T`, `(d, e)`-threaded. -/
structure Onsite (L E : Type*) where
  hw : Hardware L E
  n : ℕ
  X : Fin n → Finset L
  ω : ℕ
  Γ : ℕ
  R : ℝ
  F : ℝ
  α : ℝ
  b_w : ℝ
  b_a : ℝ
  c : Declaration
  T : Nci.Threading c
  d : ℕ
  e : ℕ

namespace Onsite

variable {L E : Type*} (P : Onsite L E)

/-- `K = R + α F ω`: the bits a server's window reads, its starting state and what its links bring in. -/
noncomputable def K : ℝ := P.R + P.α * P.F * P.ω

/-- `Λ(K)`: the most counted multiplications of `T` per `b_a` bits a window reads (`Nci.Dampener.Lambda`). -/
noncomputable def Λ : ℝ := Nci.Dampener.Lambda P.T.reuse P.e P.d P.b_w P.K

/-- **The onsite time floor** for `N` counted multiplications: `t_min = ω (b_a N / (n K Λ(K)) − 1) + 1`. The windows
number at least `b_a N / (n K Λ(K))`, the last of them partial, of at least one tick. -/
noncomputable def tMin (N : ℕ) : ℝ := P.ω * (P.b_a * N / (P.n * P.K * P.Λ) - 1) + 1

/-- `Φ_total`: the monitored servers' compute together, gates a tick. -/
def computeTotal : ℕ := ∑ i, P.hw.compute (P.X i)

/-- **Valid onsite parameters**: `DampenerWindowsCap`'s hypotheses that read only the parameters, in its order. The
window is a whole number of ticks and the servers' leaves are disjoint; the prices and `d` are positive; a server's
window does at most `Γ` gates; and `T` is `(d, e)`-threaded at `(Γ, K)`. -/
def Valid : Prop :=
  0 < P.ω ∧ (∀ i j, i ≠ j → Disjoint (P.X i) (P.X j)) ∧ 0 < P.b_w ∧ 0 < P.b_a ∧ 0 < P.d ∧
    (∀ i, P.hw.compute (P.X i) * P.ω ≤ P.Γ) ∧ P.T.Threaded P.Γ P.K P.d P.e

end Onsite

end Accounting
