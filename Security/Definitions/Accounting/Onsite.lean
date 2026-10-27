import Definitions.Accounting.Physical
import Definitions.Nci.Threads
import Definitions.Nci.Dampener

/-!
# The onsite class: the actual circuits the compute dampener's windows bound (trusted)

The end-to-end overhead proof's onsite spine (note:training-20261027T0134Z-draft-e2e-overhead-proof, steps 3 to 7).
`Onsite` is the onsite deployment's parameters: the hardware (its latencies `ℓ`, each location's compute `Φ` and the
links with their ends and capacities `β`), the `n` monitored servers' leaves `X i`, the wipe window `ω`, the small
subcircuit's gates `Γ`, the state `R` a server keeps across a wipe, the link budget `α F` bits a tick, the prices
`b_w, b_a`, and the reference circuit `c` with its threading `T` and `(d, e)`.

`Onsite.Class A t M` is the onsite class `𝒜'`: exactly the hypotheses of
`Accounting.Guarantees.Actual.DampenerWindowsCap` on the actual circuit `A` over the horizon `t`, other than the named
`Nci.Assumptions.WeightsIncompressible`, with `M` the residual's and the declared computation's counted
multiplications together (`M_r + M_d`). Each conjunct comes from physics, a parameter or a subprotocol's claim (the
spine's step 3): causality and the compute rate are the device model, the links and their capacity the network
certifier's, the state at each window's start the wipe's, the containment the sampled proofs' and PoUW's, and the
threading the reference circuit's. As in `DampenerWindowsCap`, the state is counted by width (`Placed.state`); what a
wipe bounds is information, so the wipe's bridge needs the read bound in its determination form (MEM-98).

On the class, `DampenerWindowsCap` gives `b_a M ≤ n ⌈t/ω⌉₊ K Λ(K)` with `K = R + α F ω` (`Onsite.K`) and
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

/-- **The onsite class** `𝒜'` at the horizon `t`, with `M = M_r + M_d`: `DampenerWindowsCap`'s hypotheses on the actual
circuit `A` but `WeightsIncompressible`. The window is a whole number of ticks and the servers' leaves are disjoint;
`A` is causal, does at most `Φ` gates a tick per server (unit-priced) and moves bits only on declared links within their
capacities (some wire-to-link map); the prices and `d` are positive, a server's window does at most `Γ` gates, and the
links across each server's boundary carry at most `α F` bits a tick; at the start of each window before `t` each
server's state is at most `R` bits; the developer's declaration over `[0, t)` contains `c` (at some baseline: a
containment reads none); `T` is `(d, e)`-threaded at `(Γ, K)`; and `T` counts the residual and the declared computation
together, `M ≤ |T.M|`. -/
def Class [Fintype E] [DecidableEq L] (A : Placed L) (t M : ℕ) : Prop :=
  0 < P.ω ∧ (∀ i j, i ≠ j → Disjoint (P.X i) (P.X j)) ∧
    A.Causal P.hw.ℓ ∧ A.ComputeRate (Set.range P.X) (fun _ => 1) P.hw.compute ∧
    (∃ link : Fin A.N → Fin A.N → E, A.Links P.hw.ends P.hw.β link) ∧
    0 < P.b_w ∧ 0 < P.b_a ∧ 0 < P.d ∧
    (∀ i, P.hw.compute (P.X i) * P.ω ≤ P.Γ) ∧ (∀ i, (linkCapacity P.hw.ends P.hw.β (P.X i) : ℝ) ≤ P.α * P.F) ∧
    (∀ i (k : ℕ), k * P.ω < t → (A.state (P.X i) (k * P.ω) : ℝ) ≤ P.R) ∧
    (∃ (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true),
      P.c.Contains (A.declaration (univ.biUnion P.X) t θ hθ)) ∧
    P.T.Threaded P.Γ P.K P.d P.e ∧
    M ≤ P.T.M.card

end Onsite

end Accounting
