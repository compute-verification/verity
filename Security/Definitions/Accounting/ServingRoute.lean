import Definitions.Accounting.ServingBlocks
import Definitions.Accounting.Physical
import Definitions.Core.Game

/-!
# The serving route: one play's actual circuit and the serving class (trusted)

The serving-block route of the end-to-end overhead guarantee (note:training-20261027T0134Z-draft-e2e-overhead-proof
§7.4 and §9): the route whose bound bites at the onsite reference, where the per-serving-block cap is the
activation-priced fan-out bound at the serving cut.

* `ServingRoute.Params`: the onsite deployment's parameters for the route: the hardware `hw` (of which the class reads
  the latencies `ℓ`, each location's compute and the links with their ends and capacities `β`), the monitored servers'
  leaves `X s`, the dark leaves, the wipe window `ω` in ticks, the bits `R` a wipe leaves a server, the bits `F` a
  server's boundary links carry a tick, the dark inventory `n_dark` (operation gates at the dark leaves), the serving
  block's compute budget `c_S`, the per-block cap `q` and the overlap `μ`. The serving block's read budget is
  `k_S = R + F ω` (`kS`): what a wipe leaves plus what the links bring in during a window.
* `ServingRoute.Play`: what one play of the experiment is for the route (spine §8.5, F8): the actual circuit `A`
  (`Accounting.Placed`), its registered inputs `θ`, its counted operations `J` (`ServingBlocks.Counted`), the horizon
  `t` and each server's wipe ticks `κ s` (`Cuts t`). Its serving blocks are `ServingBlocks.block A X κ S`, one per
  server and window, and `windows` counts them (server-windows: at a wipe every `ω` ticks, `n ⌈t/ω⌉`).
* `Params.Class p`: the serving class `𝒜'_serving` at the play `p`: the device model (causality, unit-priced compute
  rate at each server, every wire between leaves on a declared link within its capacity), each server's boundary links
  at most `F` bits a tick, every window at most `ω` ticks with each server's state at its start at most `R` bits, the
  counted operations' representatives on a server before `t` or at a dark leaf with at most `n_dark` operation gates at
  the dark leaves, the per-block cap `q` at `(c_S, k_S)`, and the counting's structure (`OperationBodies`, `Overlap`,
  widths at least one bit, weights at most one). On it `ServingBlocks.ServingBlockBound` gives
  `Σ_v ω_v ≤ windows · (q(c_S, k_S) + μ k_S) + n_dark` (`Accounting.Guarantees.ServingRoute.ServingEndToEnd`'s
  proof): a serving block's reads follow from the class by the split (`width_into_region_le`) and the links' capacity
  (`ingress_le`), not as a premise.

`qS = q(c_S, k_S) + μ k_S` is the per-serving-block cap with its boundary term, and `windowsMin N = (N − n_dark)/qS` the
server-windows `N` counted operations need.
-/

namespace Accounting.ServingRoute

open Verity Finset Accounting.ServingBlocks

/-- **The serving route's parameters**: the hardware `hw`, the monitored servers' leaves `X s`, the dark leaves `dark`,
the wipe window `ω` in ticks, the bits `R` a wipe leaves a server's state, the bits `F` its boundary links carry a tick,
the dark inventory `nDark` (operation gates at the dark leaves), the serving block's compute budget `cS`, the per-block
cap `q` and the overlap `μ` (each gate in at most `μ` counted operations' bodies). -/
structure Params (L σ E : Type) where
  hw : Hardware L E
  X : σ → Finset L
  dark : Finset L
  ω : ℕ
  R : ℕ
  F : ℕ
  nDark : ℕ
  cS : ℕ
  q : ℕ → ℕ → ℝ
  μ : ℕ

/-- **One play's actual circuit for the serving route**: the actual circuit `A`, its registered inputs `θ`, its counted
operations `J`, the horizon `t` and each server's wipe ticks `κ s`, the cuts of `[0, t)` its serving blocks are cut at. -/
structure Play (L σ : Type) where
  A : Placed L
  θ : Finset (Fin A.N)
  J : Counted A.toCircuit
  t : ℕ
  κ : σ → Cuts t

/-- **Server-windows**: the play's serving blocks, one per server and window between two of its wipes. -/
def Play.windows {L σ : Type} [Fintype σ] (p : Play L σ) : ℕ := Fintype.card (Block p.κ)

namespace Params

variable {L σ E : Type} (P : Params L σ E)

/-- `k_S = R + F ω`: the bits a serving block reads at most, what the wipe leaves plus what the links bring in during
its window. -/
def kS : ℕ := P.R + P.F * P.ω

/-- `q_S = q(c_S, k_S) + μ k_S`: the counted weight a serving block holds at most, computed wholly inside it or crossing
its boundary. -/
noncomputable def qS : ℝ := P.q P.cS P.kS + P.μ * P.kS

/-- **The server-windows `N` counted operations need**: `(N − n_dark) / q_S`. -/
noncomputable def windowsMin (N : ℝ) : ℝ := (N - P.nDark) / P.qS

/-- **The serving class** `𝒜'_serving` at the play `p`. The actual circuit is causal at the latencies `ℓ`, does at most
`compute (X s)` gates a tick at each server (unit-priced), and runs every wire between two leaves on a declared link
within its capacity; each server's boundary links carry at most `F` bits a tick; every window between two of a server's
wipes is at most `ω` ticks, and the server's state at its start (`Placed.state`) is at most `R` bits; every counted
operation's representative is on a server before `t` or at a dark leaf (`ServingBlocksCover`), and the dark leaves hold
at most `n_dark` operation gates; `q` is a per-block cap at `(c_S, k_S)` (`PerBlockCap`); and the counted operations'
gates lead to their representatives (`OperationBodies`), each gate in at most `μ` of them (`Overlap`), every gate at
least one bit wide and every weight at most one. -/
def Class [Fintype E] (p : Play L σ) : Prop :=
  p.A.Causal P.hw.ℓ ∧ p.A.ComputeRate (Set.range P.X) (fun _ => 1) P.hw.compute ∧
    (∃ link : Fin p.A.N → Fin p.A.N → E, p.A.Links P.hw.ends P.hw.β link) ∧
    (∀ s, linkCapacity P.hw.ends P.hw.β (P.X s) ≤ P.F) ∧
    (∀ s (j : Fin (p.κ s).J), (p.κ s).τ j.succ ≤ (p.κ s).τ j.castSucc + P.ω) ∧
    (∀ s (j : Fin (p.κ s).J), p.A.state (P.X s) ((p.κ s).τ j.castSucc) ≤ P.R) ∧
    ServingBlocksCover p.A p.J P.dark P.X p.t ∧ (p.A.region P.dark Set.univ).card ≤ P.nDark ∧
    PerBlockCap p.A.toCircuit p.θ p.J P.q P.cS P.kS ∧
    OperationBodies p.A.toCircuit p.θ p.J ∧ Overlap p.J P.μ ∧ (∀ g, 1 ≤ p.A.w g) ∧ (∀ v ∈ p.J.V, p.J.ω v ≤ 1)

end Params

end Accounting.ServingRoute
