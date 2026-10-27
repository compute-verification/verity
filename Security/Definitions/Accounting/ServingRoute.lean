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
  (`Accounting.Placed`) with its registered inputs `θ` (`Play.decl`, a `Verity.Declaration`), its counted operations
  `J` (`ServingBlocks.Counted`), and a partition `P` of its operation gates (`Verity.Partition`) whose units `S` are
  the serving blocks, the others the dark remainder, as in `ServingBlocks.ServingBlockBound`. `windows` counts the
  serving blocks: server-windows, one per server and window between two of its wipes.
* `Params.Class p`: the serving class `𝒜'_serving` at the play `p`: the device model (causality, unit-priced compute
  rate at each server, every wire between leaves on a declared link within its capacity), each server's boundary links
  at most `F` bits a tick, every serving block one server's gates over `ω` ticks from a tick at which its state is at
  most `R` bits, the dark remainder at the dark leaves with at most `n_dark` operation gates there, the per-block cap
  `q` at `(c_S, k_S)`, and the counting's structure (`OperationBodies`, `Overlap`, widths at least one bit, weights at
  most one). On it `ServingBlocks.ServingBlockBound`'s sum gives `Σ_v ω_v ≤ windows · (q(c_S, k_S) + μ k_S) + n_dark`
  (`Accounting.Guarantees.ServingRoute.ServingEndToEnd`'s proof): a serving block's reads follow from the class by the
  split (`width_into_region_le`) and the links' capacity (`ingress_le`), not as a premise.

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

/-- **One play's actual circuit for the serving route**: the actual circuit `A`, its registered input gates `θ`, its
counted operations `J`, and a partition `P` of its operation gates into `B` units, of which `S` are the serving blocks
and the others the dark remainder. -/
structure Play (L : Type) where
  A : Placed L
  θ : Finset (Fin A.N)
  θ_input : ∀ g ∈ θ, A.isInput g = true
  J : Counted A.toCircuit
  B : ℕ
  P : Partition A.toCircuit B
  S : Finset (Fin B)

namespace Play

variable {L : Type} (p : Play L)

/-- The actual circuit with its registered input gates, as a declaration. -/
abbrev decl : Declaration := ⟨p.A.toCircuit, p.θ, p.θ_input⟩

/-- **Server-windows**: the play's serving blocks, each one server's gates in one window between two of its wipes. -/
def windows : ℕ := p.S.card

end Play

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
within its capacity; each server's boundary links carry at most `F` bits a tick; every serving block is the gates of one
server `X s` over the `ω` ticks from a tick `a` (`Placed.region`) at which the server's state (`Placed.state`) is at
most `R` bits, a wipe; every other unit is at the dark leaves, which hold at most `n_dark` operation gates; `q` is a
per-block cap at `(c_S, k_S)` (`PerBlockCap`); and the counted operations' gates lead to their representatives
(`OperationBodies`), each gate in at most `μ` of them (`Overlap`), every gate at least one bit wide and every weight at
most one. -/
def Class [Fintype E] (p : Play L) : Prop :=
  p.A.Causal P.hw.ℓ ∧ p.A.ComputeRate (Set.range P.X) (fun _ => 1) P.hw.compute ∧
    (∃ link : Fin p.A.N → Fin p.A.N → E, p.A.Links P.hw.ends P.hw.β link) ∧
    (∀ s, linkCapacity P.hw.ends P.hw.β (P.X s) ≤ P.F) ∧
    (∀ u ∈ p.S, ∃ s a, p.A.state (P.X s) a ≤ P.R ∧ p.P.gates u = p.A.region (P.X s) (Set.Ico a (a + P.ω))) ∧
    (∀ u ∉ p.S, p.P.gates u ⊆ p.A.region P.dark Set.univ) ∧ (p.A.region P.dark Set.univ).card ≤ P.nDark ∧
    PerBlockCap p.decl p.J P.q P.cS P.kS ∧
    OperationBodies p.decl p.J ∧ Overlap p.J P.μ ∧ (∀ g, 1 ≤ p.A.w g) ∧ (∀ v ∈ p.J.V, p.J.ω v ≤ 1)

end Params

end Accounting.ServingRoute
