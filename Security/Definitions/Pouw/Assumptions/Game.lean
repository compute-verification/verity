import Definitions.Pouw.Game.Defs

/-!
# Named assumptions (trusted)

Each is a `Prop` taken as a hypothesis by the statements that use it, never an `axiom`.
-/

namespace Pouw.Assumptions

open Finset

variable {Q R S : Type}

/-- **Assumption TT(γ₀), tight noised transcript** (problem statement §4.4), for a cost model `CM`, a protocol `P`
(the named assumption is its instance at `CM = M_4090` and `P = D1.m1`), a domain `D`, a target `γ₀` and an error
`ε(q, N)`.

The model's programs reach the oracle only through counted queries (`CostModel.Queries`). In the game
of §3.3 (the same adversaries, budgets and oracle-call bounds as `Gγ`), let `K'` be the units whose checked values
are correct (`checkedSet`; the useful output is not required). Then
`Pr_{H,s}[ ∑_{u ∈ K'} m_u·k_u·n_u > T / (1 − γ₀) ] ≤ ε(q, N)`.

It is stated in the same game as `Gγ`, so grinding and collisions over the `q` free oracle calls are inside
`ε(q, N)`. TT is false when `γ₀ ≥ 1` or for zero noise in natural models (`Pouw.Sanity`). -/
def TT [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : UnitLayout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ) : Prop :=
  CM.Queries ∧
  ∀ U : Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T / (1 - γ₀) < ∑ u ∈ checkedSet P U x.1 x.2 (A.transcript U x.1 x.2), ((U.layout.shape u).Wmm : ℝ))
      ≤ ε q U.layout.N

end Pouw.Assumptions
