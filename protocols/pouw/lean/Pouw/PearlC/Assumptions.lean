import Pouw.PearlC.Model

/-!
# Named assumptions for Pearl-C (trusted)

Each is a `Prop` taken as a hypothesis by the statements that use it, never an `axiom`.
-/

namespace Pouw.PearlC.Assumptions

open Finset

variable {Q R S : Type}

/-- **Assumption TT_OUT(γ₀), output tickets force the noisy chain** (`docs/pouw/cheap-binding.md` §1), for a cost
model `CM`, a protocol `P`, a domain `D`, a target `γ₀` and an error `ε(q, N)`.

In the game of `Pouw.PearlC.Gγ` (the same adversaries, budgets and oracle-call bounds), let `K` be the units whose
ticket words and useful output words are both correct. Then
`Pr_{H,s}[ ∑_{u ∈ K} credit(u) > T / (1 − γ₀) ] ≤ ε(q, N)`.

The named instance is `γ₀ = 1/400` at `CM` = the H100 accounting and `P` = Pearl-C's protocol with its reference
functions taken from the pinned tensor-core models (`HOPPER_E4M3_K32` with `C ≠ 0` for the ticket chain, the BF16
wgmma for the combine), `PearlCAccounting P Params.headline` (or `headlineG4`) and `D = pearlCDomain`. It credits
forming only where it depends on the salt. It is stated in the same game as `Gγ`, so grinding over the `q` free
oracle calls is inside `ε(q, N)`. What falsifies it: any program of the model that outputs correct ticket and useful
words for less than `(1 − γ₀)` of their credited work on some in-domain family, such as identity tiles of a frozen
chain, exact runs long enough for Strassen, or forming that depends on the activations alone. -/
def TTOut [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ) : Prop :=
  CM.Queries ∧
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T / (1 - γ₀) < ∑ u ∈ correctSet P U x.1 x.2 (A.transcript U x.1 x.2), P.credit U.layout u)
      ≤ ε q U.layout.N

end Pouw.PearlC.Assumptions
