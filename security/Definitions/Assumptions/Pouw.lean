module

public import Definitions.Pouw.Game.Defs
public import Definitions.Pouw.PearlC.Game

/-!
# PoUW's hashless cost models, and Pearl-C's gap property with the budget per coin

A PoUW guarantee bounds its event together with no SHA-512 collision among the inputs the run hashed (`Pouw.Breaks`).
A model whose programs hash nothing never breaks, and the guarantees that drop the break from their event take it.

`Pouw.PearlC.GGapPerCoin` is the gap property with the budget per coin (claim `pouw-gamma-per-coin`, a named
conjecture): an adversary is bounded by `T` only on the coins where its cost is at most `T`, and wins there. The gap
property on a machine that stops a run at its budget (`Pouw.PearlC.CostModel.withStop`) is equivalent to it.

Both are named assumptions under `Definitions.Assumptions`, in a module of their own that reads only PoUW's and
Pearl-C's games, so the public PoUW proofs import it without the rest of the named assumptions.
-/

@[expose] public section

namespace Pouw

/-- The model's programs hash nothing, so no run breaks. -/
def CostModel.Hashless {Q R S : Type} (CM : CostModel Q R S) : Prop :=
  ∀ (prog : CM.Prog) (H : Q → R) (U : Workload) (act : ℕ → Mat) (σ : State) (s : S), CM.hashed prog H U act σ s = []

end Pouw

namespace Pouw.PearlC

variable {Q R S : Type}

/-- `pouw-gamma-per-coin` (conjectured): **the gap property with the budget per coin**. For every adversary within `q`
oracle calls, the probability that it costs at most `T` and its correct units' honest work exceeds `T / (1 − g)` is
at most `δ(q, N)`. It is a named conjecture: neither a proof nor a counterexample is known, and it implies the gap
property `GGap` (`Pouw.PearlC.W1Stop.gGap_of_withStop`). -/
def GGapPerCoin [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (g : ℝ) (δ : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM,
    (∀ (H : Q → R) (s : S), A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q) →
      pr (fun x : (Q → R) × S => CM.cost A.online x.1 U A.act (A.pre.eval x.1) x.2 ≤ T ∧
        Wins P U g T (A.transcript U x.1 x.2) x.1 x.2) ≤ δ q U.layout.N

end Pouw.PearlC
