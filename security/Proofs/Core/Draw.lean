module

public import Definitions.Core.Draw
public import Proofs.Core.Game.Prob

/-!
# The product of designs by kind: proof

A set of units of the product (`SamplingDesign.prod`) is its share of each kind, and the product misses it exactly when
each kind's design misses that kind's share, on coins of its own. So its miss probability is the product over kinds
(`SamplingDesign.missProb_prod`), and with the kinds the calls of a PoUW window it is the calls' product distribution.
-/

@[expose] public section

namespace Verity

namespace SamplingDesign

open Game

/-- **The product misses a set exactly as often as every kind's design misses that kind's share, independently**:
the miss probability of `⋃ₖ {k} × T k` under the product of designs by kind is `∏ₖ missProb (L k) (T k)`. -/
theorem missProb_prod {K : Type} [Fintype K] [DecidableEq K] {V : K → Type} (L : (k : K) → SamplingDesign (V k))
    (T : (k : K) → Finset (V k)) :
    (prod L).missProb (Finset.univ.sigma T) = ∏ k, (L k).missProb (T k) := by
  have hd : ∀ ω : (prod L).Ω, Disjoint ((prod L).draw ω) (Finset.univ.sigma T) ↔
      ∀ k ∈ (Finset.univ : Finset K), Disjoint ((L k).draw (ω k)) (T k) := by
    intro ω
    simp only [prod, Finset.disjoint_left, Finset.mem_sigma, Finset.mem_univ, true_and, true_implies, Sigma.forall]
  simp only [missProb]
  rw [show (fun ω : (prod L).Ω => Disjoint ((prod L).draw ω) (Finset.univ.sigma T)) =
      fun ω : (k : K) → (L k).Ω => ∀ k ∈ (Finset.univ : Finset K), Disjoint ((L k).draw (ω k)) (T k) from
    funext fun ω => propext (hd ω)]
  exact prCoin_pi_forall Finset.univ fun k (c : (L k).Ω) => Disjoint ((L k).draw c) (T k)

end SamplingDesign

end Verity
