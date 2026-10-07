import Definitions.Pouw.TileBound.AlignedExactDefs
/-!
# The aligned-exact region count's guarantee (trusted)

Each guarantee is a `def G : Prop`, with the docstring, binders and `open`s of the statement in its source
module. Its proof is the theorem `Pouw.SecurityProofs.AlignedExact.G : Pouw.Guarantees.AlignedExact.G`.
-/

namespace Pouw.AlignedExact
open Finset Classical

/-- **The region bound for known columns.** Under `RowCompat f b θ`, for every assignment `c` of column seeds, the
row-seed assignments with an all-exact `a × b` block number at most `C(m, a)·C(n, b)·θ^a` times all row-seed
assignments. -/
def _root_.Pouw.Guarantees.AlignedExact.RegionCountGivenLe : Prop :=
  ∀ {m n : ℕ} {ρ κ : Type*} [Fintype ρ] (f : Fin m → Fin n → ρ → κ → Prop) (a b : ℕ) (θ : ℚ) (hθ : 0 ≤ θ) (hf : RowCompat f b θ) (c : Fin n → κ),
    ((univ.filter fun R : Fin m → ρ => HasRegion f a b (R, c)).card : ℚ) ≤
    (m.choose a : ℚ) * (n.choose b) * θ ^ a * Fintype.card (Fin m → ρ)

end Pouw.AlignedExact
