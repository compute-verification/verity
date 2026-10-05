import Definitions.Pous.Game.Oracle

/-!
# The index-tweaked hash chain of B3

Moved verbatim from the targets file.
-/

namespace Pous.HashChain

/-- The index-tweaked hash chain from `x₀`: `x_{i+1} = H(i, x_i)`. -/
def chain {m D : ℕ} (H : Fin (D + 1) × Fin m → Fin m) (x₀ : Fin m) : ℕ → Fin m
  | 0 => x₀
  | i + 1 => H (⟨i % (D + 1), Nat.mod_lt _ (Nat.succ_pos D)⟩, chain H x₀ i)

end Pous.HashChain
