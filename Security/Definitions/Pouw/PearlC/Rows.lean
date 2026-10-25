import Definitions.Pouw.PearlC.Game

/-!
# A layout's global rows (trusted definitions)

A layout's units stacked row by row: unit `u`'s row `i` is the layout's global row `rowIdx L u i = Σ_{v < u} m_v + i`,
and every global row below `rowsOf L = Σ_u m_u` is a row of exactly one unit. The W1 machine keys each unit's
activations, ticket words and useful words by global row (`Definitions.Pouw.PearlC.W1`), and the per-row seeds' split
layout makes each global row a one-row unit (`Definitions.Pouw.Assumptions.PearlC.TTOutRowSeed`).
-/

namespace Pouw.PearlC

open Finset

/-- The rows before unit `u`: `Σ_{v < u} m_v`. -/
def rowOffset (L : Layout) (u : ℕ) : ℕ := ∑ v ∈ range u, (L.shape v).m

/-- The declared rows, `R = Σ_u m_u`. -/
def rowsOf (L : Layout) : ℕ := rowOffset L L.N

/-- Unit `u`'s row `i` as a global row of `L`: `offset(u) + i` (the split layout's unit of that row). -/
def rowIdx (L : Layout) (u i : ℕ) : ℕ := rowOffset L u + i

/-- The unit that global row `v` is a row of: the last `u < N` with `offset(u) ≤ v`. -/
def unitOfRow (L : Layout) (v : ℕ) : ℕ := Nat.findGreatest (fun u => rowOffset L u ≤ v) (L.N - 1)

/-- The row of its unit that global row `v` is. -/
def rowOfRow (L : Layout) (v : ℕ) : ℕ := v - rowOffset L (unitOfRow L v)

end Pouw.PearlC
