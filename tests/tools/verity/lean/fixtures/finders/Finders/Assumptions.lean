/-! The finder walk's controls (`finder-walk.json`): a collision premise and its finder, stand-ins of the same full names
as the table's heads and `fields` row, since the controls have no Mathlib and the walk stops by name. -/

/-- The stand-in of a collision finder: its coins, its output and its cost. -/
structure Claims.Finder where
  S : Type
  out : S → Option (Nat × Nat)
  cost : S → Nat

/-- The stand-in of a collision premise at one finder, its argument 1. -/
def Claims.ExpectedCR (H : Nat → Nat) (F : Claims.Finder) : Prop :=
  ∀ s u v, F.out s = some (u, v) → u ≠ v → H u = H v → F.cost s > 0
