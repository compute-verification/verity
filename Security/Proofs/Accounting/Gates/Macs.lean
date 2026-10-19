import Definitions.Accounting.Gates.Model

/-!
# The census's dot MACs: served, tail and the rest

`census`'s `dotMacs` are every s8-family step's MACs, read from the gate whatever its role; its `servedMacs` and
`tailMacs` are those of the steps whose role is served or a tail group's. Every step's MACs are exactly one of the three
(`dotMacs_eq`): the gap between `dotMacs` and `servedMacs` is the tail MACs and the MACs of steps of neither role
(`otherMacs`), an H1 + fx1 tile's segment, `O′` and `D_d` dots and its `z` among them, whose served role is on its
receipt's words.
-/

namespace Accounting.SecurityProofs.Gates

open Accounting.Gates

/-- The MACs of `C`'s steps at `s` whose role is neither served nor a tail group's. -/
def otherMacs {ι : Type} (C : Dag ι) (s : ℕ → Prop) [DecidablePred s] : ℕ :=
  countAt C s fun g => if g.role.isServed || g.role.isTail then 0 else g.macs

/-- A gate's MACs are its served, its tail and its other MACs: exactly one of the three is nonzero. -/
theorem macs_split {ι : Type} (g : Gate ι) :
    g.macs = (if g.role.isServed then g.macs else 0) + (if g.role.isTail then g.macs else 0) +
      (if g.role.isServed || g.role.isTail then 0 else g.macs) := by
  cases g.role <;> simp [Role.isServed, Role.isTail]

/-- `macs_split` summed over a list of gates. -/
theorem sum_macs_split {ι : Type} (L : List (Gate ι)) :
    (L.map Gate.macs).sum = (L.map fun g => if g.role.isServed then g.macs else 0).sum +
      (L.map fun g => if g.role.isTail then g.macs else 0).sum +
      (L.map fun g => if g.role.isServed || g.role.isTail then 0 else g.macs).sum := by
  induction L with
  | nil => rfl
  | cons g L ih =>
    simp only [List.map_cons, List.sum_cons]
    have h := macs_split g
    omega

/-- **The dot MACs by role**: `dotMacs` is the served, the tail and the other MACs. -/
theorem dotMacs_eq {ι : Type} (C : Dag ι) (s : ℕ → Prop) [DecidablePred s] :
    (census C s).dotMacs = (census C s).servedMacs + (census C s).tailMacs + otherMacs C s :=
  sum_macs_split (C.filter fun g => decide (s g.step))

/-- **The served MACs are within the dot MACs.** -/
theorem servedMacs_le_dotMacs {ι : Type} (C : Dag ι) (s : ℕ → Prop) [DecidablePred s] :
    (census C s).servedMacs ≤ (census C s).dotMacs := by
  rw [dotMacs_eq]
  omega

end Accounting.SecurityProofs.Gates
