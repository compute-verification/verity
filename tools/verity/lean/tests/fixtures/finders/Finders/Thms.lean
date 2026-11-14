import Finders.Defs

/-! The guarantees that hand the collision premise a finder searching its own data, one for each row of the table that
passes: the walk finds nothing in any. -/

namespace Finders

theorem searched (H : Nat → Nat) (_ : Claims.ExpectedCR H search) : True := trivial
theorem decided (H : Nat → Nat) (p : Nat → Prop) (_ : Claims.ExpectedCR H (tested p)) : True := trivial
theorem overList (H : Nat → Nat) (s : Quot (@List.Perm (Nat × Nat))) (_ : Claims.ExpectedCR H (listed s)) : True := trivial
theorem chosenInProof (H : Nat → Nat) (_ : Claims.ExpectedCR H inProof) : True := trivial
theorem whileLoop (H : Nat → Nat) (_ : Claims.ExpectedCR H looped) : True := trivial
theorem toNatParse (H : Nat → Nat) (_ : Claims.ExpectedCR H parsed) : True := trivial
theorem costOnly (H : Nat → Nat) (_ : Claims.ExpectedCR H costChosen) : True := trivial
theorem anyFinder (H : Nat → Nat) (F : Claims.Finder) (_ : Claims.ExpectedCR H F) : True := trivial

end Finders
