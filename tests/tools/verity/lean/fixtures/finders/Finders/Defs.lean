import Finders.Assumptions

/-! The finders the controls hand the collision premise, and stand-ins of the library constants the table refuses or
passes by name, under the same full names. The finders that pick by choice are handed to it only in the violations
(`finder-violations.json`); the guarantees here (`Finders.Thms`) hand it the ones that search their own data. -/

/-- The stand-in of Mathlib's `Nat.find`, which the walk refuses by name: this one chooses nothing. -/
def Nat.find {p : Nat → Prop} [DecidablePred p] (_ : ∃ n, p n) : Nat := 0

/-- The stand-in of Mathlib's `Encodable.choose`, which the walk refuses by name: this one chooses nothing. -/
def Encodable.choose {α : Type} [Inhabited α] {p : α → Prop} [DecidablePred p] (_ : ∃ x, p x) : α := default

/-- The stand-in of Mathlib's `Quot.out`, which reaches `Classical.choice` as Mathlib's does. -/
noncomputable def Quot.out {α : Type} {r : α → α → Prop} (q : Quot r) : α := Classical.choose (Quot.exists_rep q)

/-- The stand-in of Mathlib's `Multiset.toList`, which reaches `Classical.choice` as Mathlib's does and which the walk
passes by name, going on into its argument. -/
noncomputable def Multiset.toList {α : Type} (s : Quot (@List.Perm α)) : List α := Quot.out s

namespace Finders

/-- A carrier: the finder reaches the premise through a parameter. -/
def CR (H : Nat → Nat) (F : Claims.Finder) : Prop := Claims.ExpectedCR H F

/-- A structure of premises: the finder reaches the premise through a field's type. -/
structure Prem (H : Nat → Nat) (F : Claims.Finder) : Prop where
  cr : Claims.ExpectedCR H F

/-- A finder of the output `out`, at a cost of one call. -/
def ofOut (out : Nat → Option (Nat × Nat)) : Claims.Finder := ⟨Nat, out, fun _ => 1⟩

/-! Finders that pick by choice, which the walk refuses. -/

/-- A pair taken by choice. -/
noncomputable def chosen (h : ∃ p : Nat × Nat, p.1 ≠ p.2) : Claims.Finder := ofOut fun _ => some (Classical.choose h)

/-- A value proved unique, taken by choice. -/
noncomputable def uniqueChosen : Claims.Finder :=
  ofOut fun n => some (Classical.choose (⟨0, rfl⟩ : ∃ k : Nat, k = 0), n)

/-- A search with no bound. -/
def found {n : Nat} (h : ∃ k : Nat, k > n) : Claims.Finder := ofOut fun n => some (Nat.find h, n)

/-- An `Encodable.choose`. -/
def encoded {n : Nat} (h : ∃ k : Nat, k > n) : Claims.Finder := ofOut fun n => some (Encodable.choose h, n)

/-- A representative taken by `Quot.out`. -/
noncomputable def represented (q : Quot (fun a b : Nat => a % 2 = b % 2)) : Claims.Finder :=
  ofOut fun n => some (Quot.out q, n)

/-! Finders that search their own data, which the walk passes. -/

/-- A search over a list. -/
def search : Claims.Finder := ofOut fun n => [(1, 2), (3, 4)].find? fun p => p.1 + p.2 == n

open Classical in
/-- A test decided by `Classical.propDecidable`: one bit, no witness. -/
noncomputable def tested (p : Nat → Prop) : Claims.Finder := ofOut fun n => if p n then some (n, n + 1) else none

/-- A search over a `Multiset.toList` stand-in's list. -/
noncomputable def listed (s : Quot (@List.Perm (Nat × Nat))) : Claims.Finder :=
  ofOut fun n => (Multiset.toList s).find? fun p => p.1 == n

theorem big : ∃ k : Nat, k > 3 := Classical.byContradiction fun h => h ⟨4, by decide⟩

/-- A finder that uses choice inside a proof only. -/
def inProof : Claims.Finder := ofOut fun n => some ((⟨5, big⟩ : { _k : Nat // ∃ m : Nat, m > 3 }).1, n)

/-- A loop with `while`: core writes it with `Lean.Loop.forIn`. -/
def loopedOut (n : Nat) : Option (Nat × Nat) := Id.run do
  let mut i := 0
  let mut acc := 0
  while i < n do
    acc := acc + i
    i := i + 1
  return some (acc, i)

def looped : Claims.Finder := ofOut loopedOut

/-- A parse with `String.toNat?`: core writes it with `WellFounded.extrinsicFix`. -/
def parsedOut (n : Nat) : Option (Nat × Nat) := (toString n).toNat?.map fun m => (m, m + 1)

def parsed : Claims.Finder := ofOut parsedOut

/-- A finder whose cost is chosen: of `Claims.Finder.mk` the walk reads `out` alone. -/
noncomputable def costChosen : Claims.Finder :=
  ⟨Nat, fun n => some (n, n + 1), fun _ => Classical.choose (⟨0, rfl⟩ : ∃ k : Nat, k = 0)⟩

end Finders
