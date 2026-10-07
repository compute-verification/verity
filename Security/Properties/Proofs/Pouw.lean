import Properties.Pouw

/-!
# Proofs of PoUW's window properties (`Properties.Pouw`)

The served numbers, by evaluation in the kernel (`decide +kernel`: no compiled code, only the kernel's reduction), and
the refusals, by unfolding. The rest of `Properties.Pouw` is open.
-/

namespace Properties.Pouw.Window

open PouwWindow

theorem levelsServed : LevelsServed := by
  unfold LevelsServed; decide +kernel

theorem bucket8192 : Bucket8192 := by
  unfold Bucket8192; decide +kernel

theorem kPrimeServed : KPrimeServed := by
  unfold KPrimeServed; decide +kernel

theorem kPrimeExactAtEquality : KPrimeExactAtEquality := by
  unfold KPrimeExactAtEquality; decide +kernel

theorem publicItems_eq : PublicItems := by
  unfold PublicItems; decide +kernel

theorem kPrimeRefuses : KPrimeRefuses := by
  intro eps δ credited n w
  by_cases h : KPrimeDomain eps δ credited n w <;>
    simp only [kPrime, h, ↓reduceIte, decide_true, decide_false] <;> (try split) <;> rfl

theorem gridRefusals : GridRefusals := by
  intro bits n lo
  have fp : ∀ m : Int, floorPoint? bits m = if 1 ≤ m then .ok (floorPoint bits m.toNat) else
      .error s!"the grid is over positive integers, not {m}" := fun _ => rfl
  have pt : ∀ m : Int, point? bits m = if 0 ≤ m then .ok (point bits m.toNat) else
      .error s!"a level is a natural number, not {m}" := fun _ => rfl
  by_cases h1 : 1 ≤ n <;> by_cases h0 : 0 ≤ n <;> by_cases hl : 1 ≤ lo <;> cases hg : onGrid bits n.toNat <;>
    simp [pad?, bucket?, level?, onGridInt, Work.at?, levels?, fp, pt, h1, h0, hl, hg, Except.isOk, Except.toBool,
      bind, Except.bind, pure, Except.pure] <;> omega

end Properties.Pouw.Window
