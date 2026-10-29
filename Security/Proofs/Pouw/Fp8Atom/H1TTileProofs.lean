import Proofs.Pouw.Fp8Atom.H1TRunningProofs
import Definitions.Pouw.Assumptions.Fp8Atom.H1TTileAssumptions

/-!
# `TTH1T`'s two form checks

* **`tth1tAllRight`** (under `H1TRunningWords`): a program right on every tile at every salt holds a register equal to
  every credited word; a slot past the last register would make the word the constant `0`, which is free. So
  `h1tGammaRunning` bounds its cost by 32 per credited word, and every credited word is in a right tile.
* **`tth1tSat`**: at `γ = 1` the event is `cost < 0`, which never holds, so `TTH1T r c 1 0` holds for every domain.
-/

namespace Pouw.Fp8Atom.H1T.Proofs

open Pouw.Fp8Atom Pouw.Fp8Atom.H1T Pouw.Dimension.H100

theorem tth1tAllRight (hR : H1TRunningWords) : TTH1TAllRight := by
  intro pr hpr N Us hL C tl fpSem L P slot hLf hP hall G
  have hfree := (h1tUnitsDistinct (h1tDistinctLive hR) N Us hL).2
  have hout : ∀ x : Idx Us, ∃ r ∈ regs100 fpSem P L, ∀ G, r G = mword Us x G := by
    intro x
    have e : slotVal (regs100 fpSem P L) slot x = mword Us x := funext fun G => hall G (tl.of x) x rfl
    by_cases hs : slot x < (regs100 fpSem P L).length
    · refine ⟨(regs100 fpSem P L)[slot x], List.getElem_mem hs, fun G => ?_⟩
      rw [← e]; unfold slotVal; rw [List.getD_eq_getElem _ _ hs]
    · refine absurd (Or.inl ⟨0, ?_⟩) (hfree x)
      rw [← e]; unfold slotVal; funext G; rw [List.getD_eq_default _ _ (not_lt.1 hs)]
  have h := h1tGammaRunning hR pr hpr N Us hL fpSem L P hLf hP hout
  have hr : rightWords (regs100 fpSem P L) slot tl G = ∑ u, (Us u).m * (Us u).n * Fintype.card (Pos (Us u).T) := by
    have : {x : Idx Us | TileRight (regs100 fpSem P L) slot tl (tl.of x) G} = Set.univ :=
      Set.eq_univ_of_forall fun x => hall G (tl.of x)
    unfold rightWords
    rw [this, Set.ncard_univ, Nat.card_eq_fintype_card]
    simp only [Fintype.card_sigma, Fintype.card_prod, Fintype.card_fin, mul_assoc]
  rw [hr]
  exact not_lt.2 h

theorem tth1tSat (r c : ℕ) : TTH1T r c 1 fun _ => 0 := by
  intro pr _ N Us _ C tl _ fpSem L P slot _ _
  have : {G : MSalt Us | (cost100 pr P : ℚ) < (1 - 1) * 32 * rightWords (regs100 fpSem P L) slot tl G} = ∅ :=
    Set.eq_empty_of_forall_notMem fun G (hG : (cost100 pr P : ℚ) < (1 - 1) * 32 * _) => by
      rw [sub_self, zero_mul, zero_mul] at hG
      exact absurd hG (not_lt.2 (Nat.cast_nonneg _))
  rw [this, Set.ncard_empty]
  simp

end Pouw.Fp8Atom.H1T.Proofs
