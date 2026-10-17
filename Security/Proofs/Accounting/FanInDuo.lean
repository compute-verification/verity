import Proofs.Pouw.Receipt.Run

/-!
# The fan-in floor for duo and fx1

Every word of a duo or fx1 run is essential to a program whose two outputs are its final state `(y, z)`: the pair is a
fold of steps injective in word and state (`Pouw.ReceiptProofs.duoWord_word_injective`, `duoWord_state_injective`),
so `Accounting.FanInProofs.essential_of_foldl` applies, and `le_of_essential` gives `N ≤ 2 + (f − 1)·m`.
-/

namespace Accounting.FanInProofs

open Accounting.Program Pouw.Receipt Pouw.ReceiptProofs

theorem duoFoldFanIn {w N f m : ℕ} (k a : Fin N → BitVec (w + 1)) (r : Fin N → ℕ)
    (s : BitVec (w + 1) × BitVec (w + 1)) (P : Prog (BitVec (w + 1)) N f m) (out : Fin 2 → Fin (N + m)) :
    (∃ d, (P.run out d 0, P.run out d 1) ≠ duoRun k a r s d) ∨ N ≤ 2 + (f - 1) * m := by
  by_cases hP : ∃ d, (P.run out d 0, P.run out d 1) ≠ duoRun k a r s d
  · exact Or.inl hP
  simp only [not_exists, not_not] at hP
  exact Or.inr (le_of_essential P out (essential_of_foldl (P.run out) (fun v => (v 0, v 1))
    (fun j t u => duoWord (k j) (a j) (r j) t u) (fun j t => duoWord_word_injective (k j) (a j) (r j) t)
    (fun j u => duoWord_state_injective (k j) (a j) (r j) u) s hP))

theorem fx1FoldFanIn {w N f m : ℕ} (m₁ : Fin N → OddKey (w + 1)) (k a : Fin N → BitVec (w + 1)) (r : Fin N → ℕ)
    (s : BitVec (w + 1) × BitVec (w + 1)) (P : Prog (BitVec (w + 1)) N f m) (out : Fin 2 → Fin (N + m)) :
    (∃ d, (P.run out d 0, P.run out d 1) ≠ fx1Run (fun j => (m₁ j).1) k a r s d) ∨ N ≤ 2 + (f - 1) * m := by
  by_cases hP : ∃ d, (P.run out d 0, P.run out d 1) ≠ fx1Run (fun j => (m₁ j).1) k a r s d
  · exact Or.inl hP
  simp only [not_exists, not_not] at hP
  exact Or.inr (le_of_essential P out (essential_of_foldl (P.run out) (fun v => (v 0, v 1))
    (fun j t u => duoWord (k j) (a j) (r j) t (u * (m₁ j).1))
    (fun j t _ _ huv => mul_odd_cancel (m₁ j).2 (duoWord_word_injective (k j) (a j) (r j) t huv))
    (fun j u => duoWord_state_injective (k j) (a j) (r j) (u * (m₁ j).1)) s hP))

end Accounting.FanInProofs

namespace Accounting.SecurityProofs

theorem DuoFoldFanIn : Accounting.Guarantees.DuoFoldFanIn :=
  fun _ _ _ _ k a r s P out => Accounting.FanInProofs.duoFoldFanIn k a r s P out

theorem Fx1FoldFanIn : Accounting.Guarantees.Fx1FoldFanIn :=
  fun _ _ _ _ m₁ k a r s P out => Accounting.FanInProofs.fx1FoldFanIn m₁ k a r s P out

end Accounting.SecurityProofs
