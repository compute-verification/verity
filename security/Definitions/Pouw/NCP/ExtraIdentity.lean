module

public import Definitions.Pouw.NCP.Protocol

/-! An admissible `R` with a checkpoint identity other than the final one, so the rejection rule alone does not give `FinalOnly` with one identity per output entry. -/

@[expose] public section

namespace Pouw.NCP.ExtraIdentity

open Finset

def AdmissibleExtraIdentity : Prop :=
  ∀ k : ℕ, 128 ≤ k → 64 ∣ k → ∃ R : Mat, Admissible k 16 R ∧
    (∀ (E1 F1 : Mat) (i j : ℕ),
      ∑ l ∈ range (k + 64), stackX k 0 E1 R i l * stackY k 0 R F1 l j =
        ∑ l ∈ range (k + 32), stackX k 0 E1 R i l * stackY k 0 R F1 l j +
          ∑ l ∈ range 32, stackX k 0 E1 R i l * stackY k 0 R F1 l j) ∧
    (∃ E1 F1 : Mat, (∀ i l, InNoise6 (E1 i l)) ∧ (∀ l j, InNoise6 (F1 l j)) ∧
      ∑ l ∈ range 32, stackX k 0 E1 R 0 l * stackY k 0 R F1 l 0 ≠ 0) ∧
    ∀ t : ℕ, 1 ≤ t → 64 * t ≤ k → ∀ (E1 F1 : Mat) (i j : ℕ),
      ∑ l ∈ range (2 * k + 64 * t), stackX k 0 E1 R i l * stackY k 0 R F1 l j = 0

end Pouw.NCP.ExtraIdentity
