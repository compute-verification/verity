import Definitions.Pouw.NCP.Protocol

/-!
# An admissible R with a checkpoint identity other than the final one (trusted)

A finding against `FinalOnly` for NCP (the R_word-NCP witness, `Pouw.NCP.Word`). At zero input the checked value at
depth `τ` is `Σ_{l < τ} X_{il}·Y_{lj}` with `X = stackX k 0 E₁ R` and `Y = stackY k 0 R F₁`. The statement below says
that for every `k ≥ 128` with `64 ∣ k` some `R` passes all three clauses of `Admissible k 16`, and yet the checked
values at depths `k + 64`, `k + 32` and `32` satisfy the sum rule `c_{k+64} = c_{k+32} + c_{32}` for every noise
`E₁`, `F₁`. That is an identity of the noise at a checkpoint that is not the final one, so the rejection rule alone
does not give `FinalOnly` with one identity per output entry. The third clause shows the rule is not `0 = 0 + 0`:
depth `32` is not identically zero on noise in `[−31, 32]`. The last clause: the checked value at every depth
`2k + 64t` (`1 ≤ t ≤ k/64`) is identically `0`, so the constant rule `0` is an identity there. With the sum rule that
is `k/64 + 1` identity positions per output entry out of `3k/16`, the final one included.
-/

namespace Pouw.NCP.ExtraIdentity

open Finset

/-- **The finding.** -/
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
