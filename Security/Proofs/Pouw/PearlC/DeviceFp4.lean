import Definitions.Pouw.PearlC.DeviceFp4

/-! The proof-only part of `Definitions.Pouw.PearlC.DeviceFp4`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC.Fp4Dev

open Finset Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.Fp4

/-- D-SS's dead block by the salt-free screen (`dead_blocks`): block `b` of 16 along `k` has an element with
`x² > 90·ρ²`, that is `|x| > √90·ρ ≈ 9.49ρ`, `ρ²` over every 8th position (`Stat.every8`). The constant is `90`
exactly, not `9.5² = 90.25`: at it every salt-dead element lies in a screened block at the pinned scales
(`CodeProofs.saltDead4_screened`, any screen of at most `90·ρ²`). The earlier 12ρ screen (`TAU_SQ_INV = 144`) and
10ρ are superseded. The debit doesn't read it (#556 counts screened blocks, `screened_block_pairs`, and charges
none); it is `RowRules4`'s cap. -/
def screenDead {k : ℕ} (x : Fin k → ℚ) (b : ℕ) : Prop :=
  ∃ l : Fin k, (l : ℕ) / 16 = b ∧ 90 * Stat.every8 k x < x l ^ 2

open Classical in
/-- **`RhoDFp4`'s row premise** (`row_passes` before the 10:50Z ruling): D-NF as `RowAdmit4` states it, and D-SS's cap,
at most one block in 64 dead by the screen at `90·ρ²`. It is not the admission rule (`RowAdmit4` is). It keeps the cap
so that `RhoDFp4` is `CodeProofs.RhoDFp4At 90` by `rfl`: `CodeProofs.RowRules4At` states D-NF in the same form. -/
def RowRules4 (beta : (k : ℕ) → (Fin k → ℚ) → ℚ) (k : ℕ) (x : Fin k → ℚ) : Prop :=
  0 ≤ beta k x ∧ 8 ≤ (e4m3 (beta k x)).val ∧ (e4m3 (beta k x)).val < 128 ∧
    64 * ((range (k / 16)).filter fun b => screenDead x b).card ≤ k / 16

open Classical in
/-- **ρ_D = 1/64 at the FP4 forming**: at the row scales `alpha` and `beta`, on any operand of `p` rows of length `k`
(a multiple of 128, as in the domain) whose rows pass `RowRules4`, at most `p·k/64` elements are salt-dead under
`saltDead4`. It is a named `Prop`, taken as a hypothesis where it is used, and it holds at the pinned scales, not at
every `alpha` and `beta`.
- *What it bounds now.* Admission is `RowAdmit4`, so `RhoDFp4` bounds the salt-dead share of the rows that meet the
  old cap. The debit doesn't take it: terms (3) and (4) charge the salt-dead codes themselves.
- *Proved from `PinnedScales`.* bc-a8466279's `Fp4FormingCode` (`internal/pouw/fp4-forming-lean/`) proves
  `CodeProofs.rhoDFp4At_ninety : PinnedScales alpha beta → RhoDFp4At 90 alpha beta`, and
  `CodeProofs.rhoDFp4At_ninety_eq : RhoDFp4At 90 = RhoDFp4` is `rfl`, with `screenDead` at `90·ρ²`. In fact every
  salt-dead element lies in a screened block (`CodeProofs.saltDead4_screened`).
- *The scales' bounds it reads* (`CodeProofs.PinnedScales`): on rows that pass D-NF, `|α·x_l| ≤ 2688` and
  `(128/18769)·(1 − 2⁻¹⁰)²·α²·ρ² ≤ β²`; on every row, `0 ≤ β < 464`.
- *Argued, not proved, for `pearl_c4.stats`.* That the reference's scales meet `PinnedScales` rests on the FP32 error
  bound of `s5_stats`' `ρ²`, about `(k/256 + 13)·u ≈ 270u ≈ 2^-15.9` at `k = 2^16` (`u = 2⁻²⁴`), against the `2⁻¹⁰`
  allowance, and on the numerical table in `internal/pouw/fp4-forming-lean/review-response.md`. There is no Lean lemma
  about `s5_stats` until `s5_stats` is pinned. -/
def RhoDFp4 (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) : Prop :=
  ∀ (p k : ℕ) (X : Matrix (Fin p) (Fin k) ℚ), 128 ∣ k → (∀ i, RowRules4 beta k (X i)) →
    64 * ((univ : Finset (Fin p × Fin k)).filter fun e => saltDead4 alpha beta k (X e.1) e.2).card ≤ p * k

end Pouw.PearlC.Fp4Dev
