import PouwPearlPlus.Rule
import Proofs.Pouw.PearlPlus.Words

/-!
# Pearl+'s bridge, part (c): the check's per-row rules and debit replay are the game's

`Rule.checkOpened` decides each opened row by `row_passes` (the floor on the exact product ρ·α, and `live_row` on the exact
statistic), and the cap by `word_debit`, a word's flagged atoms and its later promotions into a +0 total. The game reads
`PearlCSem.RowOK` (the FMA floor `ρ·α − 1 ≥ 0` and `LiveRow` at δ = 1 under `Stat.every8`) and the tile debit
`tileDebitRev1`, on `Skip.unitFlags` and `laterZeroPromos`.

The floor agrees because ρ and α are finite FP32 words: `ρ·α − 1`, when nonzero, is at least 2⁻⁴⁸ in magnitude, far above
FP32's least subnormal, so the FMA's rounding keeps its sign. The tile's cap is not stated here: the check's cap leaves
out the game's forming term and reads its credited rows and columns, where `TileGood` reads every column and
`passRows`, so the cap's statement waits on the restated credit set and acceptance event.
-/

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus

/-- **(c) The per-row rules are `RowOK`**: on a row of `k` finite FP32 words whose `s5_stats` are a value, `row_passes`
on its ρ and α holds exactly when the semantics' `RowOK` does. -/
def RowsAreTheRules : Prop :=
  ∀ {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (p k : ℕ) (W : Codes) (i : ℕ) (x : Array ℕ)
    (s ρw αw βw : ℕ), i < p → 0 < k → x.size = k →
    (∀ l < k, x[l]! = W i l ∧ W i l < 2 ^ 32 ∧ (wordQ (W i l)).isSome) →
    Words.s5Stats x = .ok (s, ρw, αw, βw) →
      (Rule.rowPasses ρw αw x = true ↔ (semH3 saltOf rootOf : PearlCSem Q R S).RowOK p k W i)

/-- **(c) A word's replayed debit is the game's**: where `word_debit` is a value on a word's two rows of codes, its flagged
atoms are `Skip.unitFlags`' and its later promotions into a +0 total are `laterZeroPromos`' at the H100 record. -/
def DebitIsTheReplay : Prop :=
  ∀ {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Pouw.Fp8Atom.Code) (B : Matrix (Fin n) (Fin k) Pouw.Fp8Atom.Code) (i : Fin m) (j : Fin n) (f z : ℕ),
    Rule.wordDebit (Array.ofFn fun l => (A i l).val) (Array.ofFn fun l => (B j l).val) = .ok (f, z) →
      f = ((Finset.range (k / 32)).filter fun t => Pouw.PearlC.Skip.unitFlags A B i j t = true).card ∧
        z = Pouw.PearlC.laterZeroPromos Pouw.PearlC.devH100 A B i j

end Pouw.PearlPlus.Bridge
