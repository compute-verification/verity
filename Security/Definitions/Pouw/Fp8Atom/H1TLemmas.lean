import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.Floor.Ring
import Mathlib.Data.Fintype.Pi
import Definitions.Pouw.Fp8Atom.H1T

/-!
# H-1T's M3 statements (trusted)

M3 of `internal/pouw-fp8/lean-atom-m2-plan.md` (§0.6): the lemmas the one-flip clauses of `DistinctLiveH1T` rest on.

* **`H1TAtomValue`**: on finite codes, the `HOPPER_E4M3_K32` step from `+0` writes the exact sum after two truncations
  toward zero: each product to the grid `2^(E − 13)` of the slice's alignment exponent `E` (`gridExp`, the largest
  exponent of a nonzero product), and the sum of those to 14 significant bits (`trunc14`).
* **`Trunc14Mono`** (L7, `normalize` is monotone) and **`Trunc14Cell`** (L3, the cell): `trunc14` keeps order, moves a
  nonzero `y` by less than `2^(binade |y| − 13)`, and two values with the same truncation lie that close.
* **`H1TSlice`** (L1 to L3 on one slice): both blocks' sums are below `2^21·F` (block 1's at most 6,142,976, block 2's at
  most 738,304), and flipping a tag lane's sign bit keeps the alignment exponent and moves each block's sum by exactly
  twice the tag product, which is on the grid.
* **`H1TChainMono`** (L7 for the chain): the running words are monotone in the fresh-step words.
* **`H1TRange`** (L4): on a legal unit every fresh-step word is finite, block 1's are at most 6,142,976 and block 2's at
  most 738,304 in magnitude, and every running word is below `2^33`, so its ulp is at most 512. This holds at
  `x = ±448`: the legal inputs are all finite codes.
* **`H1TFormCover`** (X-M2-4 (a)): a slice of finite codes has one of the five `F` of `H1TFormTable`'s keys, so the
  table covers every slice.

# M4a's statement, narrowed in M4b and M4c (trusted)

* **`H1TOneFlip`**: the half-salt clause (`HalfApartH1T`) for every pair of distinct credited words of a row that is not
  `Hard`, and for every word that is not `Stuck`. One flip of one tag sign bit separates each such pair and moves each
  such word. `Hard` reads the two words' columns as well as their positions. It is O1 and O2 of the plan's §0.4, less
  what one flip separates by column and near the start of block 2:
  * O1, two block-1 running words at one position, unless their columns' tuples are at least 72 apart in some lane
    (`FarTags`), or have opposite signs in a lane where one of them is at least 36 in magnitude (`OppLater`, either
    way round);
  * O2, a block-2 running word against any word but a fresh-step word at a later position, unless the other word is
    a block-1 word of another column in a later slice, and the two tuples have opposite signs in a lane where the
    other column's value is at least 36 in magnitude if the other word is a running word (`OppLater`); or unless the
    running word is `Early` and the other word does not read the last slice (`SkipsLast`).
* **`Stuck`**: a block-2 running word that is not `Early`, the words `H1TOneFlip` does not show to move. An `Early`
  word `R_p` has a tag lane of magnitude at least `32·(p − T + 2) + 4`, so a flip of that lane of slice `T − 1` moves
  it through all `p − T + 2` roundings between.
-/

namespace Pouw.Fp8Atom.H1T

open Pouw.Fp8Atom

/-! ## M4a: the one-flip clauses -/

/-- A credited position's index: `p` for `I_p` (`inl`) and for `R_p` (`inr`). -/
def posIdx {T : ℕ} : Pos T → ℕ := Sum.elim (fun p => p.val) (fun p => p.val.val)

/-- A block-2 running word: `R_p` with `p ≥ T`. -/
def Run2 {T : ℕ} (a : Pos T) : Prop := a.isRight = true ∧ T ≤ posIdx a

/-- Columns `j` and `j'` whose tuples are at least 72 apart in some tag lane. -/
def FarTags (j j' : ℕ) : Prop := ∃ k : Fin 3, 72 ≤ |val (tuple j k) - val (tuple j' k)|

/-- For a running word `a` of column `j` before slice `posIdx a'`'s block 2: `a'` is a block-1 word of column `j'` in a
slice after `a`'s last block-2 slice (`posIdx a − T < posIdx a' < T`), and some tag lane has tuple values of opposite
signs in the two columns, with column `j'`'s at least 36 in magnitude if `a'` is a running word. -/
def OppLater {T : ℕ} (j j' : ℕ) (a a' : Pos T) : Prop :=
  posIdx a' < T ∧ posIdx a < T + posIdx a' ∧
    ∃ k : Fin 3, val (tuple j k) * val (tuple j' k) < 0 ∧ (a'.isLeft = true ∨ 36 ≤ |val (tuple j' k)|)

/-- A block-2 running word `R_p` of column `j` near the start of block 2: some tag lane has magnitude at least
`32·(p − T + 2) + 4`. -/
def Early {T : ℕ} (j : ℕ) (a : Pos T) : Prop :=
  ∃ k : Fin 3, 32 * ((posIdx a + 2 - T : ℕ) : ℚ) + 4 ≤ |val (tuple j k)|

/-- Word `a` does not read slice `T − 1`: a word of an earlier slice, or a block-2 fresh-step word before the last. -/
def SkipsLast {T : ℕ} (a : Pos T) : Prop :=
  posIdx a + 1 < T ∨ (a.isLeft = true ∧ T ≤ posIdx a ∧ posIdx a + 1 < 2 * T)

/-- A block-2 running word of column `j` that is not `Early`. -/
def Stuck {T : ℕ} (j : ℕ) (a : Pos T) : Prop := Run2 a ∧ ¬ Early j a

/-- **The pairs one flip does not separate**, of word `a` of column `j` and word `a'` of column `j'` (O1 and O2 of the
plan's §0.4, narrowed by column in M4b and M4c): two block-1 running words at one position whose columns are neither
`FarTags` nor `OppLater` either way round, and a block-2 running word against any word except a fresh-step word at a
later position, the block-1 words `OppLater` names, and, if the running word is `Early`, the words that skip the last
slice. -/
def Hard {T : ℕ} (j j' : ℕ) (a a' : Pos T) : Prop :=
  (a.isRight = true ∧ a'.isRight = true ∧ posIdx a < T ∧ posIdx a = posIdx a' ∧ ¬ FarTags j j' ∧
      ¬ OppLater j j' a a' ∧ ¬ OppLater j' j a' a) ∨
    (Run2 a ∧ ¬(a'.isLeft = true ∧ posIdx a < posIdx a') ∧ ¬ OppLater j j' a a' ∧ ¬(Early j a ∧ SkipsLast a')) ∨
    (Run2 a' ∧ ¬(a.isLeft = true ∧ posIdx a' < posIdx a) ∧ ¬ OppLater j' j a' a ∧ ¬(Early j' a' ∧ SkipsLast a))

end Pouw.Fp8Atom.H1T
