import Definitions.Pouw.Fp8Atom.H1TLemmas
import Proofs.Pouw.Fp8Atom.Fp32Proofs

/-!
# H-1T: the statements `H1TRunningWords` still needs (staging, unpinned)

`H1TRunningWords` (`H1TRunningAssumptions`) is the `Hard` pairs' separation and the `Stuck` words' non-constancy. The
route of `internal/pouw-fp8/h1t-running-words-blocker.md` in the store takes its cases to three statements. None is
proved or pinned, and no pin reads them.

* **`RnMonoChainGap`** carries the `Stuck` words, the pairs of one column, and the pairs against a word that does not
  read slice `T − 1` (`SkipsLast`, or `I_(T−1)` with slice `T − 2` flipped). Slices `0..T−2` take the salt that makes
  every block-2 word one-signed and at least 1,792 (tag signs against the tuple, real signs against `b`, mantissa 7),
  and all three tag lanes of slice `T − 1` flip, which moves `R_(T−1)` by at least 2,816 (2,304 when slice `T − 2` is
  the flipped one). The two chains then share their block-2 increments, and the statement says they never meet.
  `merge_one_ulp` shows the gap must be larger than one top ulp.
* **`H1TLastSliceSplit`** carries O1 (two block-1 running words at one position): whatever the two chains' values
  before slice `p`, some salt of slice `p` separates the two columns' next running words.
* **`H1TCrossLate`** is the rest: a block-2 running word against another column's running word at a position from
  `T − 1` on. Only the block-1 tag differences of the slices whose block 2 is not yet read separate such a pair (each
  slice's block-2 tag products are its block-1 ones negated), and at `R_(2T−3)` those are two slices' `32·v`, below one
  ulp at `|R| ≥ 2^32`. `H1TCrossLateStaging` proves it where block 2 adds exactly on the 512 grid and the last two
  slices are quiet (`Exact512`), a regime that is empty for `k ≤ 19,430`.
-/

namespace Pouw.Fp8Atom.H1T

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs

/-- The FP32 chain from `a` through the increments `cs`: `RN(· + c)` for each `c` in turn. -/
def chainFrom (a : ℚ) (cs : List ℚ) : ℚ := cs.foldl rnAdd a

/-- **The monotone chain gap**: two chains from representable `0 ≤ b` and `a ≥ b + 2304`, with common increments in
`[1792, 738304]` and every value of `a`'s chain below `2^33`, never meet. -/
def RnMonoChainGap : Prop :=
  ∀ (a b : ℚ) (cs : List ℚ), Representable a → Representable b → 0 ≤ b → b + 2304 ≤ a →
    (∀ c ∈ cs, 1792 ≤ c ∧ c ≤ 738304) → (∀ n ≤ cs.length, chainFrom a (cs.take n) < 2 ^ 33) →
    chainFrom b cs ≠ chainFrom a cs

/-- **The last-slice split**: for finite codes, two columns with different tuples, and any two representable chain
values `A`, `A'` below `2^33 − 2^29`, some salt of the slice separates `RN(A + I)` and `RN(A' + I')`, with `I`, `I'` the
two columns' block-1 words of the slice. The bound covers every running word of a legal unit (`run_bound`: at most
`1130 · 6,881,792 − 256 < 2^33 − 2^29`); red-team Phase 19r accepted it in place of `2^33`. -/
def H1TLastSliceSplit : Prop :=
  ∀ (xs : Fin 29 → Code) (b b' : ℕ → Code) (j j' τ : ℕ) (A A' : ℚ), (∀ l, (xs l).isNaN = false) →
    (∀ r, (b r).isNaN = false) → (∀ r, (b' r).isNaN = false) → tuple j ≠ tuple j' →
    Representable A → Representable A' → |A| < 2 ^ 33 - 2 ^ 29 → |A'| < 2 ^ 33 - 2 ^ 29 →
    ∃ σ : SliceSalt, rnAdd A (wordVal (hopperStep 0 (formSlice xs σ).1 (colLanes b j τ))) ≠
      rnAdd A' (wordVal (hopperStep 0 (formSlice xs σ).1 (colLanes b' j' τ)))

/-- **The cross-column late pairs**: on a legal unit, a block-2 running word of column `j` and a running word of
another column `j'` at a position from `T − 1` on, not `OppLater`, are different functions of the salt. -/
def H1TCrossLate : Prop :=
  ∀ U : Unit, U.Legal → ∀ (i : Fin U.m) (a a' : Fin U.n × Pos U.T), a.1 ≠ a'.1 → Run2 a.2 →
    a'.2.isRight = true → U.T ≤ posIdx a'.2 + 1 → ¬ OppLater a.1.val a'.1.val a.2 a'.2 →
    U.word (i, a) ≠ U.word (i, a')

/-- A step of at least one ulp strictly raises a nonzero representable value. -/
theorem lt_rnAdd {r c : ℚ} (hr : Representable r) (h0 : r ≠ 0) (hc : 0 ≤ c) (hu : ulp r ≤ c) : r < rnAdd r c := by
  have hle : r ≤ rnAdd r c := by
    have := rnAdd_mono r 0 r c (by linarith)
    rwa [rnAdd_exact r 0 (by simpa using hr), add_zero] at this
  exact lt_of_le_of_ne hle (Ne.symm (rnAdd_moves r c hr h0 (by rwa [abs_of_nonneg hc])))

/-- **One top ulp merges in one step**: `2^32 − 1024` and `2^32 − 512` are representable, and adding 1,792 sends
both to `2^32 + 1024` (both sums are ties on the 512 grid, and `2^32 + 1024` is the even neighbour). -/
theorem merge_one_ulp :
    Representable ((2 : ℚ) ^ 32 - 1024) ∧ Representable ((2 : ℚ) ^ 32 - 512) ∧
      rnAdd ((2 : ℚ) ^ 32 - 1024) 1792 = (2 : ℚ) ^ 32 + 1024 ∧
      rnAdd ((2 : ℚ) ^ 32 - 512) 1792 = (2 : ℚ) ^ 32 + 1024 := by
  refine ⟨⟨2 ^ 22 - 1, 10, by norm_num, by norm_num⟩, ⟨2 ^ 23 - 1, 9, by norm_num, by norm_num⟩, ?_, ?_⟩ <;>
    decide +kernel

/-! ## A late pair the last two slices do not separate

The unit of the store's `h1t-running-sims/AbsorbUnit.lean`: `k = 32,768` (`T = 1,130`), `x = b = 448` on slices
`0..1127` and 0 on slices 1128 and 1129, columns 21 (tuple 416, 32, 32) and 22 (448, 32, 32). Every word of slices
`0..1127` is a multiple of 512 for every salt, and the chain stays in `[2^32, 2^33)`. So
`R_2257(j) = X + RN(RN(A_j + i1) + i2) − A_j`, with `X` common to both columns, `A_j = R_1127(j)` a multiple of 512, and
`i1`, `i2` the words of slices 1128 and 1129, which read only their tag fields. With both of those slices' tag fields
set to `absorbSalt`, the two columns' words there are 3,872 and 4,128 (`absorb_words`), and for either parity of
`A/512` both round twice to `A + 8192` (`absorb_rounds`). So `R_2257(21) = R_2257(22)` for every salt that has these
tag fields. -/

/-- Real codes 0 on every lane of a slice. -/
def zeroSlice : Fin 29 → Code := fun _ => ⟨0, by norm_num⟩

/-- A column's real codes 0 everywhere. -/
def zeroCol : ℕ → Code := fun _ => ⟨0, by norm_num⟩

/-- Tag signs `(+, +, +)` and mantissas `(0, 0, 1)`, real sign bits clear. -/
def absorbSalt : SliceSalt := ⟨fun _ => false, fun _ => false, pick3 0 0 1⟩

theorem absorb_words :
    wordVal (hopperStep 0 (formSlice zeroSlice absorbSalt).1 (colLanes zeroCol 21 1128)) = 3872 ∧
      wordVal (hopperStep 0 (formSlice zeroSlice absorbSalt).1 (colLanes zeroCol 22 1128)) = 4128 := by
  constructor <;> decide +kernel

theorem absorb_rounds :
    rnAdd (rnAdd (2 ^ 32 + 2 ^ 31) 3872) 3872 = 2 ^ 32 + 2 ^ 31 + 8192 ∧
      rnAdd (rnAdd (2 ^ 32 + 2 ^ 31 + 512) 3872) 3872 = 2 ^ 32 + 2 ^ 31 + 512 + 8192 ∧
      rnAdd (rnAdd (2 ^ 32 + 2 ^ 31) 4128) 4128 = 2 ^ 32 + 2 ^ 31 + 8192 ∧
      rnAdd (rnAdd (2 ^ 32 + 2 ^ 31 + 512) 4128) 4128 = 2 ^ 32 + 2 ^ 31 + 512 + 8192 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> decide +kernel

end Pouw.Fp8Atom.H1T
