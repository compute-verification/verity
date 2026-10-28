import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fintype.Prod
import Mathlib.Data.Fintype.Sum
import Definitions.Pouw.Fp8Atom.H1TVectors

/-!
# H-1T's pinned statements (trusted)

The map `DistinctLiveH1T` is about (`Pouw.Fp8Atom.H1T`), the facts
about it that do not need the Target A proof, and the FP32 lemmas that proof uses.

The map is checked against three other computations (`verity/protocols/accounting/work/pouw/lean/scripts/h1t_vectors.py`;
`Definitions/Pouw/Fp8Atom/H1TCoverage.json`).
The generator checks every expected value below against an exact transcription of §3.1 over fractions,
`h1t_ref` and `chain` (`h1_tagged.py`, the pinned map), and a float16 simulation of the kernel `h1t_form_xq`, before
it writes them.

* **`H1TFormTable`**: `formReal` on every finite code, both signs and every `F` in {1/4, 1/2, 1, 2, 4}: 2,540 lanes.
  Every slice of finite codes has one of these `F`, because its maximum is at most 448.
* **`H1TTagTable`**: every tag value `±32F·(1 + m/8)` and its negation are E4M3 codes, exactly.
* **`H1TSaltConformance`**: on 74 raw salts (zero; all ones; bit 0, 7, 8, 9, 14, 15, 16, 22 to 26, 30 or 31 set in
  every word; each word all ones alone; 40 random), `extract` reads the fields the kernel's masks read.
* **`H1TChainConformance`**: on 31 rows and columns with `T ≤ 3` (random, zero-padded last slices, `x = ±448` and
  `±416`, zero rows, subnormal rows, tuples at the digit boundaries `j` = 0, 22, 23, 1,425, 1,426 and 88,411, one row
  against four columns), `freshWord` writes the fresh-step words `verity.ml.tc` writes, and `runVal` is the value of
  numpy's float32 running sum, at every position.

The structure of the map:

* **`H1TReadsFields`**: every credited word reads the salt only through `extract`, the fields the kernel reads.
* **`H1TNotRawWord`**: no credited word is a raw XOF word, because no read bit is below bit 7. With "no credited word
  is constant", which is part of `DistinctLiveH1T`, this is "no credited word is free".
* **`H1TTupleInjective`** and **`H1TTupleWrap`**: the 88,412 = 23·62·62 columns of a unit carry distinct tag tuples,
  and the tuples repeat after 88,412.
* **`H1TWrapDuplicate`**: so within a unit of more than 88,412 columns, two columns 88,412 apart with the same real
  codes have the same credited words on every salt. `DistinctLiveH1T` needs `n ≤ 88,412` per unit, and larger outputs
  need several units with their own salts (change 4).
* **`H1TPosCard`**: `4T − 3` credited words per output for `T ≥ 2` (change 2).
* **`H1TNonVacuous`**: a legal unit at `k = 32,768` and `n = 88,412` exists (`T = 1,130`, 4,517 credited words per
  output); the proof's witness has every code of `X_q` and B′ equal to 1.0 (change 9).

The FP32 add (`Pouw.Fp8Atom.Fp32`):

* **`RnErr`**: `rn` is within half an ulp of `y`.
* **`RnAddMovesHalf`**: a summand of more than half an ulp of a nonzero representable `r` moves the sum. This is the
  absorption step of the one-flip argument (absorbed implies `2·|i| ≤ ulp r`, at `±2^k` too), which `RnAddMoves` alone
  leaves open on `[960, 1,024)`.
* **`HalfOfInvolution`**: a predicate that an involution always falsifies holds on at most half of a finite set; the
  counting step of the half-salt clause's one-flip cases.
-/

namespace Pouw.Fp8Atom.H1T

open Pouw.Fp8Atom

/-- **The real-lane table**: its keys are every finite code, both signs and every `F` index, and on each of them
`formReal` gives the expected codes. -/
def H1TFormTable : Prop :=
  formTable.map (fun v => (v.x, v.pos, v.fi)) = formKeys ∧ ∀ v ∈ formTable,
    (formReal (fOf v.fi) (codeOf v.x) v.pos).1.val = v.x1 ∧ (formReal (fOf v.fi) (codeOf v.x) v.pos).2.val = v.x2

/-- **The tag table**: its keys are every `F` index, sign and mantissa, and each tag value and its negation encode
exactly to the expected codes. -/
def H1TTagTable : Prop :=
  tagTable.map (fun v => (v.fi, v.neg, v.m)) = tagKeys ∧ ∀ v ∈ tagTable,
    (enc (tagVal (fOf v.fi) v.neg (fin8 v.m))).val = v.c1 ∧ (enc (-tagVal (fOf v.fi) v.neg (fin8 v.m))).val = v.c2 ∧
    val (codeOf v.c1) = tagVal (fOf v.fi) v.neg (fin8 v.m) ∧ val (codeOf v.c2) = -tagVal (fOf v.fi) v.neg (fin8 v.m)

/-- **The salt fields**: on at least 74 raw salts, `extract` reads the expected fields. -/
def H1TSaltConformance : Prop :=
  74 ≤ saltVectors.length ∧ ∀ v ∈ saltVectors, packSalt (extract (rawOf v.r)) = (v.pos, v.neg, v.man)

/-- **The chain**: on at least 31 rows and columns, at every position `p < 2T`, `freshWord` writes the expected
fresh-step word and `runVal` is the value of the expected running word. -/
def H1TChainConformance : Prop :=
  31 ≤ rowVectors.length ∧ ∀ v ∈ rowVectors, ∀ p < 2 * v.T,
    freshWord v.T v.xs v.bs v.j v.σ p = v.I.getD p 0 ∧ wordQ (v.R.getD p 0) = some (runVal v.T v.xs v.bs v.j v.σ p)

/-- **The words read only the fields**: two salts with the same fields give every credited word the same value. -/
def H1TReadsFields : Prop :=
  ∀ (U : Unit) (g g' : U.Salt), (∀ i τ, extract (g i τ) = extract (g' i τ)) → ∀ c, U.word c g = U.word c g'

/-- **No credited word is a raw XOF word.** -/
def H1TNotRawWord : Prop :=
  ∀ (U : Unit) (c : Fin U.m × Fin U.n × Pos U.T) (i : Fin U.m) (τ : Fin U.T) (ℓ : Fin 18),
    U.word c ≠ fun g => ((g i τ ℓ : ℕ) : ℚ)

/-- **The tuples are distinct** on the 88,412 columns of a unit. -/
def H1TTupleInjective : Prop := ∀ j j' : ℕ, j < 88412 → j' < 88412 → tuple j = tuple j' → j = j'

/-- **The tuples repeat** after 88,412 columns. -/
def H1TTupleWrap : Prop := ∀ j : ℕ, tuple (j + 88412) = tuple j

/-- **Columns 88,412 apart collide**: in one unit, two columns 88,412 apart with the same real codes have the same
credited words on every salt. -/
def H1TWrapDuplicate : Prop :=
  ∀ (U : Unit) (j j' : Fin U.n), j'.val = j.val + 88412 → (∀ r, U.b r j = U.b r j') →
    ∀ (i : Fin U.m) (a : Pos U.T), U.word (i, j, a) = U.word (i, j', a)

/-- **`4T − 3` credited words per output**, for `T ≥ 2`. -/
def H1TPosCard : Prop := ∀ T : ℕ, 2 ≤ T → Fintype.card (Pos T) = 4 * T - 3

/-- **Non-vacuity**: a legal unit with `k = 32,768` and `n = 88,412` exists; it has `T = 1,130` and 4,517 credited
words per output. -/
def H1TNonVacuous : Prop :=
  ∃ U : Unit, U.Legal ∧ U.m = 1 ∧ U.k = 32768 ∧ U.n = 88412 ∧ U.T = 1130 ∧ Fintype.card (Pos U.T) = 4517

/-- **Rounding error**: `rn y` is within half an ulp of `y`. -/
def RnErr : Prop := ∀ y : ℚ, |rn y - y| ≤ ulp y / 2

/-- **Movement past half an ulp**: a summand of more than half an ulp of a nonzero representable `r` changes `r`. -/
def RnAddMovesHalf : Prop := ∀ r i : ℚ, Representable r → r ≠ 0 → ulp r < 2 * |i| → rnAdd r i ≠ r

/-- **Half of a finite set**: if an involution `ψ` maps every point where `P` holds to one where it fails, `P` holds on
at most half of the points. -/
def HalfOfInvolution : Prop :=
  ∀ (α : Type) [Fintype α] (ψ : α → α) (P : α → Bool), Function.Involutive ψ → (∀ a, P a = true → P (ψ a) = false) →
    2 * (Finset.univ.filter fun a => P a = true).card ≤ Fintype.card α

end Pouw.Fp8Atom.H1T
