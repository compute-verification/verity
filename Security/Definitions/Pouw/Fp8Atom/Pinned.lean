import Mathlib.Data.Fintype.Card
import Definitions.Pouw.Fp8Atom.Vectors
import Definitions.Pouw.Fp8Atom.Vectors.Ada
import Definitions.Pouw.Fp8Atom.Vectors.Add

/-!
# `Pouw.Fp8Atom`'s pinned statements (trusted)

The bit-exact `HOPPER_E4M3_K32` step and its basic facts.

* **`Conformance`**: on each of 631 vectors, `hopperStep` writes the word
  `verity.ml.tc.total_fp8.tc_dot_total_e4m3(HOPPER_E4M3_K32, …)` computes. They are the 49 chained steps of the
  repository's `fp8-hopper-wgmma-k32`/`-k1536` model replays, and the union of two greedy branch covers of the
  10,549 steps of `verity/protocols/accounting/work/pouw/lean/scripts/fp8atom_vectors.py` (`fp8atom_cover.json` beside it), 582 vectors with 15 in common:
  * 450 adversarial vectors, which hit each of its 24 branch outcomes at least 50 times;
  * 147 more, which reach every branch feature the 10,549 reach (739).
  The 10,549 and R1's 166,000 differential steps run in the Lean audit as a compiled test (`Proofs.Pouw.Bulk`, under `runs`), which is
  not a proof. `verity.ml.tc`'s `HOPPER_E4M3_K32` agrees with H100 silicon on two campaigns, with 0 mismatches, but
  their raw words are in neither the repository nor the store, so no H100 capture is included (R1 F2).
* **`AdaCaptures`**: the same adder with two groups of 16 (`adaStep`) reproduces the 360 RTX 4090 captures of
  `mma.sync.m16n8k32.e4m3` in `packages/verity/tests/ml/fixtures/golden/ada_e4m3_m16n8k32.json`: real silicon words
  through the same `groupStepTotal`, `groupSum` and `packValue`.
* **`CodeClasses`**: exactly `0x7F` and `0xFF` are NaN, exactly `0x00` and `0x80` decode to zero, and 254 codes are
  finite.
* **`DecodeInjective`**: two finite codes with the same value are equal, or are the two zeros.
* **`ProductExact`**: the product term's value is the product of the decoded values.
* **`AlignedMono`**: the first truncation is monotone. Aligning two terms to a common exponent `E` (at least theirs)
  on an adder of at most 24 bits preserves `≤` of their values, whatever their exponents.

The FP32 add of the running words, `rnAdd` (`Pouw.Fp8Atom.Fp32`):

* **`AddConformance`**: on each of 1,090 add vectors, the three words are finite and `rnAdd` maps the values of the
  operand words to the value of the word IEEE binary32 addition writes (numpy on the CPU, cross-checked against an
  exact rounding in the generator). They are a greedy cover of the 5,000 of `verity/protocols/accounting/work/pouw/lean/scripts/fp8atom_vectors.py` (random,
  close exponents, cancellation, ties, absorption, `±0 + x`, subnormal operands, near the top of the range, binade
  boundaries). It hits each of 42 branch features of `rnAdd` at least 75 times, or every time it occurs if fewer. Every
  vector's exact sum is zero or normal. The 5,000 run in that compiled test.
* **`RnAddExact`**: a representable sum is returned exactly.
* **`RnAddMono`**: `rnAdd` is monotone in the exact sum.
* **`RnAddAbsorb`**: a summand below half an ulp of a nonzero representable `r` is absorbed, unless `r` is a power of
  two and the summand points toward zero (where the ulp below is half as large).
* **`RnAddMoves`**: a summand of at least an ulp of a nonzero representable `r` moves the sum.
* **`WordsRepresentable`**: every finite FP32 word's value is representable, so `RnAddExact` gives
  `rnAdd 0 x = x` for every word value `x` (`C2FirstRunCopy`'s hypothesis on words).
-/

namespace Pouw.Fp8Atom

/-- **Conformance with `verity.ml.tc`**: at least 631 vectors, the branch covers and the reference steps, and on each of
them `hopperStep` writes the expected word. -/
def Conformance : Prop :=
  631 ≤ vectors.length ∧ ∀ v ∈ vectors, hopperStep v.acc (codes v.a) (codes v.b) = v.d

/-- **The RTX 4090 captures**: `adaStep` writes the measured word on each of the 360 captures. -/
def AdaCaptures : Prop :=
  Vectors.ada.length = 360 ∧ ∀ v ∈ Vectors.ada, adaStep v.acc (codes v.a) (codes v.b) = v.d

/-- **The code classes**: NaN is exactly `0x7F` and `0xFF`; `decode` is `none` exactly on NaN; it is `some 0` exactly
on `0x00` and `0x80`; and 254 codes are finite. -/
def CodeClasses : Prop :=
  (∀ c : Code, c.isNaN = true ↔ c = 0x7F ∨ c = 0xFF) ∧
  (∀ c : Code, decode c = none ↔ c.isNaN = true) ∧
  (∀ c : Code, decode c = some 0 ↔ c = 0x00 ∨ c = 0x80) ∧
  (Finset.univ.filter fun c : Code => c.isNaN = false).card = 254

/-- **`decode` is injective on the finite codes**, but for the two zeros. -/
def DecodeInjective : Prop :=
  ∀ a b : Code, a.isNaN = false → decode a = decode b → a = b ∨ (a.isZero = true ∧ b.isZero = true)

/-- **The product is exact**: for finite codes, the product term's value is the product of the decoded values. -/
def ProductExact : Prop :=
  ∀ (a b : Code) (x y : ℚ), decode a = some x → decode b = some y → (product a b).value = x * y

/-- **The first truncation is monotone**: on an adder of at most 24 bits, terms aligned to an exponent `E` at least
their own keep the order of their values. -/
def AlignedMono : Prop :=
  ∀ (width : Nat) (E : Int) (s t : Term), width ≤ 24 → s.exp ≤ E → t.exp ≤ E → s.value ≤ t.value →
    aligned width E s ≤ aligned width E t

/-- **FP32 add conformance**: at least 1,090 vectors, a branch cover, and on each of them the three words are finite and
`rnAdd` maps the values of `a` and `b` to the value of `c`. -/
def AddConformance : Prop :=
  1090 ≤ Vectors.add.length ∧ ∀ v ∈ Vectors.add,
    ∃ x y z : ℚ, wordQ v.a = some x ∧ wordQ v.b = some y ∧ wordQ v.c = some z ∧ rnAdd x y = z

/-- **The add is exact on representable sums.** -/
def RnAddExact : Prop := ∀ x y : ℚ, Representable (x + y) → rnAdd x y = x + y

/-- **The add is monotone** in the exact sum. -/
def RnAddMono : Prop := ∀ x y x' y' : ℚ, x + y ≤ x' + y' → rnAdd x y ≤ rnAdd x' y'

/-- **Absorption**: a summand `i` below half an ulp of a nonzero representable `r` leaves `r`, if `r` is not a power of
two or `i` does not point toward zero. -/
def RnAddAbsorb : Prop :=
  ∀ r i : ℚ, Representable r → r ≠ 0 → 2 * |i| < ulp r → (2 ^ binade |r| < |r| ∨ 0 ≤ r * i) → rnAdd r i = r

/-- **Movement**: a summand of at least an ulp of a nonzero representable `r` changes `r`. -/
def RnAddMoves : Prop := ∀ r i : ℚ, Representable r → r ≠ 0 → ulp r ≤ |i| → rnAdd r i ≠ r

/-- **FP32 words are representable**: every finite word's value has a 24-bit significand. -/
def WordsRepresentable : Prop := ∀ (w : Nat) (x : ℚ), wordQ w = some x → Representable x

end Pouw.Fp8Atom
